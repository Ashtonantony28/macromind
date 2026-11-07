"""Isolated, content-hash-cached ORFS jobs: design x placer x seed -> PPA JSON.

Layout under ``runs/harness`` (override with ``$MACROMIND_HARNESS_DIR``)::

    base/<platform>_<design>-<basekey>/     shared synth + 2_1_floorplan (built once per design config)
    jobs/<platform>_<design>/<placer>-s<seed>-<stop>-<key>/
        config.mk        generated wrapper (includes the stock config, selects the placer)
        macromind/       hook outputs: 2_1_floorplan.def, macro_placement.tcl, placer.json/log
        results/ logs/ objects/ reports/   ORFS WORK_HOME (base stages copied in, mtimes kept)
        make.log  ppa.json

The cache key hashes the design's config files, the ORFS version, placer,
seed, budget, stop stage, extra make variables and (for MacroMind placers) the
MacroMind code version. A job whose ``ppa.json`` exists is not re-run unless
``force=True``; failed jobs are cached too (``status: failed``) so a sweep does
not retry them silently.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..flow.orfs import (
    REPO_ROOT, STOP_TARGETS, DesignSpec, PlacerSpec, code_version, design_digest, dumps_canonical, export_def_tcl,
    make_cmd, orfs_version, print_vars, tool_env, wrapper_config,
)
from . import metrics

SCHEMA = 1


def harness_dir() -> Path:
    return Path(os.environ.get("MACROMIND_HARNESS_DIR") or REPO_ROOT / "runs/harness")


@dataclass
class Job:
    design: str
    placer: str
    seed: int = 0
    stop_after: str = "finish"
    budget: int | None = None
    num_cores: int = 2
    extra: dict = field(default_factory=dict)
    timeout_s: float | None = None
    rl_ckpt: str | None = None

    def __post_init__(self) -> None:
        self.design = str(DesignSpec.parse(self.design))
        self.placer = PlacerSpec.parse(self.placer).name
        if self.stop_after not in STOP_TARGETS:
            raise ValueError(f"stop_after must be one of {sorted(STOP_TARGETS)}")

    @property
    def dspec(self) -> DesignSpec:
        return DesignSpec.parse(self.design)

    @property
    def pspec(self) -> PlacerSpec:
        return PlacerSpec.parse(self.placer)

    def key_material(self) -> dict:
        p = self.pspec
        mat = {
            "schema": SCHEMA, "design": self.design, "design_digest": design_digest(self.dspec), "orfs": orfs_version(),
            "placer": self.placer, "seed": int(self.seed), "stop_after": self.stop_after,
            "budget": self.budget, "extra": {k: str(v) for k, v in sorted(self.extra.items())},
        }
        if p.kind == "macromind":
            mat["code"] = code_version()
            if p.method == "rl":
                ck = Path(self.rl_ckpt or os.environ.get("MACROMIND_RL_CKPT") or REPO_ROOT / "runs/rl-v1/policy.pt")
                mat["rl_ckpt"] = hashlib.sha256(ck.read_bytes()).hexdigest()[:16] if ck.exists() else f"missing:{ck}"
        return mat

    def key(self) -> str:
        return hashlib.sha256(dumps_canonical(self.key_material()).encode()).hexdigest()[:12]

    def run_dir(self) -> Path:
        return harness_dir() / "jobs" / self.dspec.tag / f"{self.placer}-s{self.seed}-{self.stop_after}-{self.key()}"


# ----------------------------------------------------------------------------- base (synth + 2_1)
BASE_STAGES = ("1_", "2_1_")


def base_key(design: DesignSpec, extra: dict | None = None) -> str:
    mat = {"schema": SCHEMA, "design": str(design), "digest": design_digest(design), "orfs": orfs_version(),
           "extra": {k: str(v) for k, v in sorted((extra or {}).items())}}
    return hashlib.sha256(dumps_canonical(mat).encode()).hexdigest()[:12]


def _orfs_dirs(config: Path, work_home: Path) -> dict[str, Path]:
    v = print_vars(config, work_home, ["RESULTS_DIR", "LOG_DIR", "OBJECTS_DIR", "REPORTS_DIR"])
    return {k: Path(x) for k, x in v.items()}


def ensure_base(design: DesignSpec, *, num_cores: int = 2, extra: dict | None = None, log=print) -> Path:
    """Build (once, under a file lock) synthesis + ``2_1_floorplan`` for ``design``; return its WORK_HOME."""
    bdir = harness_dir() / "base" / f"{design.tag}-{base_key(design, extra)}"
    bdir.mkdir(parents=True, exist_ok=True)
    stamp = bdir / "base.json"
    with open(bdir / ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if stamp.exists():
            return bdir
        cfg = bdir / "config.mk"
        cfg.write_text(f"include {design.config_path()}\n" + "".join(f"export {k} = {v}\n" for k, v in sorted((extra or {}).items())))
        dirs = _orfs_dirs(cfg, bdir)
        target = str(dirs["RESULTS_DIR"] / "2_1_floorplan.odb")
        log(f"[harness] building base {design} in {bdir} (synth + 2_1_floorplan)")
        t0 = time.time()
        with open(bdir / "make.log", "w") as fh:
            rc = subprocess.run(make_cmd(cfg, bdir, target, num_cores=num_cores), stdout=fh, stderr=subprocess.STDOUT,
                                env=tool_env()).returncode
        if rc != 0:
            raise RuntimeError(f"base build for {design} failed (rc={rc}); see {bdir/'make.log'}")
        stamp.write_text(json.dumps({"design": str(design), "wall_s": round(time.time() - t0, 1),
                                     "dirs": {k: str(v) for k, v in dirs.items()}}, indent=1))
    return bdir


def import_base(src_work_home: Path, design: DesignSpec, *, extra: dict | None = None) -> Path:
    """Adopt an existing ORFS WORK_HOME that already has synth + 2_1 as this design's base."""
    bdir = harness_dir() / "base" / f"{design.tag}-{base_key(design, extra)}"
    if (bdir / "base.json").exists():
        return bdir
    bdir.mkdir(parents=True, exist_ok=True)
    cfg = bdir / "config.mk"
    cfg.write_text(f"include {design.config_path()}\n")
    sd, dd = _orfs_dirs(cfg, src_work_home), _orfs_dirs(cfg, bdir)
    for k in sd:
        _copy_stages(sd[k], dd[k], all_files=(k == "OBJECTS_DIR"))
    (bdir / "base.json").write_text(json.dumps({"design": str(design), "imported_from": str(src_work_home)}, indent=1))
    return bdir


def _copy_stages(src: Path, dst: Path, *, all_files: bool = False) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        keep = all_files or p.name.startswith(BASE_STAGES) or p.name in ("clock_period.txt", "mem.json", "synth_stat.txt",
                                                                         "synth_check.txt")
        if not keep:
            continue
        if p.is_dir():
            shutil.copytree(p, dst / p.name, symlinks=True, dirs_exist_ok=True, copy_function=shutil.copy2)
        else:
            shutil.copy2(p, dst / p.name)


# ----------------------------------------------------------------------------- job
def _tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def run_job(job: Job, *, force: bool = False, log=print) -> dict:
    """Run (or load from cache) one job; returns the PPA record (also written to ``ppa.json``)."""
    rd = job.run_dir()
    out = rd / "ppa.json"
    if out.exists() and not force:
        rec = json.loads(out.read_text())
        rec["cached"] = True
        return rec
    if rd.exists():
        shutil.rmtree(rd)
    rd.mkdir(parents=True)
    t_start = time.time()
    rec: dict = {"schema": SCHEMA, "job": asdict(job), "key": job.key(), "key_material": job.key_material(),
                 "run_dir": str(rd), "status": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        base = ensure_base(job.dspec, num_cores=job.num_cores, log=log)
        cfg = rd / "config.mk"
        cfg.write_text(wrapper_config(job.dspec, job.pspec, job.seed, rd / "macromind", budget=job.budget,
                                      rl_ckpt=job.rl_ckpt, extra=job.extra))
        bcfg = base / "config.mk"
        bd, jd = _orfs_dirs(bcfg, base), _orfs_dirs(cfg, rd)
        for k in bd:
            _copy_stages(bd[k], jd[k], all_files=(k == "OBJECTS_DIR"))
        lefs = print_vars(cfg, rd, ["TECH_LEF", "SC_LEF", "ADDITIONAL_LEFS", "MACRO_PLACE_HALO"])
        target = STOP_TARGETS[job.stop_after]
        log(f"[harness] {job.design} {job.placer} seed={job.seed} -> {target} in {rd}")
        t0 = time.time()
        timed_out = False
        with open(rd / "make.log", "w") as fh:
            try:
                rc = subprocess.run(make_cmd(cfg, rd, target, num_cores=job.num_cores), stdout=fh,
                                    stderr=subprocess.STDOUT, env=tool_env(), timeout=job.timeout_s).returncode
            except subprocess.TimeoutExpired:
                rc, timed_out = -9, True
        rec["make_wall_s"] = round(time.time() - t0, 1)
        rec["make_rc"] = rc
        rec["timed_out"] = timed_out
        rec.update(metrics.collect(jd["LOG_DIR"]))
        rec["lefs"] = lefs
        # placement actually used = macros in the post-2_2 ODB, scored the same way for every placer
        odb = jd["RESULTS_DIR"] / "2_2_floorplan_macro.odb"
        if odb.exists():
            rec["placement"] = _score_odb(odb, rd, lefs)
        pj = rd / "macromind" / "placer.json"
        if pj.exists():
            pr = json.loads(pj.read_text())
            rec["placer"] = {k: pr.get(k) for k in ("method", "seed", "budget", "halo_um", "place_s", "total_s", "proxy",
                                                    "placer_meta", "clustering")}
        rec["status"] = "ok" if rc == 0 else ("timeout" if timed_out else "failed")
        if rc != 0:
            rec["error_tail"] = _tail(rd / "make.log")
    except Exception as e:  # noqa: BLE001 -- failures are results
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["wall_s"] = round(time.time() - t_start, 1)
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out.write_text(json.dumps(rec, indent=1, default=str))
    rec["cached"] = False
    return rec


def _score_odb(odb: Path, rd: Path, lefs: dict) -> dict:
    from ..flow.score import score_def

    def_out = rd / "macromind" / "2_2_floorplan_macro.def"
    def_out.parent.mkdir(parents=True, exist_ok=True)
    tcl = rd / "macromind" / "export_def.tcl"
    tcl.write_text(export_def_tcl(odb, def_out))
    r = subprocess.run(["openroad", "-no_splash", "-exit", str(tcl)], capture_output=True, text=True, env=tool_env(),
                       timeout=600)
    if r.returncode != 0 or not def_out.exists():
        return {"error": f"write_def failed: {r.stdout[-300:]} {r.stderr[-300:]}"}
    lef_list = [lefs["TECH_LEF"], lefs["SC_LEF"], *lefs.get("ADDITIONAL_LEFS", "").split()]
    halo = max([float(x) for x in lefs.get("MACRO_PLACE_HALO", "0").split()] or [0.0])
    try:
        s = score_def(def_out, lef_list, halo_um=halo)
    except Exception as e:  # noqa: BLE001
        return {"error": f"score failed: {type(e).__name__}: {e}"}
    return s
