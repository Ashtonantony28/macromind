"""Training-data generation for the predictors: diverse macro placements -> ORFS GRT labels.

For each design we write a deterministic *plan* (a list of placement specs: SA at
varied budgets / temperatures / proxy weights, random legal, shelf packing,
perturbations of the RTLMP and SA placements, the RTLMP placement replayed).
Each spec is turned into a ``place_macro`` TCL file offline (on the same
floorplan DEF the in-flow hook sees), and the harness runs the stock ORFS flow
to ``5_1_grt`` with

* ``MACRO_PLACEMENT_TCL = <that file>`` -- ORFS's own hook replays it in stage 2_2;
* ``PRE_GLOBAL_ROUTE = macromind/predict/grt_labels.tcl`` -- wraps ORFS's final
  GRT ``report_metrics`` to dump every setup endpoint's slack (post repair, GRT
  parasitics);

and afterwards the per-layer gcell capacity/usage is read out of ``5_1_grt.odb``
(:mod:`macromind.predict.odb_gcells`). No ORFS file is patched. Labels per
sample (``<run dir>/labels/``): ``ppa.json`` scalars (GRT WL, WNS, TNS, power,
overflow), ``gcells.npz`` (H/V capacity + usage maps), ``endpoints.json``
(endpoint -> slack). Heavy ORFS results are pruned after extraction (the disk
is small), so a finished sample takes about 1 MB.

Resumable: a sample is done when its run dir has ``ppa.json`` and
``labels/done.json``; re-running ``run`` skips done samples, re-extracts labels
for runs whose ORFS part finished, and restarts interrupted ones.

CLI::

    python -m macromind.predict.datagen plan   --design bp_fe_top [--n 48]
    python -m macromind.predict.datagen run    --design bp_fe_top tinyRocket [--max-jobs 10] [--concurrency 2]
    python -m macromind.predict.datagen retry-congested              # re-run GRT-overflow failures with -allow_congestion
    python -m macromind.predict.datagen legacy                       # label existing harness GRT runs
    python -m macromind.predict.datagen status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..flow.orfs import REPO_ROOT, DesignSpec, tool_env

DATAGEN_DIR = REPO_ROOT / "runs" / "datagen"
LABEL_TCL = Path(__file__).resolve().parent / "grt_labels.tcl"
GCELL_PY = Path(__file__).resolve().parent / "odb_gcells.py"
DESIGNS = {"bp_fe_top": "nangate45/bp_fe_top", "tinyRocket": "nangate45/tinyRocket"}


def datagen_dir() -> Path:
    return Path(os.environ.get("MACROMIND_DATAGEN_DIR") or DATAGEN_DIR).resolve()


# ----------------------------------------------------------------------------- design context
@dataclass
class DesignContext:
    """Everything needed to generate placements for one design offline."""

    name: str
    spec: DesignSpec
    def_path: Path
    lefs: list[str]
    halo_um: float

    def load(self, n_clusters: int | None = None):
        """``(flat, reduced, stats)`` designs (cached per process)."""
        key = (str(self.def_path), n_clusters)
        if key not in _LOAD_CACHE:
            from ..flow.score import load_reduced

            _LOAD_CACHE[key] = load_reduced(self.def_path, self.lefs, halo_um=self.halo_um, n_clusters=n_clusters)
        return _LOAD_CACHE[key]


_LOAD_CACHE: dict = {}


def _harness_jobs(spec: DesignSpec) -> Path:
    return REPO_ROOT / "runs" / "harness" / "jobs" / spec.tag


def context(name: str) -> DesignContext:
    """Floorplan DEF + LEFs of ``name``; the DEF is taken from (or exported from) the harness base."""
    spec = DesignSpec.parse(DESIGNS.get(name, name))
    cdir = datagen_dir() / "ctx" / spec.tag
    cdir.mkdir(parents=True, exist_ok=True)
    meta_p = cdir / "context.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        return DesignContext(name, spec, Path(meta["def"]), meta["lefs"], meta["halo_um"])
    from ..harness.runner import ensure_base, _orfs_dirs
    from ..flow.orfs import print_vars

    base = ensure_base(spec)
    v = print_vars(base / "config.mk", base, ["TECH_LEF", "SC_LEF", "ADDITIONAL_LEFS", "MACRO_PLACE_HALO"])
    lefs = [v["TECH_LEF"], v["SC_LEF"], *v.get("ADDITIONAL_LEFS", "").split()]
    halo = max([float(x) for x in v.get("MACRO_PLACE_HALO", "0").split()] or [0.0])
    odb = _orfs_dirs(base / "config.mk", base)["RESULTS_DIR"] / "2_1_floorplan.odb"
    def_path = cdir / "2_1_floorplan.def"
    tcl = cdir / "export.tcl"
    tcl.write_text(f"read_db {{{odb}}}\nwrite_def {{{def_path}}}\n")
    r = subprocess.run(["openroad", "-no_splash", "-exit", str(tcl)], capture_output=True, text=True, env=tool_env(), timeout=900)
    if r.returncode != 0 or not def_path.exists():
        raise RuntimeError(f"exporting {odb} failed: {r.stdout[-500:]} {r.stderr[-500:]}")
    meta_p.write_text(json.dumps({"def": str(def_path), "lefs": lefs, "halo_um": halo}, indent=1))
    return DesignContext(name, spec, def_path, lefs, halo)


def rtlmp_placement_tcl(spec: DesignSpec) -> Path | None:
    """The RTLMP placement (``2_2_floorplan_macro.tcl``) of an existing harness rtlmp run, if any."""
    for p in sorted(_harness_jobs(spec).glob("rtlmp-s0-*/results/*/*/*/2_2_floorplan_macro.tcl")):
        return p
    return None


# ----------------------------------------------------------------------------- plans
def make_plan(name: str, n: int | None = None, seed: int = 0) -> list[dict]:
    """Deterministic list of placement specs for ``name`` (ids are stable; order interleaves kinds)."""
    rng = np.random.default_rng(1000 + seed)
    big = name == "bp_fe_top"
    specs: list[dict] = [{"kind": "replay_rtlmp"}]
    n_sa, n_rand, n_grid, n_prt, n_psa = (16, 8, 3, 12, 6) if big else (8, 8, 2, 8, 0)
    for k in range(n_sa):
        specs.append({"kind": "sa", "seed": 100 + k, "budget": int(rng.choice([300, 1000, 3000, 10000, 20000])),
                      "accept_init": float(rng.choice([0.2, 0.5, 0.9])), "w_density": float(rng.choice([0.1, 0.5, 1.5])),
                      "w_congestion": float(rng.choice([0.0, 0.5, 2.0])), "init": str(rng.choice(["grid", "random"]))})
    for k in range(n_rand):
        specs.append({"kind": "random", "seed": 200 + k})
    for k in range(n_grid):
        specs.append({"kind": "grid", "seed": 300 + k})
    for k in range(n_prt):
        specs.append({"kind": "perturb", "base": "rtlmp", "seed": 400 + k, "sigma": float(rng.choice([0.02, 0.05, 0.1, 0.2])),
                      "swaps": int(rng.integers(0, 3 if big else 2)), "p_flip": float(rng.choice([0.0, 0.3]))})
    for k in range(n_psa):
        specs.append({"kind": "perturb", "base": "sa", "seed": 500 + k, "sigma": float(rng.choice([0.03, 0.1])),
                      "swaps": int(rng.integers(0, 3)), "p_flip": 0.2})
    # interleave kinds so a partial run is already diverse
    by_kind: dict[str, list[dict]] = {}
    for s in specs:
        by_kind.setdefault(s["kind"] + s.get("base", ""), []).append(s)
    order: list[dict] = []
    while any(by_kind.values()):
        for k in list(by_kind):
            if by_kind[k]:
                order.append(by_kind[k].pop(0))
    if n is not None:
        order = order[:n]
    for s in order:
        s["id"] = spec_id(s)
        s["design"] = name
    return order


def spec_id(spec: dict) -> str:
    s = {k: v for k, v in spec.items() if k not in ("id", "design")}
    h = hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()[:8]
    return f"{spec['kind']}{('-' + spec['base']) if 'base' in spec else ''}-{h}"


def plan_path(name: str) -> Path:
    return datagen_dir() / "plans" / f"{name}.json"


def load_or_make_plan(name: str, n: int | None = None) -> list[dict]:
    p = plan_path(name)
    if p.exists():
        plan = json.loads(p.read_text())
        return plan[:n] if n else plan
    plan = make_plan(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(plan, indent=1))
    return plan[:n] if n else plan


# ----------------------------------------------------------------------------- placement generation
def perturb(design, base, *, seed: int, sigma: float, swaps: int = 0, p_flip: float = 0.0):
    """Gaussian jitter (``sigma`` x core size), ``swaps`` random same-size-class swaps, random flips; legalised."""
    from ..legalize.legalizer import legalize

    rng = np.random.default_rng(seed)
    p = base.copy()
    n = len(p)
    mov = np.flatnonzero([not m.fixed for m in design.macros])
    c = design.core
    p.x[mov] = np.round(p.x[mov] + rng.normal(0, sigma * c.width, len(mov))).astype(np.int64)
    p.y[mov] = np.round(p.y[mov] + rng.normal(0, sigma * c.height, len(mov))).astype(np.int64)
    for _ in range(swaps):
        if len(mov) < 2:
            break
        i, j = rng.choice(mov, 2, replace=False)
        p.x[i], p.x[j] = p.x[j], p.x[i]
        p.y[i], p.y[j] = p.y[j], p.y[i]
    for i in mov:
        if rng.random() < p_flip:
            # mirror about x (MX: N<->FS) or y (MY: N<->FN) keeps the footprint
            p.orient[i] = int(p.orient[i]) ^ (6 if rng.random() < 0.5 else 4)
    p.x = np.clip(p.x, c.xl, c.xh)
    p.y = np.clip(p.y, c.yl, c.yh)
    out = legalize(design, p, choose_orientation=False)
    out.meta.update(method="perturb", seed=seed, sigma=sigma, swaps=swaps)
    assert len(out) == n
    return out


def generate_placement(ctx: DesignContext, spec: dict):
    """``(flat design, placement on the reduced design)`` for one plan entry."""
    from ..baselines.placers import SAConfig, grid_place, random_place, sa_place
    from ..cost.proxy import ProxyConfig
    from ..flow.emit import apply_um_placement, parse_place_macro_tcl

    flat, red, _ = ctx.load()
    kind = spec["kind"]

    def rtlmp():
        t = rtlmp_placement_tcl(ctx.spec)
        if t is None:
            raise FileNotFoundError(f"no rtlmp run for {ctx.spec} under {_harness_jobs(ctx.spec)}")
        return apply_um_placement(red, parse_place_macro_tcl(t.read_text()))

    def sa(s):
        cfg = SAConfig(budget=int(s.get("budget", 20000)), init=s.get("init", "grid"),
                       accept_init=float(s.get("accept_init", 0.5)),
                       proxy=ProxyConfig(w_density=float(s.get("w_density", 0.5)), w_congestion=float(s.get("w_congestion", 0.5))))
        return sa_place(red, seed=int(s.get("seed", 0)), config=cfg)

    if kind == "replay_rtlmp":
        p = rtlmp()
    elif kind == "sa":
        p = sa(spec)
    elif kind == "random":
        p = random_place(red, seed=spec["seed"])
    elif kind == "grid":
        p = grid_place(red, seed=spec["seed"])
    elif kind == "perturb":
        base = rtlmp() if spec["base"] == "rtlmp" else sa({"seed": 0, "budget": 20000})
        p = perturb(red, base, seed=spec["seed"], sigma=spec["sigma"], swaps=spec.get("swaps", 0), p_flip=spec.get("p_flip", 0.0))
    else:
        raise ValueError(f"unknown spec kind {kind!r}")
    return flat, p


# ----------------------------------------------------------------------------- jobs
def _job_cls():
    from ..harness.runner import Job, harness_dir

    @dataclass
    class DatagenJob(Job):
        """A harness job that replays a precomputed placement (``MACRO_PLACEMENT_TCL``) and dumps GRT labels."""

        sample_id: str = ""
        placement_sha: str = ""

        def key_material(self) -> dict:
            m = super().key_material()
            m.update(datagen=self.sample_id, placement_sha=self.placement_sha,
                     label_tcl=hashlib.sha256(LABEL_TCL.read_bytes()).hexdigest()[:12])
            return m

        def run_dir(self) -> Path:
            return harness_dir() / "jobs" / self.dspec.tag / f"{self.sample_id}-{self.key()}"

    return DatagenJob


def sample_dir(ctx: DesignContext, spec: dict) -> Path:
    return datagen_dir() / "placements" / ctx.spec.tag / spec["id"]


def prepare_sample(ctx: DesignContext, spec: dict) -> Path:
    """Write ``placement.tcl`` + ``spec.json`` for ``spec`` (idempotent)."""
    from ..flow.emit import write_tcl
    from ..flow.score import placement_record

    d = sample_dir(ctx, spec)
    tcl = d / "placement.tcl"
    if tcl.exists() and (d / "spec.json").exists():
        return tcl
    d.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    flat, p = generate_placement(ctx, spec)
    write_tcl(tcl, flat, p)
    meta = {k: v for k, v in p.meta.items() if isinstance(v, (int, float, str, bool)) or v is None}
    (d / "spec.json").write_text(json.dumps({"spec": spec, "gen_s": round(time.perf_counter() - t, 2), "placer_meta": meta,
                                             "macros": placement_record(flat, p)}, indent=1))
    return tcl


#: ORFS's default GLOBAL_ROUTE_ARGS plus -allow_congestion: used only to re-run placements whose
#: global route ended with overflow (GRT-0116), which ORFS otherwise turns into a failed stage
ALLOW_CONGESTION_ARGS = "-congestion_iterations 30 -congestion_report_iter_step 5 -verbose -allow_congestion"


def make_job(ctx: DesignContext, spec: dict, tcl: Path, *, num_cores: int = 2, timeout_s: float | None = 3 * 3600,
             allow_congestion: bool = False):
    J = _job_cls()
    sha = hashlib.sha256(tcl.read_bytes()).hexdigest()[:16]
    extra = {"MACRO_PLACEMENT_TCL": str(tcl), "MACRO_PLACEMENT": "", "PRE_GLOBAL_ROUTE": str(LABEL_TCL)}
    if allow_congestion:
        extra["GLOBAL_ROUTE_ARGS"] = ALLOW_CONGESTION_ARGS
    return J(design=str(ctx.spec), placer="rtlmp", seed=0, stop_after="grt", num_cores=num_cores, timeout_s=timeout_s,
             extra=extra, sample_id=spec["id"], placement_sha=sha)


def failure_kind(run_dir: Path) -> str | None:
    """``congested`` (GRT overflow, GRT-0116), ``pdn`` (PDN channel repair failed), ``other`` or None (ok / not run)."""
    pj = run_dir / "ppa.json"
    if not pj.exists():
        return None
    rec = json.loads(pj.read_text())
    if rec.get("status") == "ok":
        return None
    tail = rec.get("error_tail", "") + rec.get("error", "")
    if "GRT-0116" in tail:
        return "congested"
    if "PDN-0179" in tail:
        return "pdn"
    return "other"


def retry_congested(designs: list[str], *, num_cores: int = 2, log=print) -> list[dict]:
    """Re-run (with ``-allow_congestion``) every plan sample whose GRT failed only because of overflow."""
    os.environ["MACROMIND_HARNESS_DIR"] = str(datagen_dir())
    from ..harness.runner import run_job

    out = []
    for d in designs:
        ctx = context(d)
        for spec in load_or_make_plan(d):
            tcl = sample_dir(ctx, spec) / "placement.tcl"
            if not tcl.exists():
                continue
            rd = make_job(ctx, spec, tcl).run_dir()
            if failure_kind(rd) != "congested":
                continue
            job = make_job(ctx, spec, tcl, num_cores=num_cores, allow_congestion=True)
            rd2 = job.run_dir()
            if not sample_done(rd2):
                if not (rd2 / "ppa.json").exists():
                    rec = run_job(job, log=log)
                    log(f"[datagen] retry {d} {spec['id']} allow_congestion: {rec['status']} overflow={rec.get('ppa', {}).get('grt_overflow')}")
                if json.loads((rd2 / "ppa.json").read_text()).get("status") == "ok":
                    extract_labels(rd2)
            out.append({"design": d, "id": spec["id"], "run_dir": str(rd2), "done": sample_done(rd2)})
    return out


# ----------------------------------------------------------------------------- label extraction
def extract_gcells(odb: Path, out_npz: Path) -> dict:
    """Per-direction capacity/usage maps (gcell grid) of a GRT ODB -> ``gcells.npz``."""
    tmp = out_npz.with_suffix(".json")
    r = subprocess.run(["openroad", "-python", "-no_splash", "-exit", str(GCELL_PY), str(odb), str(tmp)], capture_output=True,
                       text=True, env=tool_env(), timeout=900)
    if r.returncode != 0 or not tmp.exists():
        raise RuntimeError(f"gcell dump failed for {odb}: {r.stdout[-400:]} {r.stderr[-400:]}")
    d = json.loads(tmp.read_text())
    arrs = gcell_arrays(d)
    np.savez_compressed(out_npz, **arrs)
    tmp.unlink()
    return {k: (v.shape if hasattr(v, "shape") else v) for k, v in arrs.items()}


def gcell_arrays(d: dict) -> dict[str, np.ndarray]:
    """Aggregate the per-layer JSON of :mod:`odb_gcells` into H/V capacity and usage maps ``(ny, nx)``."""
    if "error" in d or not d.get("layers"):
        raise ValueError(f"no gcell data: {d.get('error')}")
    cap = {"H": 0, "V": 0}
    use = {"H": 0, "V": 0}
    for v in d["layers"].values():
        k = "H" if v["direction"].upper().startswith("H") else "V"
        cap[k] = cap[k] + np.asarray(v["capacity"], np.float32)
        use[k] = use[k] + np.asarray(v["usage"], np.float32)
    return {"cap_h": np.asarray(cap["H"], np.float32), "cap_v": np.asarray(cap["V"], np.float32),
            "use_h": np.asarray(use["H"], np.float32), "use_v": np.asarray(use["V"], np.float32),
            "grid_x": np.asarray(d["grid_x"], np.int64), "grid_y": np.asarray(d["grid_y"], np.int64),
            "die": np.asarray(d["die"], np.int64), "dbu_per_micron": np.asarray(d["dbu_per_micron"])}


def extract_labels(run_dir: Path, *, prune: bool = True) -> dict:
    """Write ``run_dir/labels/{gcells.npz,endpoints.json,done.json}`` for a finished GRT run."""
    from .labels import parse_endpoint_slack

    run_dir = Path(run_dir)
    ld = run_dir / "labels"
    if (ld / "done.json").exists():
        return json.loads((ld / "done.json").read_text())
    ld.mkdir(exist_ok=True)
    odbs = sorted(run_dir.glob("results/*/*/*/5_1_grt.odb"))
    if not odbs:
        raise FileNotFoundError(f"{run_dir}: no 5_1_grt.odb")
    info: dict = {"gcells": extract_gcells(odbs[0], ld / "gcells.npz")}
    eps = sorted(run_dir.glob("reports/*/*/*/macromind_labels/endpoint_slack.rpt"))
    if eps:
        e = parse_endpoint_slack(eps[0].read_text())
        (ld / "endpoints.json").write_text(json.dumps(e))
        info["endpoints"] = len(e)
    info["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (ld / "done.json").write_text(json.dumps(info, default=str))
    if prune:
        prune_run(run_dir)
    return info


def prune_run(run_dir: Path) -> None:
    """Drop the heavy ORFS artefacts (ODBs, netlists, guides, objects, DEF dumps) of a labelled run."""
    for pat in ("results/*/*/*/*.odb", "results/*/*/*/*.v", "results/*/*/*/*.rtlil", "results/*/*/*/route.guide",
                "results/*/*/*/*.gds", "results/*/*/*/*.def", "results/*/*/*/*.spef", "results/*/*/*/mem.json",
                "macromind/*.def"):
        for p in run_dir.glob(pat):
            p.unlink(missing_ok=True)
    shutil.rmtree(run_dir / "objects", ignore_errors=True)


def sample_done(run_dir: Path) -> bool:
    return (run_dir / "ppa.json").exists() and (run_dir / "labels" / "done.json").exists()


# ----------------------------------------------------------------------------- runner
def tinygpu_active() -> bool:
    r = subprocess.run(["pgrep", "-f", "run_tinygpu"], capture_output=True, text=True)
    return bool(r.stdout.strip())


def orfs_jobs_active() -> int:
    """ORFS make processes of *other* MacroMind harness jobs (best effort)."""
    r = subprocess.run(["pgrep", "-fc", "make -C .*tools/ORFS/flow DESIGN_CONFIG=.*runs/(harness|datagen)/jobs"],
                       capture_output=True, text=True)
    try:
        return int(r.stdout.strip() or 0)
    except ValueError:
        return 0


def run_sample(ctx: DesignContext, spec: dict, *, num_cores: int = 2, log=print) -> dict:
    from ..harness.runner import harness_dir, run_job

    tcl = prepare_sample(ctx, spec)
    job = make_job(ctx, spec, tcl, num_cores=num_cores)
    rd = job.run_dir()
    if sample_done(rd):
        return {"id": spec["id"], "status": "cached", "run_dir": str(rd)}
    ppa = rd / "ppa.json"
    if ppa.exists() and json.loads(ppa.read_text()).get("status") != "ok":
        rec = json.loads(ppa.read_text())
        return {"id": spec["id"], "status": rec.get("status"), "run_dir": str(rd)}
    if not ppa.exists():
        assert harness_dir() == datagen_dir(), "set MACROMIND_HARNESS_DIR to the datagen dir"
        t = time.time()
        rec = run_job(job, log=log)
        log(f"[datagen] {ctx.name} {spec['id']} status={rec['status']} grt_wl={rec.get('ppa', {}).get('grt_wl_um')} "
            f"wns={rec.get('ppa', {}).get('wns_ns')} in {time.time() - t:.0f}s")
        if rec["status"] != "ok":
            return {"id": spec["id"], "status": rec["status"], "run_dir": str(rd)}
    info = extract_labels(rd)
    return {"id": spec["id"], "status": "ok", "run_dir": str(rd), **info}


def run(designs: list[str], *, max_jobs: int | None = None, concurrency: int = 2, num_cores: int = 2,
        respect_tinygpu: bool = True, log=print) -> list[dict]:
    """Run the plans of ``designs`` (interleaved) until done or ``max_jobs`` new ORFS jobs were started."""
    os.environ["MACROMIND_HARNESS_DIR"] = str(datagen_dir())
    base_link = datagen_dir() / "base"
    if not base_link.exists():  # share the harness bases (synth + 2_1), they do not depend on the placer
        base_link.parent.mkdir(parents=True, exist_ok=True)
        base_link.symlink_to(REPO_ROOT / "runs" / "harness" / "base")
    ctxs = {d: context(d) for d in designs}
    plans = {d: load_or_make_plan(d) for d in designs}
    queue: list[tuple[str, dict]] = []
    for i in range(max(len(p) for p in plans.values())):
        for d in designs:
            if i < len(plans[d]):
                queue.append((d, plans[d][i]))
    results: list[dict] = []
    lock = threading.Lock()
    state = {"started": 0, "active": 0}

    def allowed() -> int:
        return 1 if (respect_tinygpu and tinygpu_active()) else concurrency

    def worker(wid: int) -> None:
        while True:
            with lock:
                if not queue or (max_jobs is not None and state["started"] >= max_jobs):
                    return
            # wait for a slot
            while True:
                with lock:
                    if state["active"] < allowed():
                        state["active"] += 1
                        break
                time.sleep(30)
            with lock:
                if not queue or (max_jobs is not None and state["started"] >= max_jobs):
                    state["active"] -= 1
                    return
                d, spec = queue.pop(0)
            try:
                job = make_job(ctxs[d], spec, prepare_sample(ctxs[d], spec), num_cores=num_cores)
                fresh = not (job.run_dir() / "ppa.json").exists()
                if fresh:
                    with lock:
                        state["started"] += 1
                r = run_sample(ctxs[d], spec, num_cores=num_cores, log=log)
            except Exception as e:  # noqa: BLE001 -- keep going
                r = {"id": spec["id"], "status": "error", "error": f"{type(e).__name__}: {e}"}
                log(f"[datagen] {d} {spec['id']} ERROR {r['error']}")
            finally:
                with lock:
                    state["active"] -= 1
            r["design"] = d
            with lock:
                results.append(r)

    threads = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(max(1, concurrency))]
    for k, t in enumerate(threads):
        t.start()
        time.sleep(2 + 5 * k)
    for t in threads:
        t.join()
    return results


# ----------------------------------------------------------------------------- legacy harness runs
def legacy_runs() -> list[Path]:
    """Existing harness runs (any placer) that reached GRT and still have ``5_1_grt.odb``."""
    out = []
    for d in DESIGNS.values():
        for rd in sorted(_harness_jobs(DesignSpec.parse(d)).glob("*")):
            pj = rd / "ppa.json"
            if not pj.exists() or json.loads(pj.read_text()).get("status") != "ok":
                continue
            if (rd / "labels" / "done.json").exists() or list(rd.glob("results/*/*/*/5_1_grt.odb")):
                out.append(rd)
    return out


def label_legacy(log=print) -> list[dict]:
    """Extract gcell maps from existing harness runs (not pruned: those dirs belong to the harness)."""
    out = []
    for rd in legacy_runs():
        try:
            info = extract_labels(rd, prune=False)
            out.append({"run_dir": str(rd), "status": "ok", **info})
        except Exception as e:  # noqa: BLE001
            out.append({"run_dir": str(rd), "status": "error", "error": str(e)})
        log(f"[datagen] legacy {rd.name}: {out[-1]['status']}")
    return out


def status() -> dict:
    """Counts of planned / done / failed samples per design."""
    out = {}
    os.environ["MACROMIND_HARNESS_DIR"] = str(datagen_dir())
    for name in DESIGNS:
        p = plan_path(name)
        if not p.exists():
            continue
        plan = json.loads(p.read_text())
        jobs = datagen_dir() / "jobs" / DesignSpec.parse(DESIGNS[name]).tag
        done = running = 0
        fails: dict[str, int] = {}
        for s in plan:
            dirs = list(jobs.glob(f"{s['id']}-*"))
            if any(sample_done(d) for d in dirs):
                done += 1
            elif any((d / "ppa.json").exists() for d in dirs):
                k = next((failure_kind(d) for d in dirs if failure_kind(d)), "unlabelled")
                fails[k] = fails.get(k, 0) + 1
            elif dirs:
                running += 1
        out[name] = {"planned": len(plan), "done": done, "failed": fails, "started": running}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m macromind.predict.datagen", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan")
    a.add_argument("--design", nargs="+", default=list(DESIGNS))
    a.add_argument("--n", type=int, default=None)
    a.add_argument("--force", action="store_true")
    a = sub.add_parser("run")
    a.add_argument("--design", nargs="+", default=list(DESIGNS))
    a.add_argument("--max-jobs", type=int, default=None)
    a.add_argument("--concurrency", type=int, default=2)
    a.add_argument("--num-cores", type=int, default=2)
    a.add_argument("--ignore-tinygpu", action="store_true")
    a = sub.add_parser("retry-congested")
    a.add_argument("--design", nargs="+", default=list(DESIGNS))
    sub.add_parser("legacy")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        for d in a.design:
            p = plan_path(d)
            if p.exists() and not a.force:
                print(f"{p} exists ({len(json.loads(p.read_text()))} specs)")
                continue
            plan = make_plan(d, a.n)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(plan, indent=1))
            print(f"wrote {p}: {len(plan)} specs")
    elif a.cmd == "run":
        res = run(a.design, max_jobs=a.max_jobs, concurrency=a.concurrency, num_cores=a.num_cores,
                  respect_tinygpu=not a.ignore_tinygpu, log=lambda s: print(s, flush=True))
        print(json.dumps([{k: r.get(k) for k in ("design", "id", "status", "error")} for r in res], indent=1))
    elif a.cmd == "retry-congested":
        print(json.dumps(retry_congested(a.design, log=lambda s: print(s, flush=True)), indent=1))
    elif a.cmd == "legacy":
        label_legacy(log=lambda s: print(s, flush=True))
    elif a.cmd == "status":
        print(json.dumps(status(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
