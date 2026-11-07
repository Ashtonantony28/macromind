"""MacroMind evaluation harness CLI.

    python -m macromind.harness run --design nangate45/bp_fe_top --placer macromind-sa --seed 0 [--stop-after grt]
    python -m macromind.harness sweep --design nangate45/bp_fe_top --placer rtlmp sa grid --seeds 0 1 --max-concurrency 2
    python -m macromind.harness key --design nangate45/gcd --placer rtlmp       # print cache key / run dir
    python -m macromind.harness import-base --design nangate45/tinyRocket --work-home runs/orfs/smoke

Placers: ``rtlmp`` (ORFS default hier-RTLMP) or ``macromind-<random|grid|sa|rl>``
(a bare method name is accepted). Each job runs in its own WORK_HOME under
runs/harness/jobs, is cached by content hash and writes ``ppa.json``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..flow.orfs import DesignSpec
from .runner import Job, ensure_base, import_base, run_job


def _summary(rec: dict) -> str:
    j, p = rec.get("job", {}), rec.get("ppa", {}) or {}
    pl = rec.get("placement", {}) or {}
    f = lambda v, fmt: (fmt.format(v) if isinstance(v, (int, float)) else "n/a")  # noqa: E731
    return (f"{j.get('design')} {j.get('placer')} s{j.get('seed')} [{rec.get('status')}{', cached' if rec.get('cached') else ''}] "
            f"WL={f(p.get('routed_wl_um') or p.get('grt_wl_um'), '{:.0f}')} WNS={f(p.get('wns_ns'), '{:.3f}')} "
            f"TNS={f(p.get('tns_ns'), '{:.2f}')} P={f(p.get('power_w'), '{:.4f}')} DRC={p.get('drc_errors')} "
            f"OVF={p.get('grt_overflow')} proxy={f(pl.get('proxy'), '{:.4f}')} wall={rec.get('wall_s')}s -> {rec.get('run_dir')}")


def _add_job_args(ap: argparse.ArgumentParser, multi: bool) -> None:
    n = "+" if multi else None
    ap.add_argument("--design", required=True, nargs=n, help="<platform>/<design>, e.g. nangate45/bp_fe_top")
    ap.add_argument("--placer", required=True, nargs=n, help="rtlmp | macromind-{random,grid,sa,rl}")
    if multi:
        ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    else:
        ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stop-after", default="finish", choices=["floorplan", "place", "cts", "grt", "route", "finish"])
    ap.add_argument("--budget", type=int, default=None, help="MacroMind placer budget")
    ap.add_argument("--num-cores", type=int, default=2, help="ORFS NUM_CORES per job")
    ap.add_argument("--timeout", type=float, default=None, help="per-job make timeout (s)")
    ap.add_argument("--rl-ckpt", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="VAR=VAL", help="extra ORFS variable (repeatable)")
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    ap.add_argument("--resume", action="store_true", help="continue a failed/interrupted job in place (make resumes)")


def _extra(a) -> dict:
    out = {}
    for s in a.set:
        k, _, v = s.partition("=")
        out[k.strip()] = v.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m macromind.harness", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one job")
    _add_job_args(r, multi=False)
    r.add_argument("--json", action="store_true", help="print the full PPA record")
    k = sub.add_parser("key", help="print a job's cache key and run dir")
    _add_job_args(k, multi=False)
    s = sub.add_parser("sweep", help="design x placer x seed fan-out")
    _add_job_args(s, multi=True)
    s.add_argument("--max-concurrency", type=int, default=1)
    s.add_argument("--dry-run", action="store_true")
    b = sub.add_parser("import-base", help="adopt an existing WORK_HOME (synth + 2_1 done) as a design's base")
    b.add_argument("--design", required=True)
    b.add_argument("--work-home", required=True)
    rr = sub.add_parser("resume-dir", help="continue a failed/interrupted job in its existing run dir")
    rr.add_argument("run_dir")
    bb = sub.add_parser("base", help="build a design's base (synth + 2_1_floorplan)")
    bb.add_argument("--design", required=True)
    bb.add_argument("--num-cores", type=int, default=2)
    a = ap.parse_args(argv)

    if a.cmd == "import-base":
        print(import_base(Path(a.work_home).resolve(), DesignSpec.parse(a.design)))
        return 0
    if a.cmd == "resume-dir":
        from .runner import resume_dir

        rec = resume_dir(a.run_dir)
        print(_summary(rec))
        return 0 if rec.get("status") == "ok" else 1
    if a.cmd == "base":
        print(ensure_base(DesignSpec.parse(a.design), num_cores=a.num_cores))
        return 0
    if a.cmd in ("run", "key"):
        job = Job(a.design, a.placer, a.seed, a.stop_after, a.budget, a.num_cores, _extra(a), a.timeout, a.rl_ckpt)
        if a.cmd == "key":
            print(json.dumps({"key": job.key(), "run_dir": str(job.run_dir()), "material": job.key_material()}, indent=1))
            return 0
        rec = run_job(job, force=a.force, resume=a.resume)
        print(json.dumps(rec, indent=1, default=str) if a.json else _summary(rec))
        return 0 if rec.get("status") == "ok" else 1

    jobs = [Job(d, p, sd, a.stop_after, a.budget, a.num_cores, _extra(a), a.timeout, a.rl_ckpt)
            for d, p, sd in itertools.product(a.design, a.placer, a.seeds)]
    if a.dry_run:
        for j in jobs:
            print(j.design, j.placer, j.seed, j.run_dir())
        return 0
    # build bases first (serially) so concurrent jobs do not race on synthesis
    for d in sorted({j.design for j in jobs}):
        ensure_base(DesignSpec.parse(d), num_cores=a.num_cores)
    bad = 0
    with ThreadPoolExecutor(max_workers=max(1, a.max_concurrency)) as ex:
        futs = {ex.submit(run_job, j, force=a.force, resume=a.resume): j for j in jobs}
        for f in as_completed(futs):
            try:
                rec = f.result()
            except Exception as e:  # noqa: BLE001
                j = futs[f]
                print(f"{j.design} {j.placer} s{j.seed} [crashed] {e}", flush=True)
                bad += 1
                continue
            bad += rec.get("status") != "ok"
            print(_summary(rec), flush=True)
    print(f"sweep: {len(jobs)} jobs, {bad} not ok")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
