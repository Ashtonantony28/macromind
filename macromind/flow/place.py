"""Run a MacroMind placer on a real floorplan DEF and write the placement.

Called from the ORFS hook (``macromind/flow/orfs_hook.tcl``) inside stage
``2_2_floorplan_macro``, but usable stand-alone::

    python -m macromind.flow.place --def 2_1_floorplan.def --lef tech.lef cells.lef ram.lef \\
        --method sa --seed 0 --halo 10 --out placed.tcl        # place_macro commands
    python -m macromind.flow.place ... --out placed.def        # DEF with FIXED macros

Pipeline: DEF+LEF -> Design -> std-cell clustering (HEM, seed 0) -> placer on
the clustered design (``random``/``grid``/``sa``/``rl``; all go through the
shared legaliser) -> proxy score (cluster pass + TILOS proxy) -> TCL/DEF.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

METHODS = ("random", "grid", "sa", "rl")


def run(def_path: str, lefs: list[str], method: str, seed: int = 0, *, budget: int | None = None, halo_um: float = 0.0,
        n_clusters: int | None = None, out: str | None = None, checkpoint: str | None = None) -> dict:
    from ..baselines.placers import get_placer
    from .emit import write_placement
    from .score import load_reduced, placement_record, score_placement

    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    flat, red, stats = load_reduced(def_path, lefs, halo_um=halo_um, n_clusters=n_clusters)
    if not red.macros:
        raise RuntimeError(f"{def_path}: no macros (LEF CLASS BLOCK) found; unknown masters={stats['unknown_masters']}")
    t = time.perf_counter()
    if method == "rl":
        p = get_placer("rl")(red, seed=seed, budget=budget, checkpoint=checkpoint)
    else:
        p = get_placer(method)(red, seed=seed, budget=budget)
    place_s = time.perf_counter() - t
    score = score_placement(red, p)
    fmt = None
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fmt = write_placement(out, flat, p, src_def=def_path)
    meta = {k: v for k, v in p.meta.items() if isinstance(v, (int, float, str, bool)) or v is None}
    return {
        "design": flat.name, "method": method, "seed": seed, "budget": budget, "halo_um": halo_um,
        "place_s": round(place_s, 3), "out": out, "format": fmt, "clustering": stats, "proxy": score,
        "placer_meta": meta, "macros": placement_record(flat, p),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m macromind.flow.place", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--def", dest="def_path", required=True, help="floorplan DEF (macros unplaced or placed)")
    ap.add_argument("--lef", nargs="+", required=True, help="tech + cell + macro LEFs")
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=None, help="placer budget (SA proxy evaluations / RL samples)")
    ap.add_argument("--halo", type=float, default=0.0, help="min macro-to-macro spacing in um (ORFS MACRO_PLACE_HALO)")
    ap.add_argument("--clusters", type=int, default=None, help="std-cell clusters (default n_cells/40, 16..1000)")
    ap.add_argument("--checkpoint", default=None, help="RL checkpoint (default $MACROMIND_RL_CKPT or runs/rl-v1/policy.pt)")
    ap.add_argument("--out", required=True, help="output .tcl (place_macro commands) or .def")
    ap.add_argument("--json", dest="json_out", default=None, help="write a JSON record here")
    a = ap.parse_args(argv)
    t = time.perf_counter()
    rec = run(a.def_path, a.lef, a.method, a.seed, budget=a.budget, halo_um=a.halo, n_clusters=a.clusters, out=a.out,
              checkpoint=a.checkpoint)
    rec["total_s"] = round(time.perf_counter() - t, 3)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(rec, indent=1))
    pr = rec["proxy"]
    print(f"macromind.flow.place: {rec['design']} method={a.method} seed={a.seed} macros={len(rec['macros'])} "
          f"proxy={pr['proxy']:.5f} legal={pr['legal']} place={rec['place_s']:.1f}s total={rec['total_s']:.1f}s -> {a.out}")
    return 0 if pr["legal"] else 3


if __name__ == "__main__":
    sys.exit(main())
