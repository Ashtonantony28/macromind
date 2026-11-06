"""CLI: run a baseline placer and print proxy-cost components.

Examples::

    python -m macromind.baselines --synthetic 50 --clusters 500 --method sa --seed 0
    python -m macromind.baselines --def floorplan.def --lef a.lef b.lef --method grid --out placed.def
    python -m macromind.baselines --bookshelf ibm01.aux --method random
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from ..core.synthetic import generate_synthetic
from ..cost.proxy import ProxyConfig, proxy_cost
from ..legalize.legalizer import check_legality
from .placers import PLACERS, SAConfig, sa_place


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m macromind.baselines", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", type=int, metavar="N", help="synthetic design with N macros")
    src.add_argument("--def", dest="def_path", help="DEF file (requires --lef)")
    src.add_argument("--bookshelf", help="Bookshelf .aux file")
    ap.add_argument("--lef", nargs="*", default=[], help="LEF files for --def")
    ap.add_argument("--clusters", type=int, default=None, help="synthetic: std-cell clusters (default 10*N)")
    ap.add_argument("--design-seed", type=int, default=0, help="synthetic: generator seed")
    ap.add_argument("--method", choices=sorted(PLACERS), default="sa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=None, help="proxy evaluations (SA)")
    ap.add_argument("--grid", type=int, default=32, help="cost grid bins along x")
    ap.add_argument("--out", help="write placed DEF (only with --def)")
    ap.add_argument("--json", action="store_true", help="print a JSON record instead of text")
    a = ap.parse_args(argv)

    if a.synthetic is not None:
        design = generate_synthetic(a.synthetic, a.clusters if a.clusters is not None else 10 * a.synthetic, seed=a.design_seed)
    elif a.def_path:
        from ..io.defio import read_def

        design = read_def(a.def_path, a.lef)
    else:
        from ..io.bookshelf import read_bookshelf

        design = read_bookshelf(a.bookshelf)

    cfg = ProxyConfig(grid_nx=a.grid)
    t = time.perf_counter()
    if a.method == "sa":
        placement = sa_place(design, a.seed, a.budget, SAConfig(proxy=cfg))
    else:
        placement = PLACERS[a.method](design, seed=a.seed, budget=a.budget)
    runtime = time.perf_counter() - t
    res = proxy_cost(design, placement, cfg)
    legal = not check_legality(design, placement)
    if a.out:
        if not a.def_path:
            ap.error("--out requires --def")
        from ..io.defio import write_def_placement

        write_def_placement(a.def_path, a.out, design, placement)

    record = {
        "design": design.name, "method": a.method, "seed": a.seed, "runtime_s": round(runtime, 3), "legal": legal,
        **res.as_dict(), **{k: v for k, v in placement.meta.items() if k in ("evaluations", "accepted")},
    }
    if a.json:
        print(json.dumps(record))
    else:
        print(design.summary())
        print(f"method={a.method} seed={a.seed} runtime={runtime:.2f}s legal={legal}")
        for k in ("proxy", "wirelength", "density", "congestion", "hpwl"):
            print(f"  {k:<11} {record[k]:.6g}")
        if "evaluations" in record:
            print(f"  evaluations {record['evaluations']} accepted {record['accepted']}")
    return 0 if legal else 1


if __name__ == "__main__":
    sys.exit(main())
