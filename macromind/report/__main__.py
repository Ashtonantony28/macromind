"""python -m macromind.report [--root runs/harness/jobs] [--out docs/results/table.md]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import build_report, load_records


def main(argv: list[str] | None = None) -> int:
    from ..harness.runner import harness_dir

    ap = argparse.ArgumentParser(prog="python -m macromind.report", description=__doc__)
    ap.add_argument("--root", default=None, help="directory searched for ppa.json (default runs/harness/jobs)")
    ap.add_argument("--design", nargs="*", default=None, help="only these designs (platform/design)")
    ap.add_argument("--title", default="MacroMind harness results")
    ap.add_argument("--out", default=None, help="write markdown here (default stdout)")
    a = ap.parse_args(argv)
    recs = load_records(a.root or harness_dir() / "jobs")
    if a.design:
        recs = [r for r in recs if r.get("job", {}).get("design") in a.design]
    md = build_report(recs, title=a.title)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
