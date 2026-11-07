#!/usr/bin/env python3
"""Summarise an ORFS run of designs/tinygpu (metrics JSON + reports) as a table / JSON.

Usage: python3 designs/tinygpu/report.py <WORK_HOME> <config> [--wall S] [--json OUT]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re


def stage_runtimes(logdir: str) -> dict[str, float]:
    out = {}
    for f in sorted(glob.glob(os.path.join(logdir, "*.log"))):
        el = None
        for line in open(f, errors="replace"):
            if line.startswith("Elapsed time:"):
                t = line.split()[2].replace("[h:]min:sec.", "")
                s = 0.0
                for p in t.split(":"):
                    s = s * 60 + float(p)
                el = s
        if el is not None:
            out[os.path.basename(f)[:-4]] = el
    return out


def synth_stats(repdir: str) -> dict:
    """Cell count / area / macro cells from ORFS synth_stat.txt."""
    f = os.path.join(repdir, "synth_stat.txt")
    if not os.path.exists(f):
        return {}
    txt = open(f).read()
    res: dict = {}
    m = re.search(r"Chip area for (?:top )?module.*?:\s*([\d.]+)", txt)
    if m:
        res["synth_area_um2"] = float(m.group(1))
    m = re.findall(r"^\s+([\d]+)\s+[\d.E+-]*\s*cells$", txt, re.M) or re.findall(r"Number of cells:\s+(\d+)", txt)
    if m:
        res["synth_cells"] = int(m[-1])
    macros: dict[str, int] = {}
    for name, n in re.findall(r"(fakeram45_\w+)\s+(\d+)", txt) + [(b, a) for a, b in re.findall(r"(\d+)\s+(?:[\d.]+\s+)?(fakeram45_\w+)", txt)]:
        macros[name] = max(macros.get(name, 0), int(n))
    res["synth_macros"] = macros
    res["synth_macro_count"] = sum(macros.values())
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("work_home")
    ap.add_argument("config")
    ap.add_argument("--wall", type=int)
    ap.add_argument("--json")
    a = ap.parse_args()
    d = f"tinygpu_{a.config}"
    logdir = os.path.join(a.work_home, "logs", "nangate45", d, "base")
    repdir = os.path.join(a.work_home, "reports", "nangate45", d, "base")
    m: dict = {}
    for f in sorted(glob.glob(os.path.join(logdir, "*.json"))):
        try:
            m.update(json.load(open(f)))
        except Exception:  # noqa: BLE001
            pass

    def first(*keys):
        for k in keys:
            if k in m:
                return m[k]
        return None

    stages = ("finish", "detailedroute", "globalroute", "cts", "detailedplace", "globalplace", "floorplan")
    res = {
        "config": a.config,
        **synth_stats(repdir),
        "macros": first(*(f"{s}__design__instance__count__macros" for s in stages)),
        "stdcells": first(*(f"{s}__design__instance__count__stdcell" for s in stages)),
        "stdcell_area_um2": first(*(f"{s}__design__instance__area__stdcell" for s in stages)),
        "macro_area_um2": first(*(f"{s}__design__instance__area__macros" for s in stages)),
        "die_area_um2": first(*(f"{s}__design__die__area" for s in stages)),
        "core_util": first(*(f"{s}__design__instance__utilization" for s in stages)),
        "wns_ns": first(*(f"{s}__timing__setup__ws" for s in stages)),
        "tns_ns": first(*(f"{s}__timing__setup__tns" for s in stages)),
        "grt_wirelength_um": first("globalroute__route__wirelength"),
        "routed_wirelength_um": first("detailedroute__route__wirelength"),
        "drc_errors": first("detailedroute__route__drc_errors"),
        "power_w": first("finish__power__total"),
        "stage_runtime_s": stage_runtimes(logdir),
        "make_wall_s": a.wall,
    }
    for k, v in res.items():
        if k != "stage_runtime_s":
            print(f"  {k:22s} {v}")
    for s, e in res["stage_runtime_s"].items():
        print(f"  {s:28s} {e:8.1f} s")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
