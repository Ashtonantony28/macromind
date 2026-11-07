"""Parse ORFS per-stage metric JSONs and logs into a PPA record.

ORFS writes ``logs/<platform>/<nick>/<variant>/<stage>.json`` (flat
``<stage>__<metric>`` keys, the same data ``make metadata`` aggregates into
``metadata.json``) and ``<stage>.log`` whose last ``Elapsed time:`` line is the
GNU time summary of the stage. Global-route overflow is only in the GRT log
(the ``Total ... Max H / Max V / Total Overflow`` table).
"""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

_ELAPSED = re.compile(r"^Elapsed time:\s*(\S+)\[h:\]min:sec\..*?Peak memory:\s*(\d+)KB", re.M)
_GRT_TOTAL = re.compile(r"^Total\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*$", re.M)


def merge_stage_jsons(log_dir: str | Path) -> dict:
    """Merge every ``*.json`` in an ORFS log dir (later stages win on key clashes)."""
    m: dict = {}
    for f in sorted(glob.glob(os.path.join(str(log_dir), "*.json"))):
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(d, dict):
            m.update(d)
    return m


def parse_hms(s: str) -> float:
    total = 0.0
    for part in s.split(":"):
        total = total * 60 + float(part)
    return total


def stage_runtimes(log_dir: str | Path) -> dict[str, dict[str, float]]:
    """``{stage: {"s": seconds, "peak_mb": MB}}`` from the stage logs' GNU time lines (last one wins)."""
    out: dict[str, dict[str, float]] = {}
    for f in sorted(glob.glob(os.path.join(str(log_dir), "*.log"))):
        with open(f, errors="replace") as fh:
            text = fh.read()
        hits = _ELAPSED.findall(text)
        if not hits:
            continue
        t = sum(parse_hms(h[0]) for h in hits)  # some stages run several timed commands
        mem = max(int(h[1]) for h in hits) / 1024.0
        out[os.path.basename(f)[:-4]] = {"s": round(t, 2), "peak_mb": round(mem, 1)}
    return out


def grt_overflow(log_text: str) -> dict | None:
    """Last GRT congestion table total: ``{resource, demand, usage_pct, max_h, max_v, total}``."""
    hits = _GRT_TOTAL.findall(log_text)
    if not hits:
        return None
    r, d, u, mh, mv, tot = hits[-1]
    return {"resource": int(r), "demand": int(d), "usage_pct": float(u), "max_h": int(mh), "max_v": int(mv),
            "total": int(tot), "tables": len(hits)}


def _first(m: dict, *keys):
    for k in keys:
        if k in m and m[k] is not None:
            return m[k], k
    return None, None


#: (output name, candidate metric keys in priority order)
PPA_KEYS: list[tuple[str, tuple[str, ...]]] = [
    ("routed_wl_um", ("detailedroute__route__wirelength",)),
    ("grt_wl_um", ("globalroute__global_route__wirelength",)),
    ("est_wl_um", ("globalroute__route__wirelength__estimated", "cts__route__wirelength__estimated",
                   "detailedplace__route__wirelength__estimated")),
    ("wns_ns", ("finish__timing__setup__ws", "detailedroute__timing__setup__ws", "globalroute__timing__setup__ws",
                "cts__timing__setup__ws", "detailedplace__timing__setup__ws")),
    ("tns_ns", ("finish__timing__setup__tns", "detailedroute__timing__setup__tns", "globalroute__timing__setup__tns",
                "cts__timing__setup__tns", "detailedplace__timing__setup__tns")),
    ("hold_wns_ns", ("finish__timing__hold__ws", "globalroute__timing__hold__ws", "cts__timing__hold__ws")),
    ("power_w", ("finish__power__total", "globalroute__power__total", "cts__power__total", "detailedplace__power__total")),
    ("drc_errors", ("detailedroute__route__drc_errors",)),
    ("antenna_nets", ("detailedroute__antenna__violating__nets", "globalroute__antenna__violating__nets")),
    ("setup_violations", ("globalroute__timing__drv__setup_violation_count",)),
    ("macros", ("floorplan__design__instance__count__macros", "globalroute__design__instance__count__macros")),
    ("stdcells", ("finish__design__instance__count__stdcell", "globalroute__design__instance__count__stdcell",
                  "detailedplace__design__instance__count__stdcell")),
    ("utilization", ("finish__design__instance__utilization", "globalroute__design__instance__utilization",
                     "detailedplace__design__instance__utilization")),
    ("die_area_um2", ("finish__design__die__area", "floorplan__design__die__area")),
    ("rtlmp_wirelength", ("macro_place__wirelength",)),
    ("flow_errors", ("flow__errors__count",)),
]


def extract_ppa(m: dict) -> tuple[dict, dict]:
    """``(ppa, sources)``: headline metrics and the ORFS key each came from."""
    ppa, src = {}, {}
    for name, keys in PPA_KEYS:
        v, k = _first(m, *keys)
        ppa[name] = v
        if k:
            src[name] = k
    return ppa, src


def collect(log_dir: str | Path) -> dict:
    """Everything the harness stores from one ORFS run's log dir."""
    log_dir = Path(log_dir)
    merged = merge_stage_jsons(log_dir)
    ppa, src = extract_ppa(merged)
    grt_log = log_dir / "5_1_grt.log"
    ov = grt_overflow(grt_log.read_text(errors="replace")) if grt_log.exists() else None
    ppa["grt_overflow"] = ov["total"] if ov else None
    rt = stage_runtimes(log_dir)
    return {"ppa": ppa, "sources": src, "grt": ov, "stage_runtime": rt,
            "flow_runtime_s": round(sum(v["s"] for v in rt.values()), 1),
            # stages this job ran itself (synth + 2_1_floorplan come from the shared base)
            "job_runtime_s": round(sum(v["s"] for k, v in rt.items() if not k.startswith(("1_", "2_1_"))), 1),
            "orfs_metrics": merged}
