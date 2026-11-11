"""Results report: aggregate harness ``ppa.json`` records into markdown tables (mean +- 95% CI over seeds)."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

#: (column header, getter on a record)
COLUMNS: list[tuple[str, str]] = [
    ("routed WL (um)", "ppa.routed_wl_um"),
    ("GRT WL (um)", "ppa.grt_wl_um"),
    ("WNS (ns)", "ppa.wns_ns"),
    ("TNS (ns)", "ppa.tns_ns"),
    ("power (mW)", "ppa.power_mw"),
    ("DRC", "ppa.drc_errors"),
    ("GRT overflow", "ppa.grt_overflow"),
    ("GRT hot gcells (%)", "hot_pct"),
    ("proxy", "placement.proxy"),
    ("placer time (s)", "placer_s"),
    ("flow time (min)", "flow_min"),
]


def _get(rec: dict, path: str):
    if path == "ppa.power_mw":
        v = _get(rec, "ppa.power_w")
        return None if v is None else 1e3 * v
    if path == "hot_pct":  # share of GRT gcells at >= 90% utilisation (harness label-prune)
        v = (rec.get("grt_congestion") or {}).get("hot_frac")
        return None if v is None else 100.0 * v
    if path == "placer_s":
        # the macro placer itself: MacroMind = placer subprocess (DEF read + clustering + placement);
        # RTLMP = the whole 2_2_floorplan_macro stage
        if (rec.get("job") or {}).get("placer") == "rtlmp":
            return ((rec.get("stage_runtime") or {}).get("2_2_floorplan_macro") or {}).get("s")
        return (rec.get("placer") or {}).get("total_s")
    if path == "flow_min":
        v = rec.get("make_wall_s")
        return None if v is None else v / 60.0
    cur = rec
    for k in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur if isinstance(cur, (int, float)) and not isinstance(cur, bool) else None


def t_crit(df: int) -> float:
    """Two-sided 95% Student-t critical value."""
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, df))
    except Exception:  # noqa: BLE001
        table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
        return table.get(df, 1.96)


def mean_ci(xs: list[float]) -> tuple[float, float | None, int]:
    """``(mean, half-width of the 95% CI or None for n<2, n)``."""
    n = len(xs)
    if n == 0:
        return math.nan, None, 0
    m = sum(xs) / n
    if n < 2:
        return m, None, n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, t_crit(n - 1) * sd / math.sqrt(n), n


def _fmt(v: float) -> str:
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 0.1:
        return f"{v:.3f}"
    return f"{v:.4f}"


def fmt_ci(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    m, h, n = mean_ci(xs)
    return _fmt(m) if h is None else f"{_fmt(m)} ± {_fmt(h)}"


def load_records(root: str | Path) -> list[dict]:
    recs = []
    for p in sorted(Path(root).rglob("ppa.json")):
        try:
            recs.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    return recs


def dedupe(recs: list[dict]) -> list[dict]:
    """One record per (design, placer, seed, stop, budget, extra): runs keyed by older code versions are
    superseded by the most recently finished one."""
    best: dict[str, dict] = {}
    for r in recs:
        j = r.get("job", {})
        ident = json.dumps([j.get("design"), label(r), j.get("seed"), j.get("stop_after"), j.get("budget"),
                            j.get("extra")], sort_keys=True, default=str)
        if ident not in best or str(r.get("finished", "")) >= str(best[ident].get("finished", "")):
            best[ident] = r
    return list(best.values())


def label(rec: dict) -> str:
    """Report name of a record's placer: the job tag (e.g. ``rl-v2-proxy``) or the placer name."""
    j = rec.get("job", {}) or {}
    return j.get("tag") or j.get("placer") or "?"


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    try:
        from scipy.stats import spearmanr

        r = spearmanr(xs, ys).statistic
        return None if r != r else float(r)
    except Exception:  # noqa: BLE001
        return None


def build_report(recs: list[dict], *, title: str = "MacroMind harness results") -> str:
    recs = dedupe(recs)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        j = r.get("job", {})
        groups[(j.get("design"), j.get("stop_after"), label(r))].append(r)
    lines = [f"# {title}", "", f"{len(recs)} job records. Cells are mean ± 95% CI (Student t) over seeds with status ok; "
             "a single value means one seed.", ""]
    by_design: dict[tuple, list] = defaultdict(list)
    for (d, s, p), rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        by_design[(d, s)].append((p, rs))
    for (d, s), items in by_design.items():
        lines += [f"## {d} (stop after {s})", ""]
        lines.append("| placer | ok/runs | seeds | " + " | ".join(c for c, _ in COLUMNS) + " |")
        lines.append("|---" * (3 + len(COLUMNS)) + "|")
        allx, ally = [], []
        for p, rs in items:
            ok = [r for r in rs if r.get("status") == "ok"]
            seeds = sorted({r.get("job", {}).get("seed") for r in ok})
            cells = []
            for _, path in COLUMNS:
                cells.append(fmt_ci([v for v in (_get(r, path) for r in ok) if v is not None]))
            lines.append(f"| {p} | {len(ok)}/{len(rs)} | {','.join(map(str, seeds))} | " + " | ".join(cells) + " |")
            for r in ok:
                x = _get(r, "placement.proxy")
                y = _get(r, "ppa.routed_wl_um") or _get(r, "ppa.grt_wl_um")
                if x is not None and y is not None:
                    allx.append(x)
                    ally.append(y)
        bad = [r for _, rs in items for r in rs if r.get("status") != "ok"]
        for r in bad:
            j = r.get("job", {})
            err = r.get("error") or (r.get("error_tail") or "").strip().splitlines()[-1:] or [""]
            lines.append("")
            lines.append(f"- {j.get('placer')} seed {j.get('seed')}: **{r.get('status')}** "
                         f"({err if isinstance(err, str) else err[0]})")
        rho = spearman(allx, ally)
        if rho is not None:
            lines += ["", f"Spearman(proxy, wirelength) over {len(allx)} runs: {rho:+.2f}"]
        lines.append("")
    return "\n".join(lines)
