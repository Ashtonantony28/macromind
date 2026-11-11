"""SG10 analysis: baseline-vs-RL tables, deltas vs RTLMP, the predictor-reward ablation, proxy/predictor-vs-real
rank correlations and compute accounting, from the harness cache (``ppa.json``) and the RL v2 fine-tuning runs.

    python -m macromind.report.sg10 [--out docs/results/report-v1-data.md] [--no-predictor]

Every number comes from cached harness records; nothing is re-run. The predictor columns re-evaluate the SG8
predictor bundle on the macro placements found in each job's post-2_2 ODB (``ppa.json["placement"]["macros"]``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from . import _fmt, dedupe, fmt_ci, label, load_records, mean_ci

ROOT = Path(__file__).resolve().parents[2]
DESIGNS = {"nangate45/bp_fe_top": "bp_fe_top", "nangate45/tinygpu_c4": "tinygpu_c4"}
ORDER = ["rtlmp", "macromind-sa", "macromind-random", "macromind-grid", "macromind-rl", "rl-v1", "rl-v2-proxy",
         "rl-v2-pred"]
NAMES = {"macromind-rl": "rl-v1 greedy (SG7)", "rl-v1": "rl-v1 zero-shot (best-of-64)",
         "rl-v2-proxy": "rl-v2-proxy (best-of-64)", "rl-v2-pred": "rl-v2-pred (best-of-64)",
         "macromind-sa": "sa (20k evals)", "macromind-random": "random", "macromind-grid": "grid"}
PRED_KEY = {"bp_fe_top": "bp_fe_top", "tinygpu_c4": None}  # predictor training statistics per design


def g(rec: dict, path: str):
    from . import _get

    if path == "hot_frac":
        v = (rec.get("grt_congestion") or {}).get("hot_frac")
        return None if v is None else float(v)
    return _get(rec, path)


def _order(lbl: str) -> int:
    return ORDER.index(lbl) if lbl in ORDER else len(ORDER)


def _vals(rs, path):
    return [v for v in (g(r, path) for r in rs) if v is not None]


def kendall(x, y):
    if len(x) < 3:
        return None
    from scipy.stats import kendalltau

    t = kendalltau(x, y).statistic
    return None if t != t else float(t)


def spearman(x, y):
    if len(x) < 3:
        return None
    from scipy.stats import spearmanr

    r = spearmanr(x, y).statistic
    return None if r != r else float(r)


def welch(a, b):
    """(mean(a) - mean(b), 95% CI half-width, p) with Welch's t; None when n < 2 on either side."""
    if len(a) < 2 or len(b) < 2:
        return None
    from scipy import stats

    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return ma - mb, 0.0, (1.0 if ma == mb else 0.0)
    df = se ** 4 / ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
    t = (ma - mb) / se
    return ma - mb, float(stats.t.ppf(0.975, df)) * se, float(2 * stats.t.sf(abs(t), df))


# ----------------------------------------------------------------------------- predictor on harness placements
def _predictions(recs: list[dict], design: str, cache_path: Path) -> dict[str, dict]:
    """``{run_dir: {"grt_wl_um", "hot_frac", ...}}`` from the SG8 predictor on each run's post-2_2 placement."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [r for r in recs if r["run_dir"] not in cache and (r.get("placement") or {}).get("macros")]
    if todo:
        import torch

        from ..flow.emit import OPENDB_ORIENT
        from ..flow.score import load_reduced
        from ..predict.runtime import Predictor

        torch.set_num_threads(1)
        d = ROOT / "runs/rl-v2" / design
        halo = {"bp_fe_top": 10.0, "tinygpu_c4": 8.0}[design]
        _, red, _ = load_reduced(d / "floorplan.def", (d / "lefs.txt").read_text().split(), halo_um=halo)
        pred = Predictor.load(ROOT / "runs/predict/bundle.pt", w_congestion=1, w_timing=0, w_wl=1,
                              design_key=PRED_KEY[design])
        inv = {v: k for k, v in OPENDB_ORIENT.items()}
        idx = {m.name: i for i, m in enumerate(red.macros)}
        dbu = float(red.dbu_per_micron)
        for r in todo:
            p = red.placement()
            for m in r["placement"]["macros"]:
                i = idx[m["name"]]
                p.x[i], p.y[i] = int(round(m["x"] * dbu)), int(round(m["y"] * dbu))
                p.orient[i] = inv[m["orient"]]
            out = pred.predict(red, p)
            mu, sd = pred.penalty(red, p)
            cache[r["run_dir"]] = {"grt_wl_um": out["grt_wl_um"], "hot_frac": out["hot_frac"],
                                   "wns_over_t": out.get("wns_over_t"), "penalty_mu": mu, "penalty_sigma": sd}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=1))
    return cache


# ----------------------------------------------------------------------------- sections
def _bundle_time() -> str:
    import time

    p = ROOT / "runs/predict/bundle.pt"
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(p.stat().st_mtime)) if p.exists() else ""


def main_table(groups: dict[str, list[dict]], ref_wl: float | None) -> list[str]:
    cols = [("GRT WL (um)", "ppa.grt_wl_um"), ("GRT hot gcells (%)", "hot_pct"), ("GRT overflow", "ppa.grt_overflow"),
            ("WNS (ns)", "ppa.wns_ns"), ("TNS (ns)", "ppa.tns_ns"), ("power (mW)", "ppa.power_mw"),
            ("proxy", "placement.proxy"), ("placer time (s)", "placer_s")]
    out = ["| placer | ok/runs | seeds | distinct placements | " + " | ".join(c for c, _ in cols[:1]) +
           " | GRT WL vs rtlmp | " + " | ".join(c for c, _ in cols[1:]) + " |", "|---" * (5 + len(cols)) + "|"]
    for lbl in sorted(groups, key=_order):
        rs = groups[lbl]
        ok = [r for r in rs if r.get("status") == "ok"]
        seeds = sorted({r["job"]["seed"] for r in ok})
        distinct = len({json.dumps(sorted((m["name"], m["x"], m["y"], m["orient"]) for m in r["placement"]["macros"]))
                        for r in ok if (r.get("placement") or {}).get("macros")})
        wl = _vals(ok, "ppa.grt_wl_um")
        rel = fmt_ci([100 * (v / ref_wl - 1) for v in wl]) + " %" if wl and ref_wl else "n/a"
        cells = [fmt_ci(_vals(ok, p)) for _, p in cols]
        out.append(f"| {NAMES.get(lbl, lbl)} | {len(ok)}/{len(rs)} | {','.join(map(str, seeds))} | {distinct} | "
                   f"{cells[0]} | {rel} | " + " | ".join(cells[1:]) + " |")
    return out


def per_run_table(groups: dict[str, list[dict]], preds: dict) -> list[str]:
    out = ["| placer | seed | status | GRT WL (um) | hot gcells (%) | WNS (ns) | TNS (ns) | proxy | pred. GRT WL (um) | "
           "pred. hot (%) | placer (s) | flow (min) |", "|---" * 12 + "|"]
    for lbl in sorted(groups, key=_order):
        for r in sorted(groups[lbl], key=lambda r: r["job"]["seed"]):
            pr = preds.get(r["run_dir"], {})
            f = lambda v: "n/a" if v is None else _fmt(v)  # noqa: E731
            st = r.get("status")
            out.append(f"| {NAMES.get(lbl, lbl)} | {r['job']['seed']} | {st} | {f(g(r, 'ppa.grt_wl_um'))} | "
                       f"{f(g(r, 'hot_pct'))} | {f(g(r, 'ppa.wns_ns'))} | {f(g(r, 'ppa.tns_ns'))} | "
                       f"{f(g(r, 'placement.proxy'))} | {f(pr.get('grt_wl_um'))} | "
                       f"{f(None if pr.get('hot_frac') is None else 100 * pr['hot_frac'])} | {f(g(r, 'placer_s'))} | "
                       f"{f(g(r, 'flow_min'))} |")
    return out


def ablation(groups: dict[str, list[dict]]) -> list[str]:
    pairs = [("rl-v2-pred", "rl-v2-proxy"), ("rl-v2-proxy", "rl-v1"), ("rl-v2-pred", "rl-v1"),
             ("rl-v2-pred", "macromind-sa"), ("rl-v2-proxy", "macromind-sa")]
    mets = [("GRT WL (um)", "ppa.grt_wl_um"), ("hot gcells (%)", "hot_pct"), ("WNS (ns)", "ppa.wns_ns"),
            ("TNS (ns)", "ppa.tns_ns"), ("proxy", "placement.proxy")]
    out = ["| A vs B | n (A, B) | " + " | ".join(f"Δ {m} (A − B), 95% CI, p" for m, _ in mets) + " |",
           "|---" * (2 + len(mets)) + "|"]
    for a, b in pairs:
        if a not in groups or b not in groups:
            continue
        A = [r for r in groups[a] if r.get("status") == "ok"]
        B = [r for r in groups[b] if r.get("status") == "ok"]
        cells = []
        for _, p in mets:
            w = welch(_vals(A, p), _vals(B, p))
            cells.append("n/a" if w is None else f"{_fmt(w[0])} ± {_fmt(w[1])}, p={w[2]:.2f}")
        out.append(f"| {a} vs {b} | {len(A)}, {len(B)} | " + " | ".join(cells) + " |")
    return out


def correlations(recs: list[dict], preds: dict) -> list[str]:
    ok = [r for r in recs if r.get("status") == "ok" and g(r, "ppa.grt_wl_um") is not None]
    out = [f"Over {len(ok)} GRT runs (all placers; runs with identical placements count separately).", "",
           "| ranker | vs GRT WL: tau / rho | vs hot gcells: tau / rho | vs WNS (cost = −WNS): tau / rho |",
           "|---|---|---|---|"]
    rankers = [("MacroMind proxy", lambda r: g(r, "placement.proxy")),
               ("proxy wirelength term", lambda r: (r.get("placement") or {}).get("wirelength")),
               ("proxy congestion term", lambda r: (r.get("placement") or {}).get("congestion")),
               ("predictor: GRT WL", lambda r: preds.get(r["run_dir"], {}).get("grt_wl_um")),
               ("predictor: hot_frac", lambda r: preds.get(r["run_dir"], {}).get("hot_frac")),
               ("predictor: RL penalty mu", lambda r: preds.get(r["run_dir"], {}).get("penalty_mu"))]
    targets = [("ppa.grt_wl_um", 1), ("hot_frac", 1), ("ppa.wns_ns", -1)]
    for name, fx in rankers:
        cells = []
        for t, sgn in targets:
            xy = [(fx(r), g(r, t)) for r in ok]
            xy = [(x, sgn * y) for x, y in xy if x is not None and y is not None]
            if len(xy) < 3:
                cells.append("n/a")
                continue
            xs, ys = zip(*xy)
            tau, rho = kendall(xs, ys), spearman(xs, ys)
            cells.append("n/a" if tau is None else f"{tau:+.2f} / {rho:+.2f} (n={len(xs)})")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return out


def compute(groups: dict[str, list[dict]], design: str) -> list[str]:
    out = ["| item | wall | CPU | notes |", "|---|---|---|---|"]
    for v in ("proxy", "pred"):
        fj = ROOT / "runs/rl-v2" / design / v / "finetune.json"
        if fj.exists():
            f = json.loads(fj.read_text())
            out.append(f"| fine-tune rl-v2-{v} | {f['wall_s'] / 60:.1f} min | {f['cpu_s'] / 60:.1f} min | "
                       f"{f['iters']} PPO iters, {f['iters'] * f['episodes_per_iter']} episodes, {f['steps']} env steps, "
                       f"{f['threads']} thread(s); mean batch reward {f['reward_first']:+.3f} -> {f['reward_last']:+.3f} |")
    for lbl in sorted(groups, key=_order):
        ok = [r for r in groups[lbl] if r.get("make_wall_s")]
        tot = sum(r["make_wall_s"] for r in ok) / 3600
        pl = _vals(ok, "placer_s")
        out.append(f"| ORFS jobs {NAMES.get(lbl, lbl)} | {tot:.2f} h total ({len(ok)} jobs) | - | placer itself "
                   f"{fmt_ci(pl)} s per run |")
    return out


def build(no_predictor: bool = False) -> str:
    from ..harness.runner import harness_dir

    recs = dedupe(load_records(harness_dir() / "jobs"))
    lines = ["<!-- generated by `python -m macromind.report.sg10`; do not edit by hand -->", ""]
    for dspec, dname in DESIGNS.items():
        for stop in ("grt", "finish"):
            rs = [r for r in recs if r.get("job", {}).get("design") == dspec and r["job"].get("stop_after") == stop]
            if not rs:
                continue
            groups: dict[str, list[dict]] = defaultdict(list)
            for r in rs:
                groups[label(r)].append(r)
            ref = _vals([r for r in groups.get("rtlmp", []) if r.get("status") == "ok"], "ppa.grt_wl_um")
            ref_wl = ref[0] if ref else None
            preds = {}
            if not no_predictor and stop == "grt":
                try:
                    preds = _predictions(rs, dname, ROOT / "runs/sg10" / f"pred_cache_{dname}.json")
                except Exception as e:  # noqa: BLE001
                    lines += [f"(predictor evaluation failed: {type(e).__name__}: {e})", ""]
            lines += [f"## {dspec}, stop after {stop}", "", "Mean ± 95% CI (Student t) over seeds; one value = one run.",
                      "", *main_table(groups, ref_wl), ""]
            if stop == "finish":
                cols = [("routed WL (um)", "ppa.routed_wl_um"), ("WNS (ns)", "ppa.wns_ns"), ("TNS (ns)", "ppa.tns_ns"),
                        ("power (mW)", "ppa.power_mw"), ("DRC", "ppa.drc_errors"), ("flow (min)", "flow_min")]
                lines += ["Detailed route (finish):", "", "| placer | seed | " + " | ".join(c for c, _ in cols) + " |",
                          "|---" * (2 + len(cols)) + "|"]
                for lbl in sorted(groups, key=_order):
                    for r in groups[lbl]:
                        if r.get("status") == "ok":
                            lines.append(f"| {NAMES.get(lbl, lbl)} | {r['job']['seed']} | " +
                                         " | ".join(_fmt(v) if (v := g(r, p)) is not None else "n/a" for _, p in cols) + " |")
                lines.append("")
                continue
            bad = [r for r in rs if r.get("status") != "ok"]
            for r in bad:
                tail = (r.get("error") or (r.get("error_tail") or "").strip().splitlines()[-1:] or [""])
                lines.append(f"- {label(r)} seed {r['job']['seed']}: **{r.get('status')}** "
                             f"({tail if isinstance(tail, str) else tail[0]})")
            lines += ["", "### Per run", "", *per_run_table(groups, preds), "",
                      "### Ablation and pairwise differences (Welch t, small n: indicative only)", "",
                      *ablation(groups), "", "### Rank correlation with the real GRT result (Kendall tau / Spearman rho)",
                      "", "All runs (the SG7 placements among them were also in the predictor's training set):", "",
                      *correlations(rs, preds), "",
                      "Fresh placements only (jobs started after the predictor bundle was trained, so never seen by it; "
                      "same design, so still in-distribution for bp_fe_top):", "",
                      *correlations([r for r in rs if str(r.get("started", "")) > _bundle_time()], preds), "",
                      "### Compute", "", *compute(groups, dname), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m macromind.report.sg10", description=__doc__)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-predictor", action="store_true")
    a = ap.parse_args(argv)
    md = build(a.no_predictor)
    if a.out:
        Path(a.out).write_text(md + "\n")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
