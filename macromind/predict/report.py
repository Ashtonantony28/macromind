"""Markdown tables from the predictor evaluation JSONs (``python -m macromind.predict.train cv/lodo --out``).

    python -m macromind.predict.report runs/predict/cv.json [runs/predict/lodo.json]
"""

from __future__ import annotations

import json
import sys


def _f(v, d=2):
    if v is None:
        return "-"
    try:
        if v != v:  # NaN
            return "n/a"
    except TypeError:
        return str(v)
    return f"{v:+.{d}f}" if isinstance(v, float) else str(v)


TARGET_NAMES = {"grt_wl": "GRT wirelength, scalar head", "grt_wl_map": "GRT wirelength, sum of predicted usage map", "hot_frac": "congestion (hot gcell fraction), scalar head",
                "hot_frac_map": "congestion (hot gcell fraction), top-10% of predicted map", "wns": "WNS (GRT)",
                "tns": "TNS (GRT)"}


def ranking_md(res: dict) -> str:
    lines = ["| design | n | target | tau predictor | tau proxy | tau ridge (proxy feats) | tau cluster HPWL | rho predictor | rho proxy | rho(unc, abs err) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for d, row in res["ranking"].items():
        for t, name in TARGET_NAMES.items():
            if t not in row:
                continue
            r = row[t]
            lines.append(f"| {d} | {row['n']} | {name} | {_f(r['tau_pred'])} | {_f(r['tau_proxy'])} | {_f(r.get('tau_ridge'))} | "
                         f"{_f(r['tau_hpwl'])} | {_f(r['rho_pred'])} | {_f(r['rho_proxy'])} | {_f(r.get('rho_unc_err'))} |")
    return "\n".join(lines)


def maps_md(res: dict) -> str:
    if "maps" not in res:
        return ""
    keys = ["mse", "mse_design_mean", "pixel_corr", "pixel_corr_design_mean", "pixel_corr_rudy", "pixel_corr_delta"]
    lines = ["| design | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for d, m in res["maps"].items():
        lines.append(f"| {d} | " + " | ".join(_f(m.get(k), 3) for k in keys) + " |")
    return "\n".join(lines)


def nodes_md(res: dict) -> str:
    if "nodes" not in res:
        return ""
    lines = ["| design | samples with endpoint labels | rho node slack (predictor) | rho node slack (train mean per node) |",
             "|---|---|---|---|"]
    for d, m in res["nodes"].items():
        lines.append(f"| {d} | {m['n_samples']} | {_f(m['rho_node_pred'])} | {_f(m.get('rho_node_train_mean'))} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    for p in (argv or sys.argv[1:]):
        res = json.load(open(p))
        print(f"### {p}: {res.get('protocol')}\n")
        print(ranking_md(res) + "\n")
        print(maps_md(res) + "\n")
        print(nodes_md(res) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
