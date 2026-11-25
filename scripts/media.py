"""Render the README demo media into docs/media/.

    python3 scripts/media.py

  sa-placement.{gif,mp4}  the MacroMind simulated-annealing placer on a synthetic benchmark design (the same
                          generator the RL agent pre-trains on): macro positions, RUDY routing-demand map and the
                          proxy cost components, sampled from inside the real sa_place run
  results-grt.png         GRT wirelength vs OpenROAD's RTLMP for every placer on bp_fe_top, parsed from
                          docs/results/report-v1.md (ORFS global-route runs)

The animation is pure Python (no OpenROAD): it shows the proxy optimisation only. Real-flow numbers are in
the results chart. Needs matplotlib and ffmpeg.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from macromind.baselines import placers as P  # noqa: E402
from macromind.core.geometry import oriented_dims  # noqa: E402
from macromind.core.synthetic import generate_synthetic  # noqa: E402
from macromind.cost.proxy import IncrementalProxy, proxy_cost  # noqa: E402

BG, PANEL, GRID, TXT, DIM = "#0d1117", "#161b22", "#30363d", "#e6edf3", "#8b949e"
BLUE, PURPLE, ORANGE, GREEN, RED = "#58a6ff", "#d2a8ff", "#ffa657", "#3fb950", "#f85149"
OUT = ROOT / "docs" / "media"


def record_sa(design, seed=0, budget=20000, every=200):
    """Run the real sa_place, sampling the committed state every `every` proxy evaluations."""
    snaps, best = [], {}
    nm = design.n_macros

    class Rec(IncrementalProxy):
        n_eval = 0

        def propose(self, nodes, cx, cy, orient):
            if Rec.n_eval % every == 0:
                snaps.append(dict(ev=Rec.n_eval, cx=self.cx[:nm].copy(), cy=self.cy[:nm].copy(),
                                  o=self.orient[:nm].copy(), res=self.result,
                                  rudy=(self.rh / self.h_cap + self.rv / self.v_cap) / 2))
            Rec.n_eval += 1
            return super().propose(nodes, cx, cy, orient)

        def accept(self):
            res = super().accept()
            if not best or res.proxy < best["res"].proxy:
                best.update(ev=Rec.n_eval, cx=self.cx[:nm].copy(), cy=self.cy[:nm].copy(), o=self.orient[:nm].copy(),
                            res=res, rudy=(self.rh / self.h_cap + self.rv / self.v_cap) / 2)
            return res

    orig = P.IncrementalProxy
    P.IncrementalProxy = Rec
    try:
        final = P.sa_place(design, seed=seed, budget=budget)
    finally:
        P.IncrementalProxy = orig
    return snaps, best, final


def draw(fig, design, s, hist, final_res=None, label=""):
    arr = design.arrays()
    nm = design.n_macros
    c = design.core
    fig.clf()
    fig.text(0.035, 0.935, "MacroMind: simulated-annealing macro placement", fontsize=18, weight="bold", color=TXT)
    fig.text(0.035, 0.895, "SIMULATION on a synthetic benchmark design (the RL pre-training generator): "
             f"{nm} hard macros, {len(design.cells)} std-cell clusters, {len(design.nets)} nets.  "
             "Proxy cost only; real OpenROAD results are in the README table.", fontsize=9.3, color=DIM)
    ax = fig.add_axes([0.035, 0.07, 0.47, 0.78])
    ax.set_facecolor(PANEL)
    ext = (c.xl, c.xh, c.yl, c.yh)
    cm = LinearSegmentedColormap.from_list("r", [PANEL, "#3b2c5e", "#8957e5", ORANGE, RED])
    ax.imshow(s["rudy"].T, origin="lower", extent=ext, cmap=cm, vmin=0, vmax=1.6, interpolation="bilinear", alpha=0.95)
    w, h = oriented_dims(arr.node_w[:nm], arr.node_h[:nm], s["o"])
    for i in range(nm):
        x0, y0 = s["cx"][i] - w[i] / 2, s["cy"][i] - h[i] / 2
        ax.add_patch(Rectangle((x0, y0), w[i], h[i], fc="#1f6feb", ec=BLUE, lw=1.2, alpha=0.88))
        # pin-1 corner marker shows orientation
        o = int(s["o"][i])
        mx = x0 + (w[i] * 0.88 if o in (1, 2, 5, 6) else w[i] * 0.12)
        my = y0 + (h[i] * 0.88 if o in (1, 2, 4, 7) else h[i] * 0.12)
        ax.plot(mx, my, "o", ms=2.5, color=TXT)
    px = [p.x for p in design.ports]
    py = [p.y for p in design.ports]
    ax.plot(px, py, "s", ms=3, color=GREEN)
    d = design.die
    ax.add_patch(Rectangle((c.xl, c.yl), c.width, c.height, fc="none", ec=DIM, lw=0.8, ls="--"))
    ax.set_xlim(d.xl, d.xh); ax.set_ylim(d.yl, d.yh)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_title(f"core, after {s['ev']:,} proxy evaluations   {label}", color=TXT, fontsize=11, loc="left")
    for xx, t, col in ((0.035, "■ macros (dot = pin-1 corner)", BLUE), (0.19, "■ I/O ports", GREEN),
                       (0.255, "background: RUDY routing demand / capacity (dark = low, red = over capacity)", DIM)):
        fig.text(xx, 0.035, t, color=col, fontsize=8.5)

    ev = [hh["ev"] for hh in hist]
    for k, (key, name, col) in enumerate([("proxy", "proxy cost (weighted sum)", TXT), ("wirelength", "wirelength", BLUE),
                                          ("density", "density", PURPLE), ("congestion", "congestion", ORANGE)]):
        ax = fig.add_axes([0.57, 0.66 - k * 0.19, 0.40, 0.15])
        ax.set_facecolor(BG)
        v = [getattr(hh["res"], key) for hh in hist]
        ax.plot(ev, v, color=col, lw=1.6)
        ax.set_xlim(0, ALL_EV)
        lo, hi = LIM[key]
        ax.set_ylim(lo, hi)
        ax.tick_params(colors=DIM, labelsize=7.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.set_title(f"{name}  {v[-1]:.4f}", color=col, fontsize=10, loc="left", pad=2,
                     weight="bold" if key == "proxy" else "normal")
        if k < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("proxy evaluations", color=DIM, fontsize=9)
    if final_res is not None:
        b, lg = final_res
        fig.text(0.57, 0.875, f"best placement: proxy {b.proxy:.4f}, "
                 f"{(1 - b.proxy / hist[0]['res'].proxy) * 100:.0f}% below the grid start",
                 color=GREEN, fontsize=11, weight="bold")
        fig.text(0.57, 0.85, f"(sa_place returns it after the legaliser's orientation pass: proxy {lg.proxy:.4f})",
                 color=DIM, fontsize=8.5)


ALL_EV = 1
LIM: dict = {}


def sa_animation(fps=12):
    global ALL_EV, LIM
    design = generate_synthetic(n_macros=16, n_clusters=300, n_ports=40, seed=4, macro_area_fraction=0.45)
    snaps, best, final = record_sa(design, seed=0, budget=20000, every=100)
    legal_res = proxy_cost(design, final)
    assert abs(best["res"].proxy - final.meta["best_proxy"]) < 1e-9, "recorded best != sa_place's best"
    ALL_EV = snaps[-1]["ev"]
    for key in ("proxy", "wirelength", "density", "congestion"):
        v = np.array([getattr(s["res"], key) for s in snaps])
        pad = (v.max() - v.min()) * 0.12 + 1e-9
        LIM[key] = (v.min() - pad, v.max() + pad)
    last = dict(best, ev=ALL_EV)
    # time-warp: annealing does most of its moving early, so sample early frames densely
    idx = np.unique(np.round(np.geomspace(1, len(snaps), 110)).astype(int) - 1)
    tmp = Path(tempfile.mkdtemp())
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    n = 0
    for _ in range(8):
        draw(fig, design, snaps[0], snaps[:1], label="(grid start)")
        fig.savefig(tmp / f"f{n:04d}.png", facecolor=BG); n += 1
    for i in idx:
        draw(fig, design, snaps[i], snaps[: i + 1])
        fig.savefig(tmp / f"f{n:04d}.png", facecolor=BG); n += 1
    for _ in range(30):
        draw(fig, design, last, snaps, (best["res"], legal_res), label=f"(best, found at {best['ev']:,})")
        fig.savefig(tmp / f"f{n:04d}.png", facecolor=BG); n += 1
    fig.savefig(OUT / "sa-placement.png", facecolor=BG)
    plt.close(fig)
    ff = ["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", str(tmp / "f%04d.png")]
    subprocess.run(ff + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(OUT / "sa-placement.mp4")], check=True)
    subprocess.run(ff + ["-vf", "scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];"
                         "[b][p]paletteuse=dither=none", str(OUT / "sa-placement.gif")], check=True)
    shutil.rmtree(tmp)


def results_chart():
    text = (ROOT / "docs" / "results" / "report-v1.md").read_text()
    sec = text.split("## Main table", 1)[1]
    rows = []
    for ln in sec.splitlines():
        cells = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(cells) >= 10 and cells[0] not in ("placer",) and not set(cells[0]) <= set("-"):
            wl = re.match(r"([\d,]+)(?: ± ([\d,]+))?", cells[2])
            if wl:
                rows.append((cells[0], int(wl.group(1).replace(",", "")),
                             int(wl.group(2).replace(",", "")) if wl.group(2) else 0, cells[4]))
        elif rows and not ln.strip():
            break
    base = dict((r[0], r[1]) for r in rows)["rtlmp"]
    fig, ax = plt.subplots(figsize=(12.8, 4.8), dpi=100, facecolor=BG)
    ax.set_facecolor(BG)
    names = [r[0] for r in rows][::-1]
    vals = [(r[1] / base - 1) * 100 for r in rows][::-1]
    errs = [r[2] / base * 100 for r in rows][::-1]
    cols = [GREEN if n == "rtlmp" else (BLUE if n.startswith("rl") else (PURPLE if n == "sa" else DIM)) for n in names]
    ax.barh(range(len(names)), vals, xerr=errs, color=cols, height=0.6, ecolor=TXT, capsize=3)
    for i, (v, r) in enumerate(zip(vals, rows[::-1])):
        ax.text(max(v, 0) + max(errs[i], 0) + 0.4, i, f"{r[1]:,} µm" + ("  (reference)" if r[0] == "rtlmp" else f"   +{v:.1f}%")
                + f"   hot gcells {r[3]}%", va="center", color=TXT, fontsize=9.5)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, color=TXT, fontsize=10.5)
    ax.axvline(0, color=GRID)
    ax.set_xlim(-4, max(v + e for v, e in zip(vals, errs)) * 1.55)
    ax.set_xlabel("global-route wirelength vs OpenROAD RTLMP (%), lower is better; bars: 95% CI over seeds", color=DIM)
    ax.tick_params(colors=DIM)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    fig.suptitle("Real flow: bp_fe_top (11 macros, nangate45) through OpenROAD-flow-scripts to global route",
                 color=TXT, fontsize=14, weight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.885, "Source: docs/results/report-v1.md. RTLMP still wins on wirelength; fine-tuned RL reaches SA level.",
             color=DIM, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(OUT / "results-grt.png", facecolor=BG)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results_chart()
    sa_animation()


if __name__ == "__main__":
    main()
