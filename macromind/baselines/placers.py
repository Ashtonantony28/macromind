"""Baseline macro placers: random legal, shelf packing, simulated annealing.

Every placer has the signature ``place(design, seed=0, budget=None) -> Placement``
and returns a placement that has been through :func:`macromind.legalize.legalize`.
``budget`` is the number of proxy-cost evaluations a placer may spend
(ignored by the constructive placers).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from ..core.design import Design, Placement
from ..core.geometry import oriented_dims
from ..cost.proxy import IncrementalProxy, ProxyConfig
from ..legalize.legalizer import SnapGrid, check_legality, legalize, macro_bounds


class Placer(Protocol):
    def __call__(self, design: Design, seed: int = 0, budget: int | None = None) -> Placement: ...


def _movable(design: Design) -> np.ndarray:
    return np.flatnonzero([not m.fixed for m in design.macros])


def _dims(design: Design, orient: np.ndarray):
    arr = design.arrays()
    nm = arr.n_macros
    w, h = oriented_dims(arr.node_w[:nm], arr.node_h[:nm], orient)
    return np.asarray(w, np.int64), np.asarray(h, np.int64)


# ---------------------------------------------------------------- random
def random_place(design: Design, seed: int = 0, budget: int | None = None) -> Placement:
    """Uniformly random grid positions (orientation ``N``), then legalised."""
    rng = np.random.default_rng(seed)
    p = design.placement()
    mov = _movable(design)
    p.orient[mov] = 0
    w, h = _dims(design, p.orient)
    xmin, xmax, ymin, ymax = macro_bounds(design, w, h)
    g = SnapGrid.of(design)
    for i in mov:
        p.x[i] = g.near_x(rng.uniform(xmin[i], max(xmin[i], xmax[i])))
        p.y[i] = g.near_y(rng.uniform(ymin[i], max(ymin[i], ymax[i])))
    out = legalize(design, p)
    out.meta.update(method="random", seed=seed)
    return out


# ------------------------------------------------------------------ grid
def grid_place(design: Design, seed: int = 0, budget: int | None = None) -> Placement:
    """Shelf packing: macros sorted by height (ties shuffled by ``seed``) are
    packed left to right in shelves from the bottom of the core, with halo
    spacing; shelves are then spread evenly over the core height. Legalised."""
    rng = np.random.default_rng(seed)
    p = design.placement()
    mov = _movable(design)
    p.orient[mov] = 0
    w, h = _dims(design, p.orient)
    halo = design.halo
    c = design.core
    g = SnapGrid.of(design)
    jitter = rng.permutation(len(mov))
    order = mov[np.lexsort((jitter, -h[mov]))]
    shelves: list[list[int]] = [[]]
    x = c.xl
    for i in order:
        if shelves[-1] and x + w[i] > c.xh:
            shelves.append([])
            x = c.xl
        shelves[-1].append(int(i))
        x += w[i] + halo
    heights = [max(h[s]) for s in shelves if s]
    total = sum(heights) + halo * (len(heights) - 1)
    gap = max(0.0, (c.height - total) / max(len(heights) - 1, 1)) if len(heights) > 1 else 0.0
    y = c.yl
    for s, sh in zip(shelves, heights):
        row_w = sum(w[i] for i in s) + halo * (len(s) - 1)
        gx = max(0.0, (c.width - row_w) / max(len(s) - 1, 1)) if len(s) > 1 else 0.0
        x = c.xl
        for i in s:
            p.x[i] = g.up_x(x)
            p.y[i] = g.up_y(y)
            x += w[i] + halo + gx
        y += sh + halo + gap
    out = legalize(design, p)
    out.meta.update(method="grid", seed=seed)
    return out


# ------------------------------------------------------ simulated annealing
@dataclass
class SAConfig:
    """Simulated-annealing parameters."""

    budget: int = 20000  # proxy evaluations (including temperature calibration)
    init: str = "grid"  # "grid" | "random" | "current"
    p_shift: float = 0.6
    p_swap: float = 0.25
    p_flip: float = 0.15
    accept_init: float = 0.5  # target acceptance of an average uphill move at T0
    t_final_ratio: float = 1e-3
    shift_start: float = 0.5  # max shift as a fraction of core size, at start
    shift_end: float = 0.01
    calib_moves: int = 200
    resync_every: int = 5000
    max_attempt_factor: int = 20  # cap on proposals (including infeasible ones) per evaluation
    proxy: ProxyConfig = field(default_factory=ProxyConfig)


class _SAState:
    """Macro rectangles kept for O(n) overlap checks during annealing."""

    def __init__(self, design: Design, p: Placement):
        self.halo = design.halo
        self.w0 = design.arrays().node_w[: len(p)].astype(np.int64)
        self.h0 = design.arrays().node_h[: len(p)].astype(np.int64)
        self.x = p.x.copy()
        self.y = p.y.copy()
        self.o = p.orient.copy()
        self.w, self.h = _dims(design, self.o)
        self.fixed = np.array([m.fixed for m in design.macros], dtype=bool)

    def free(self, i: int, x: int, y: int, w: int, h: int, ignore: int = -1) -> bool:
        hal = self.halo
        hit = (x < self.x + self.w + hal) & (x + w + hal > self.x) & (y < self.y + self.h + hal) & (y + h + hal > self.y)
        hit[i] = False
        if ignore >= 0:
            hit[ignore] = False
        return not hit.any()


def sa_place(design: Design, seed: int = 0, budget: int | None = None, config: SAConfig | None = None) -> Placement:
    """Simulated annealing on the proxy cost.

    Moves: *shift* (random displacement whose radius shrinks geometrically),
    *swap* (exchange the centres of two movable macros) and *flip* (another
    footprint-preserving orientation). The state is kept overlap-free (with
    halo) at all times: infeasible proposals are discarded without spending a
    cost evaluation. Costs are updated incrementally
    (:class:`~macromind.cost.IncrementalProxy`). Temperature follows a geometric
    schedule from ``T0`` (calibrated so that an average uphill move is accepted
    with probability ``accept_init``) to ``T0 * t_final_ratio``. The best
    placement seen is legalised and returned.
    """
    cfg = config or SAConfig()
    budget = int(budget if budget is not None else cfg.budget)
    rng = np.random.default_rng(seed)
    t_start = time.perf_counter()

    if cfg.init == "grid":
        p0 = grid_place(design, seed)
    elif cfg.init == "random":
        p0 = random_place(design, seed)
    elif cfg.init == "current":
        p0 = legalize(design)
    else:
        raise ValueError(f"unknown init {cfg.init!r}")
    mov = _movable(design)
    if len(mov) == 0 or budget <= 0:
        p0.meta.update(method="sa", seed=seed, evaluations=0)
        return p0

    st = _SAState(design, p0)
    inc = IncrementalProxy(design, p0, cfg.proxy)
    grid = SnapGrid.of(design)
    c = design.core

    def bnd(i: int, w: int, h: int):
        return int(grid.up_x(c.xl)), int(grid.down_x(c.xh - w)), int(grid.up_y(c.yl)), int(grid.down_y(c.yh - h))

    probs = np.array([cfg.p_shift, cfg.p_swap if len(mov) > 1 else 0.0, cfg.p_flip], dtype=float)
    probs /= probs.sum()
    span = max(c.width, c.height)

    def propose(radius: float):
        """Return (nodes, xs, ys, os) of a feasible move, or None."""
        kind = rng.choice(3, p=probs)
        if kind == 0:
            i = int(mov[rng.integers(len(mov))])
            w, h = int(st.w[i]), int(st.h[i])
            x0, x1, y0, y1 = bnd(i, w, h)
            nx = int(np.clip(grid.near_x(st.x[i] + rng.uniform(-radius, radius)), x0, x1))
            ny = int(np.clip(grid.near_y(st.y[i] + rng.uniform(-radius, radius)), y0, y1))
            if (nx == st.x[i] and ny == st.y[i]) or not st.free(i, nx, ny, w, h):
                return None
            return [i], [nx], [ny], [int(st.o[i])]
        if kind == 1:
            a, b = (int(v) for v in mov[rng.choice(len(mov), 2, replace=False)])
            wa, ha, wb, hb = int(st.w[a]), int(st.h[a]), int(st.w[b]), int(st.h[b])
            cxa, cya = st.x[a] + wa / 2, st.y[a] + ha / 2
            cxb, cyb = st.x[b] + wb / 2, st.y[b] + hb / 2
            ba, bb = bnd(a, wa, ha), bnd(b, wb, hb)
            nxa = int(np.clip(grid.near_x(cxb - wa / 2), ba[0], ba[1]))
            nya = int(np.clip(grid.near_y(cyb - ha / 2), ba[2], ba[3]))
            nxb = int(np.clip(grid.near_x(cxa - wb / 2), bb[0], bb[1]))
            nyb = int(np.clip(grid.near_y(cya - hb / 2), bb[2], bb[3]))
            if not st.free(a, nxa, nya, wa, ha, ignore=b) or not st.free(b, nxb, nyb, wb, hb, ignore=a):
                return None
            hal = st.halo
            if nxa < nxb + wb + hal and nxa + wa + hal > nxb and nya < nyb + hb + hal and nya + ha + hal > nyb:
                return None
            return [a, b], [nxa, nxb], [nya, nyb], [int(st.o[a]), int(st.o[b])]
        i = int(mov[rng.integers(len(mov))])
        cur = int(st.o[i])
        choices = [o for o in range(cur & 1, 8, 2) if o != cur]
        return [i], [int(st.x[i])], [int(st.y[i])], [int(rng.choice(choices))]

    def evaluate(move):
        nodes, xs, ys, os_ = move
        cxs = [x + st.w[n] / 2 for n, x in zip(nodes, xs)]
        cys = [y + st.h[n] / 2 for n, y in zip(nodes, ys)]
        return inc.propose(nodes, cxs, cys, os_)

    def commit(move):
        nodes, xs, ys, os_ = move
        inc.accept()
        for n, x, y, o in zip(nodes, xs, ys, os_):
            st.x[n], st.y[n], st.o[n] = x, y, o

    # --- temperature calibration
    evals = 0
    attempts = 0
    ups: list[float] = []
    cur_cost = inc.cost
    max_attempts = cfg.max_attempt_factor * budget
    calib = min(cfg.calib_moves, max(1, budget // 10))
    while evals < calib and attempts < max_attempts:
        attempts += 1
        mv = propose(cfg.shift_start * span)
        if mv is None:
            continue
        new = evaluate(mv)
        inc.reject()
        evals += 1
        if new > cur_cost:
            ups.append(new - cur_cost)
    mean_up = float(np.mean(ups)) if ups else max(abs(cur_cost) * 1e-3, 1e-9)
    t0 = -mean_up / math.log(cfg.accept_init)
    t_end = t0 * cfg.t_final_ratio

    best_cost = cur_cost
    best = (st.x.copy(), st.y.copy(), st.o.copy())
    accepted = 0
    n_main = max(1, budget - evals)
    k = 0
    log_ratio_t = math.log(t_end / t0)
    log_ratio_r = math.log(cfg.shift_end / cfg.shift_start)
    while evals < budget and attempts < max_attempts:
        attempts += 1
        frac = k / n_main
        T = t0 * math.exp(log_ratio_t * frac)
        radius = max(cfg.shift_start * span * math.exp(log_ratio_r * frac), 2 * grid.px)
        mv = propose(radius)
        if mv is None:
            continue
        new = evaluate(mv)
        evals += 1
        k += 1
        d = new - cur_cost
        if d <= 0 or rng.random() < math.exp(-d / T):
            commit(mv)
            cur_cost = new
            accepted += 1
            if cur_cost < best_cost:
                best_cost = cur_cost
                best = (st.x.copy(), st.y.copy(), st.o.copy())
            if cfg.resync_every and accepted % cfg.resync_every == 0:
                cur_cost = inc.resync().proxy
        else:
            inc.reject()

    p = Placement(best[0], best[1], best[2])
    errs = check_legality(design, p)
    if errs:  # pragma: no cover - invariant
        raise RuntimeError(f"SA state became illegal: {errs[:3]}")
    out = legalize(design, p)
    out.meta.update(
        method="sa", seed=seed, evaluations=evals, attempts=attempts, accepted=accepted,
        t0=t0, best_proxy=best_cost, runtime_s=time.perf_counter() - t_start,
    )
    return out


def rl_place(design: Design, seed: int = 0, budget: int | None = None, checkpoint: str | None = None) -> Placement:
    """RL policy placer (lazy torch import). ``budget`` = sampled episodes (best-of-k); ``None`` = greedy.

    The checkpoint comes from ``checkpoint``, else ``$MACROMIND_RL_CKPT``, else ``runs/rl-v1/policy.pt``.
    """
    from ..place.agent import rl_place as _rl

    return _rl(design, seed=seed, budget=budget, checkpoint=checkpoint)


PLACERS: dict[str, Callable[..., Placement]] = {
    "random": random_place,
    "grid": grid_place,
    "sa": sa_place,
    "rl": rl_place,
}


def get_placer(name: str) -> Callable[..., Placement]:
    try:
        return PLACERS[name]
    except KeyError:
        raise KeyError(f"unknown placer {name!r}; choose from {sorted(PLACERS)}") from None


def place(design: Design, method: str = "sa", seed: int = 0, budget: int | None = None, *, cluster_pass: bool = False) -> Placement:
    """Run the named placer.

    With ``cluster_pass=True`` the std-cell clusters are then re-placed around
    the macros by :func:`macromind.graph.place_clusters` (the same pass the RL
    placer uses), and the resulting design is stored in
    ``placement.meta["cluster_design"]``; evaluate the proxy on that design.
    """
    p = get_placer(method)(design, seed=seed, budget=budget)
    if cluster_pass:
        from ..graph.quadratic import place_clusters

        p.meta["cluster_design"] = place_clusters(design, p)
    return p
