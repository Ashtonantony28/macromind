"""Macro legaliser shared by every placer.

Steps:

1. **Snap** each movable macro's lower-left corner to the site/row grid
   (``x = core.xl + k * site_width``, ``y = core.yl + r * row_height``) and
   clamp it inside the core.
2. **Overlap removal with halo**: macros are committed one by one (fixed
   macros first, then movable ones by decreasing area). A macro whose snapped
   position is free is kept; otherwise it moves to the nearest (Euclidean)
   free position. The search is exact over the candidate set formed by its
   desired coordinate, the core bounds and every committed macro's edges
   inflated by the halo; any free region contains such a candidate (slide it
   left, then down, until it touches something), so if the search fails there
   is genuinely no room for that macro given the earlier commitments.
3. **Orientation**: each movable macro picks, among the orientations that keep
   its footprint (``N/S/FN/FS``, or the rotated set for rotated macros), the one
   with the lowest HPWL of its nets (one greedy pass in commit order).

Legality means: inside the core, on the grid, and for every pair of macros at
least ``halo`` DBU of separation along x or y (fixed-fixed pairs are exempt).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.design import Design, Placement
from ..core.geometry import oriented_dims


class LegalizationError(RuntimeError):
    """Raised when a macro cannot be placed legally."""


@dataclass(frozen=True)
class SnapGrid:
    x0: int
    y0: int
    px: int
    py: int

    @staticmethod
    def of(design: Design) -> "SnapGrid":
        return SnapGrid(int(design.core.xl), int(design.core.yl), max(1, design.site_width), max(1, design.row_height))

    def down_x(self, x):
        return self.x0 + np.floor((np.asarray(x) - self.x0) / self.px).astype(np.int64) * self.px

    def up_x(self, x):
        return self.x0 + np.ceil((np.asarray(x) - self.x0) / self.px).astype(np.int64) * self.px

    def near_x(self, x):
        return self.x0 + np.rint((np.asarray(x) - self.x0) / self.px).astype(np.int64) * self.px

    def down_y(self, y):
        return self.y0 + np.floor((np.asarray(y) - self.y0) / self.py).astype(np.int64) * self.py

    def up_y(self, y):
        return self.y0 + np.ceil((np.asarray(y) - self.y0) / self.py).astype(np.int64) * self.py

    def near_y(self, y):
        return self.y0 + np.rint((np.asarray(y) - self.y0) / self.py).astype(np.int64) * self.py


def macro_bounds(design: Design, w: np.ndarray, h: np.ndarray, grid: SnapGrid | None = None):
    """Grid-aligned ranges ``[xmin, xmax] x [ymin, ymax]`` for lower-left corners."""
    g = grid or SnapGrid.of(design)
    c = design.core
    xmin = np.full(len(w), int(g.up_x(c.xl)), dtype=np.int64)
    ymin = np.full(len(h), int(g.up_y(c.yl)), dtype=np.int64)
    xmax = g.down_x(c.xh - np.asarray(w))
    ymax = g.down_y(c.yh - np.asarray(h))
    return xmin, xmax, ymin, ymax


def _oriented_wh(design: Design, orient: np.ndarray):
    arr = design.arrays()
    nm = arr.n_macros
    w, h = oriented_dims(arr.node_w[:nm], arr.node_h[:nm], orient)
    return np.asarray(w, dtype=np.int64), np.asarray(h, dtype=np.int64)


def check_legality(design: Design, placement: Placement | None = None, *, halo: int | None = None, check_grid: bool = True) -> list[str]:
    """Return a list of human-readable violations (empty if legal)."""
    p = placement if placement is not None else design.placement()
    halo = design.halo if halo is None else int(halo)
    fixed = np.array([m.fixed for m in design.macros], dtype=bool)
    w, h = _oriented_wh(design, p.orient)
    xl, yl = p.x, p.y
    xh, yh = xl + w, yl + h
    c = design.core
    errs: list[str] = []
    mov = ~fixed
    out = mov & ((xl < c.xl) | (yl < c.yl) | (xh > c.xh) | (yh > c.yh))
    for i in np.flatnonzero(out):
        errs.append(f"{design.macros[i].name}: outside core")
    if check_grid:
        g = SnapGrid.of(design)
        off = mov & (((xl - g.x0) % g.px != 0) | ((yl - g.y0) % g.py != 0))
        for i in np.flatnonzero(off):
            errs.append(f"{design.macros[i].name}: off grid")
    n = len(xl)
    if n > 1:
        ov = (xl[:, None] < xh[None, :] + halo) & (xh[:, None] + halo > xl[None, :])
        ov &= (yl[:, None] < yh[None, :] + halo) & (yh[:, None] + halo > yl[None, :])
        ov &= ~(fixed[:, None] & fixed[None, :])
        np.fill_diagonal(ov, False)
        ii, jj = np.nonzero(np.triu(ov))
        for i, j in zip(ii, jj):
            errs.append(f"{design.macros[i].name} overlaps {design.macros[j].name} (halo {halo})")
    return errs


def is_legal(design: Design, placement: Placement | None = None, **kw) -> bool:
    return not check_legality(design, placement, **kw)


def _find_spot(
    dx: float, dy: float, w: int, h: int, bounds: tuple[int, int, int, int],
    obs: np.ndarray, grid: SnapGrid, batch: int = 4096,
) -> tuple[int, int] | None:
    """Nearest free grid position to desired lower-left (dx, dy); obs = inflated rects (k, 4)."""
    xmin, xmax, ymin, ymax = bounds
    if xmax < xmin or ymax < ymin:
        return None
    sx = int(np.clip(grid.near_x(dx), xmin, xmax))
    sy = int(np.clip(grid.near_y(dy), ymin, ymax))
    if len(obs) == 0:
        return sx, sy
    oxl, oyl, oxh, oyh = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]

    def free(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        hit = (X[:, None] < oxh[None, :]) & (X[:, None] + w > oxl[None, :])
        hit &= (Y[:, None] < oyh[None, :]) & (Y[:, None] + h > oyl[None, :])
        return ~hit.any(axis=1)

    if free(np.array([sx]), np.array([sy]))[0]:
        return sx, sy
    xs = np.concatenate([[sx, xmin, xmax], grid.up_x(oxh), grid.down_x(oxl - w)])
    ys = np.concatenate([[sy, ymin, ymax], grid.up_y(oyh), grid.down_y(oyl - h)])
    xs = np.unique(xs[(xs >= xmin) & (xs <= xmax)])
    ys = np.unique(ys[(ys >= ymin) & (ys <= ymax)])
    X = np.repeat(xs, len(ys))
    Y = np.tile(ys, len(xs))
    d = (X - dx) ** 2 + (Y - dy) ** 2
    order = np.argsort(d, kind="stable")
    for s in range(0, len(order), batch):
        sel = order[s : s + batch]
        ok = free(X[sel], Y[sel])
        if ok.any():
            k = sel[int(np.argmax(ok))]
            return int(X[k]), int(Y[k])
    return None


def legalize(
    design: Design,
    placement: Placement | None = None,
    *,
    halo: int | None = None,
    choose_orientation: bool = True,
) -> Placement:
    """Legalise a macro placement (see module docstring). Fixed macros never move.

    Raises :class:`LegalizationError` if some macro cannot be placed.
    """
    p = (placement if placement is not None else design.placement()).copy()
    halo = design.halo if halo is None else int(halo)
    grid = SnapGrid.of(design)
    n = len(p)
    fixed = np.array([m.fixed for m in design.macros], dtype=bool)
    w, h = _oriented_wh(design, p.orient)
    xmin, xmax, ymin, ymax = macro_bounds(design, w, h, grid)

    obs = np.zeros((n, 4), dtype=np.int64)
    k = 0
    for i in np.flatnonzero(fixed):
        obs[k] = (p.x[i] - halo, p.y[i] - halo, p.x[i] + w[i] + halo, p.y[i] + h[i] + halo)
        k += 1
    movable = np.flatnonzero(~fixed)
    order = movable[np.argsort(-(w[movable] * h[movable]), kind="stable")]
    for i in order:
        spot = _find_spot(float(p.x[i]), float(p.y[i]), int(w[i]), int(h[i]), (int(xmin[i]), int(xmax[i]), int(ymin[i]), int(ymax[i])), obs[:k], grid)
        if spot is None:
            raise LegalizationError(f"no legal position for macro {design.macros[i].name} ({w[i]}x{h[i]})")
        p.x[i], p.y[i] = spot
        obs[k] = (spot[0] - halo, spot[1] - halo, spot[0] + w[i] + halo, spot[1] + h[i] + halo)
        k += 1

    if choose_orientation and len(order):
        p = choose_orientations(design, p, order)
    p.meta["legalized"] = True
    return p


def choose_orientations(design: Design, placement: Placement, order=None) -> Placement:
    """Greedy per-macro orientation choice (footprint-preserving) by local HPWL."""
    from ..cost.proxy import IncrementalProxy, ProxyConfig

    p = placement.copy()
    inc = IncrementalProxy(design, p, ProxyConfig(w_density=0.0, w_congestion=0.0))
    if order is None:
        order = np.flatnonzero([not m.fixed for m in design.macros])
    for i in order:
        i = int(i)
        cur = int(inc.orient[i])
        best_o, best_wl = cur, inc.hpwl
        for o in range(cur & 1, 8, 2):  # same parity => same footprint
            if o == cur:
                continue
            inc.propose([i], [inc.cx[i]], [inc.cy[i]], [o])
            wl = inc.pending_result.hpwl
            inc.reject()
            if wl < best_wl - 1e-9:
                best_o, best_wl = o, wl
        if best_o != cur:
            inc.propose([i], [inc.cx[i]], [inc.cy[i]], [best_o])
            inc.accept()
            p.orient[i] = best_o
    return p
