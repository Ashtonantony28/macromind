"""Fast quadratic placement of std-cell clusters around fixed macros.

After macros are placed, every placer (RL and baselines alike) runs
:func:`place_clusters` so that soft clusters follow the macros, as AlphaChip
does with its force-directed pass. The method is a small SimPL/FastPlace-style
loop:

1. **Quadratic solve.** Nets become springs: 2-pin nets a single spring,
   nets with up to ``clique_max`` pins a clique with weight ``w / (k - 1)``,
   larger nets a star with an extra free variable (weight ``w * k / (k - 1)``).
   Macros (at the given placement), ports and fixed cells are anchors. The
   sparse SPD system is solved separately for x and y (``scipy`` SuperLU).
2. **Cell shifting** (FastPlace): on an ``n x n`` bin grid whose capacity is
   the free area left by macros, each row (column) of bins has its interior
   boundaries moved towards less-utilised neighbours,
   ``b' = (c_i (U_{i+1} + d) + c_{i+1} (U_i + d)) / (U_i + U_{i+1} + 2d)``,
   and clusters are mapped linearly into their stretched bins. This gives
   spread target positions.
3. **Pseudo-anchors.** The next solve adds springs to the spread targets with
   a weight that grows each iteration, trading wirelength for density.

Topology-dependent data are cached per :class:`DesignArrays`, so repeated calls
on the same design (one per RL episode) only rebuild the right-hand side.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..core.design import KIND_CELL, KIND_MACRO, Design, DesignArrays, Placement
from ..core.geometry import oriented_dims, transform_offsets
from ..cost.grid import GridSpec, axis_overlap
from ..cost.wirelength import hpwl

_CACHE_ATTR = "_quadratic_model"


@dataclass
class QPConfig:
    iters: int = 30
    clique_max: int = 6
    grid: int | None = None  # spreading bins per side (default ~ sqrt(#movable), 4..32)
    delta: float = 0.5  # FastPlace cell-shifting damping
    anchor_start: float = 0.1  # pseudo-anchor weight, relative to mean spring diagonal
    anchor_growth: float = 1.3
    target_density: float = 0.6  # spread clusters until area/free-area ~ this
    regularize: float = 1e-6


@dataclass
class _Springs:
    """Static spring list. Terminals ``>= 0`` are pin indices, ``< 0`` star variables ``-1 - s``."""

    ta: np.ndarray
    tb: np.ndarray
    w: np.ndarray
    n_star: int


def _springs(arr: DesignArrays, clique_max: int) -> _Springs:
    ta: list[np.ndarray] = []
    tb: list[np.ndarray] = []
    ww: list[np.ndarray] = []
    n_star = 0
    ptr = arr.net_ptr
    deg = np.diff(ptr)
    for k in np.unique(deg):
        nets = np.flatnonzero(deg == k)
        if k < 2 or len(nets) == 0:
            continue
        pins = ptr[nets][:, None] + np.arange(k)[None, :]  # (n, k)
        wn = arr.net_weight[nets]
        if k <= clique_max:
            iu, ju = np.triu_indices(k, 1)
            ta.append(pins[:, iu].ravel())
            tb.append(pins[:, ju].ravel())
            ww.append(np.repeat(wn / (k - 1), len(iu)))
        else:
            star = -1 - (n_star + np.arange(len(nets)))
            n_star += len(nets)
            ta.append(pins.ravel())
            tb.append(np.repeat(star, k))
            ww.append(np.repeat(wn * k / (k - 1), k))
    if not ta:
        z = np.zeros(0, np.int64)
        return _Springs(z, z, np.zeros(0), 0)
    return _Springs(np.concatenate(ta).astype(np.int64), np.concatenate(tb).astype(np.int64), np.concatenate(ww), n_star)


def _model(arr: DesignArrays, clique_max: int) -> _Springs:
    cache = getattr(arr, "__dict__", {}).get(_CACHE_ATTR)
    if cache is None or cache[0] != clique_max:
        cache = (clique_max, _springs(arr, clique_max))
        arr.__dict__[_CACHE_ATTR] = cache
    return cache[1]


def movable_cells(design: Design) -> np.ndarray:
    """Node indices (in ``DesignArrays`` order) of non-fixed cells/clusters."""
    nm = len(design.macros)
    return nm + np.flatnonzero([not c.fixed for c in design.cells]).astype(np.int64)


def place_clusters(
    design: Design,
    placement: Placement | None = None,
    *,
    config: QPConfig | None = None,
    movable: np.ndarray | None = None,
    spread: bool = True,
) -> Design:
    """Return a copy of ``design`` with clusters placed around the macros.

    ``placement`` gives macro positions (default: the design's own; unplaced
    macros are then ignored as anchors). ``movable`` optionally restricts the
    moved cells (indices into ``design.cells``); by default every non-fixed
    cell moves. The returned design has macros at ``placement`` and fresh
    arrays, so it can be passed straight to :func:`macromind.cost.proxy_cost`.
    """
    cfg = config or QPConfig()
    arr = design.arrays()
    nm = arr.n_macros
    if movable is None:
        mov_nodes = movable_cells(design)
    else:
        mov_nodes = nm + np.asarray(movable, dtype=np.int64)
    out = design.apply_placement(placement) if placement is not None else copy.copy(design)
    if len(mov_nodes) == 0:
        return out
    ignore_macros = placement is None and not all(m.placed for m in design.macros)
    cx, cy, orient = arr.node_state(placement)
    ignore = None
    if ignore_macros:
        ignore = np.zeros(arr.n_nodes, dtype=bool)
        ignore[:nm] = [not m.placed for m in design.macros]
    x, y = _solve(design, arr, cx, cy, orient, mov_nodes, cfg, spread, ignore)
    return _with_cells(out, arr, mov_nodes, x, y)


def _with_cells(out: Design, arr: DesignArrays, mov_nodes: np.ndarray, x: np.ndarray, y: np.ndarray) -> Design:
    nm = arr.n_macros
    core = out.core
    out.cells = list(out.cells)
    new_cx, new_cy = arr.node_cx.copy(), arr.node_cy.copy()
    for k, node in enumerate(mov_nodes):
        j = int(node) - nm
        c = copy.copy(out.cells[j])
        w, h = c.oriented_size()
        c.x = int(np.clip(round(x[k] - 0.5 * w), core.xl, max(core.xl, core.xh - w)))
        c.y = int(np.clip(round(y[k] - 0.5 * h), core.yl, max(core.yl, core.yh - h)))
        c.placed = True
        out.cells[j] = c
        new_cx[node], new_cy[node] = c.x + 0.5 * w, c.y + 0.5 * h
    # topology unchanged: share everything except default positions (cache travels with arr)
    new_arr = replace(arr, node_cx=new_cx, node_cy=new_cy)
    if _CACHE_ATTR in arr.__dict__:
        new_arr.__dict__[_CACHE_ATTR] = arr.__dict__[_CACHE_ATTR]
    out._arrays = new_arr
    return out


def _solve(design, arr, cx, cy, orient, mov_nodes, cfg: QPConfig, spread: bool, ignore=None):
    S = _model(arr, cfg.clique_max)
    n_nodes = arr.n_nodes
    nmov = len(mov_nodes)
    var_of_node = np.full(n_nodes, -1, dtype=np.int64)
    var_of_node[mov_nodes] = np.arange(nmov)
    nvar = nmov + S.n_star

    # absolute pin positions (for anchors) and rotated offsets (for movable pins)
    odx, ody = transform_offsets(arr.pin_dx, arr.pin_dy, orient[arr.pin_node])
    px, py = cx[arr.pin_node] + odx, cy[arr.pin_node] + ody

    def term(t):
        is_star = t < 0
        tt = np.where(is_star, 0, t)
        var = np.where(is_star, nmov + (-1 - t), var_of_node[arr.pin_node[tt]])
        offx = np.where(is_star, 0.0, odx[tt])
        offy = np.where(is_star, 0.0, ody[tt])
        absx = np.where(is_star, 0.0, px[tt])
        absy = np.where(is_star, 0.0, py[tt])
        bad = np.zeros(len(t), dtype=bool) if ignore is None else (~is_star & ignore[arr.pin_node[tt]])
        return var, offx, offy, absx, absy, bad

    va, oxa, oya, axa, aya, bada = term(S.ta)
    vb, oxb, oyb, axb, ayb, badb = term(S.tb)
    w = np.where(bada | badb, 0.0, S.w)
    ma, mb = va >= 0, vb >= 0
    both = ma & mb
    ia, ib = va[both], vb[both]
    wb = w[both]
    # Q entries
    rows = [ia, ib, ia, ib]
    cols = [ia, ib, ib, ia]
    vals = [wb, wb, -wb, -wb]
    diag = np.zeros(nvar)
    np.add.at(diag, ia, wb)
    np.add.at(diag, ib, wb)
    bx = np.zeros(nvar)
    by = np.zeros(nvar)
    # (x_a + oa) - (x_b + ob): gradient terms
    np.add.at(bx, ia, -wb * (oxa[both] - oxb[both]))
    np.add.at(bx, ib, wb * (oxa[both] - oxb[both]))
    np.add.at(by, ia, -wb * (oya[both] - oyb[both]))
    np.add.at(by, ib, wb * (oya[both] - oyb[both]))
    for mv, fx, fy, ox_, oy_, vv in ((ma & ~mb, axb, ayb, oxa, oya, va), (mb & ~ma, axa, aya, oxb, oyb, vb)):
        i = vv[mv]
        wm = w[mv]
        np.add.at(diag, i, wm)
        np.add.at(bx, i, wm * (fx[mv] - ox_[mv]))
        np.add.at(by, i, wm * (fy[mv] - oy_[mv]))
    off = sp.coo_matrix((np.concatenate(vals[2:]), (np.concatenate(rows[2:]), np.concatenate(cols[2:]))), shape=(nvar, nvar)).tocsr()

    x0 = np.concatenate([cx[mov_nodes], np.full(S.n_star, design.core.cx)])
    y0 = np.concatenate([cy[mov_nodes], np.full(S.n_star, design.core.cy)])
    mean_d = float(diag[diag > 0].mean()) if (diag > 0).any() else 1.0
    reg = cfg.regularize * mean_d

    def solve(anchor_w: np.ndarray, tx: np.ndarray, ty: np.ndarray):
        d = diag + reg + anchor_w
        Q = (off + sp.diags(d)).tocsc()
        lu = spla.splu(Q)
        return lu.solve(bx + reg * x0 + anchor_w * tx), lu.solve(by + reg * y0 + anchor_w * ty)

    zero = np.zeros(nvar)
    x, y = solve(zero, x0, y0)
    x, y = _clamp(design, arr, mov_nodes, x, y)
    if not spread:
        return x[:nmov], y[:nmov]

    grid, cap = _capacity(design, arr, cx, cy, orient, mov_nodes, cfg)
    aw = cfg.anchor_start * mean_d
    for _ in range(cfg.iters):
        tx, ty = _cell_shift(arr, mov_nodes, x[:nmov], y[:nmov], grid, cap, cfg.delta)
        anchor = np.concatenate([np.full(nmov, aw), np.zeros(S.n_star)])
        x, y = solve(anchor, np.concatenate([tx, x[nmov:]]), np.concatenate([ty, y[nmov:]]))
        x, y = _clamp(design, arr, mov_nodes, x, y)
        aw *= cfg.anchor_growth
    return x[:nmov], y[:nmov]


def _clamp(design, arr, mov_nodes, x, y):
    c = design.core
    n = len(mov_nodes)
    hw, hh = 0.5 * arr.node_w[mov_nodes], 0.5 * arr.node_h[mov_nodes]
    x = x.copy()
    y = y.copy()
    x[:n] = np.clip(x[:n], c.xl + hw, np.maximum(c.xl + hw, c.xh - hw))
    y[:n] = np.clip(y[:n], c.yl + hh, np.maximum(c.yl + hh, c.yh - hh))
    x[n:] = np.clip(x[n:], c.xl, c.xh)
    y[n:] = np.clip(y[n:], c.yl, c.yh)
    return x, y


def _capacity(design, arr, cx, cy, orient, mov_nodes, cfg: QPConfig):
    n = cfg.grid or int(np.clip(round(np.sqrt(len(mov_nodes))), 4, 32))
    grid = GridSpec.over(design.core, n, n)
    fixed = np.ones(arr.n_nodes, dtype=bool)
    fixed[mov_nodes] = False
    fixed &= (arr.node_kind == KIND_MACRO) | (arr.node_kind == KIND_CELL)
    idx = np.flatnonzero(fixed)
    w, h = oriented_dims(arr.node_w[idx], arr.node_h[idx], orient[idx])
    blocked = axis_overlap(cx[idx] - w / 2, cx[idx] + w / 2, grid.x_edges).T @ axis_overlap(cy[idx] - h / 2, cy[idx] + h / 2, grid.y_edges)
    free = np.maximum(grid.bin_area - blocked, 0.0)
    mov_area = float((arr.node_w[mov_nodes] * arr.node_h[mov_nodes]).sum())
    cap = np.maximum(free * cfg.target_density, 1e-3 * grid.bin_area)
    if mov_area > cap.sum():  # over-full: share the overflow evenly
        cap *= mov_area / cap.sum()
    return grid, cap


def _cell_shift(arr, mov_nodes, x, y, grid: GridSpec, cap: np.ndarray, delta: float):
    w, h = arr.node_w[mov_nodes], arr.node_h[mov_nodes]
    a = w * h
    bx = np.clip(((x - grid.x0) / grid.bin_w).astype(np.int64), 0, grid.nx - 1)
    by = np.clip(((y - grid.y0) / grid.bin_h).astype(np.int64), 0, grid.ny - 1)
    util = np.zeros((grid.nx, grid.ny))
    np.add.at(util, (bx, by), a)
    U = util / cap
    tx = _shift_axis(x, bx, by, U, grid.x_edges, delta)
    ty = _shift_axis(y, by, bx, U.T, grid.y_edges, delta)
    return tx, ty


def _shift_axis(v, b_along, b_across, U, edges, delta):
    """Shift coordinates ``v`` along one axis; ``U[i_along, j_across]``."""
    centres = 0.5 * (edges[:-1] + edges[1:])
    n = len(centres)
    Ul, Ur = U[:-1, :], U[1:, :]  # (n-1, m)
    newb = (centres[:-1, None] * (Ur + delta) + centres[1:, None] * (Ul + delta)) / (Ul + Ur + 2 * delta)
    lo_new = np.vstack([np.full((1, U.shape[1]), edges[0]), newb])  # (n, m)
    hi_new = np.vstack([newb, np.full((1, U.shape[1]), edges[-1])])
    lo, hi = edges[b_along], edges[b_along + 1]
    t = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    l2, h2 = lo_new[b_along, b_across], hi_new[b_along, b_across]
    return l2 + t * (h2 - l2)


def cluster_hpwl(design: Design, placement: Placement | None = None) -> float:
    arr = design.arrays()
    cx, cy, o = arr.node_state(placement)
    return hpwl(arr, cx, cy, o)
