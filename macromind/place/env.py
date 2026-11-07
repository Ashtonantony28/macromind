"""Sequential macro-placement environment (AlphaChip-style) on a G x G grid.

One movable macro is placed per step, largest area first (fixed macros are
pre-placed). The action is a grid cell ``a = row * G + col`` (row = y index);
the macro's centre goes to the cell centre, then its lower-left corner is
snapped to the site/row grid and clamped inside the core. The action mask
marks cells whose snapped, clamped footprint would overlap an already placed
macro inflated by the halo, so masked-valid actions never create overlap. If
no cell is valid the macro is placed at the nearest free spot found by the
legaliser's exact search (``info["fallback"]``), and the mask is all-true so
the policy still has a distribution.

Observation (numpy, see :data:`CHANNELS`):

* ``image`` ``(C, G, G)``: placed-macro occupancy, std-cell cluster density,
  RUDY congestion of nets over placed pins, current macro footprint (centred),
  action mask, and the *wire mask* (normalised HPWL increase of placing the
  current macro at each cell, MaskPlace-style; separable in x and y).
* ``mask`` ``(G*G,)`` bool, ``macro`` (node index of the current macro),
  ``placed`` ``(n_macros,)`` bool, ``scalars`` ``(S,)``.

Terminal step: legalise (orientation choice only, positions are already
legal), run the cluster placement pass, compute the proxy cost.
``reward = (ref - proxy) / ref`` where ``ref`` is the proxy of
:func:`~macromind.baselines.grid_place` under the same protocol, so 0 means
"as good as shelf packing" and rewards are comparable across designs.

Optional learned-predictor term (``EnvConfig.predictor``, see
:class:`macromind.predict.runtime.Predictor`): the terminal cost becomes
``proxy + pred_weight * mu + unc_weight * sigma`` where ``mu`` / ``sigma`` are the
ensemble mean / std of the predicted (standardised) congestion + timing
penalty, so uncertain placements are penalised rather than exploited. The
reference uses the same cost for grid packing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.design import Design, Placement
from ..cost.congestion import capacity_per_bin, rudy_maps
from ..cost.grid import GridSpec, axis_overlap
from ..cost.proxy import ProxyConfig, ProxyResult, proxy_cost
from ..legalize.legalizer import LegalizationError, SnapGrid, _find_spot, legalize, macro_bounds

CHANNELS = ["occupancy", "cell_density", "rudy", "footprint", "mask", "wiremask"]
N_CHANNELS = len(CHANNELS)
N_SCALARS = 5


@dataclass
class EnvConfig:
    grid: int = 32
    cluster_pass: bool = True
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    reward_clip: float = 2.0
    #: optional learned predictor with ``penalty(design, placement, scored) -> (mu, sigma)``
    predictor: object | None = None
    pred_weight: float = 0.0  #: lambda on the predicted congestion/timing penalty
    unc_weight: float = 0.0  #: kappa on the ensemble std (uncertainty penalty)


def terminal_cost(design: Design, placement: Placement, cfg: "EnvConfig", info: dict | None = None) -> tuple[ProxyResult, float]:
    """``(proxy result, cost)``: the proxy plus the optional predictor terms of ``cfg``."""
    res, scored = evaluate_placement(design, placement, proxy=cfg.proxy, cluster_pass=cfg.cluster_pass)
    cost = res.proxy
    if cfg.predictor is not None and (cfg.pred_weight or cfg.unc_weight):
        mu, sigma = cfg.predictor.penalty(design, placement, scored)
        cost = cost + cfg.pred_weight * mu + cfg.unc_weight * sigma
        if info is not None:
            info["pred_penalty"], info["pred_std"] = float(mu), float(sigma)
    return res, float(cost)


def evaluate_placement(design: Design, placement: Placement, *, proxy: ProxyConfig | None = None, cluster_pass: bool = True) -> tuple[ProxyResult, Design]:
    """The shared evaluation protocol: optional cluster pass, then proxy cost."""
    scored = design
    if cluster_pass:
        from ..graph.quadratic import place_clusters

        scored = place_clusters(design, placement)
    return proxy_cost(scored, placement, proxy), scored


def reference_proxy(design: Design, cfg: EnvConfig | None = None) -> float:
    """Cost of grid packing under the env's protocol (reward normaliser; proxy unless a predictor is set)."""
    from ..baselines.placers import grid_place

    cfg = cfg or EnvConfig()
    return terminal_cost(design, grid_place(design), cfg)[1]


class PlacementEnv:
    def __init__(self, design: Design, config: EnvConfig | None = None, reference: float | None = None):
        self.design = design
        self.cfg = cfg = config or EnvConfig()
        self.arr = arr = design.arrays()
        G = self.G = cfg.grid
        c = design.core
        self.grid = GridSpec(G, G, float(c.xl), float(c.yl), float(c.width), float(c.height))
        nm = arr.n_macros
        self.n_macros = nm
        self.fixed = np.array([m.fixed for m in design.macros], dtype=bool)
        mov = np.flatnonzero(~self.fixed)
        area = arr.node_w[mov] * arr.node_h[mov]
        self.order = mov[np.argsort(-area, kind="stable")]
        self.n_steps = len(self.order)
        self.w = arr.node_w[:nm].astype(np.int64)
        self.h = arr.node_h[:nm].astype(np.int64)
        self.snap = SnapGrid.of(design)
        self.bounds = macro_bounds(design, self.w, self.h, self.snap)
        ce = self.grid.x0 + self.grid.bin_w * (np.arange(G) + 0.5)
        re = self.grid.y0 + self.grid.bin_h * (np.arange(G) + 0.5)
        # candidate lower-left corners per macro: (nm, G)
        self.cand_x = np.clip(self.snap.near_x(ce[None, :] - self.w[:, None] / 2), self.bounds[0][:, None], self.bounds[1][:, None])
        self.cand_y = np.clip(self.snap.near_y(re[None, :] - self.h[:, None] / 2), self.bounds[2][:, None], self.bounds[3][:, None])
        self.reference = float(reference) if reference is not None else reference_proxy(design, cfg)
        # static channels
        g = self.grid
        cells = np.flatnonzero(arr.node_kind == 1)
        cx, cy = arr.node_cx[cells], arr.node_cy[cells]
        cw, ch = arr.node_w[cells], arr.node_h[cells]
        self.cell_density = (axis_overlap(cx - cw / 2, cx + cw / 2, g.x_edges).T @ axis_overlap(cy - ch / 2, cy + ch / 2, g.y_edges)).T / g.bin_area
        dbu = float(design.dbu_per_micron)
        self.h_cap, self.v_cap = capacity_per_bin(g, design.hroutes_per_micron / dbu, design.vroutes_per_micron / dbu)
        self.pin_dx, self.pin_dy = arr.pin_dx, arr.pin_dy  # orientation N during the episode
        self.reset()

    # --------------------------------------------------------------- state
    def reset(self) -> dict:
        d = self.design
        self.t = 0
        self.x = np.array([m.x for m in d.macros], dtype=np.int64)
        self.y = np.array([m.y for m in d.macros], dtype=np.int64)
        self.placed = self.fixed.copy()
        self.node_placed = np.ones(self.arr.n_nodes, dtype=bool)
        self.node_placed[: self.n_macros] = self.placed
        self.cx = self.arr.node_cx.copy()
        self.cy = self.arr.node_cy.copy()
        self.done = self.n_steps == 0
        self.result: ProxyResult | None = None
        self.fallbacks = 0
        return self.observe()

    @property
    def current(self) -> int:
        return int(self.order[self.t]) if self.t < self.n_steps else -1

    def valid_mask(self, i: int | None = None) -> np.ndarray:
        """(G*G,) bool: cells where macro ``i`` (default current) can go without overlap."""
        i = self.current if i is None else i
        G, halo = self.G, self.design.halo
        X, Y = self.cand_x[i], self.cand_y[i]
        k = np.flatnonzero(self.placed)
        if len(k) == 0:
            return np.ones(G * G, dtype=bool)
        xl, yl = self.x[k] - halo, self.y[k] - halo
        xh, yh = self.x[k] + self.w[k] + halo, self.y[k] + self.h[k] + halo
        ox = (X[:, None] < xh[None, :]) & (X[:, None] + self.w[i] > xl[None, :])  # (G, K)
        oy = (Y[:, None] < yh[None, :]) & (Y[:, None] + self.h[i] > yl[None, :])
        occ = (oy.astype(np.float32) @ ox.T.astype(np.float32)) > 0  # (Gy, Gx)
        return ~occ.ravel()

    def _wiremask(self, i: int) -> np.ndarray:
        arr = self.arr
        nets = arr.node_net_idx[arr.node_net_ptr[i] : arr.node_net_ptr[i + 1]]
        G = self.G
        if len(nets) == 0:
            return np.zeros((G, G))
        idx, seg = arr.pins_of_nets(nets)
        node = arr.pin_node[idx]
        ok = self.node_placed[node] & (node != i)
        px = self.cx[node] + self.pin_dx[idx]
        py = self.cy[node] + self.pin_dy[idx]
        x0 = np.minimum.reduceat(np.where(ok, px, np.inf), seg)
        x1 = np.maximum.reduceat(np.where(ok, px, -np.inf), seg)
        y0 = np.minimum.reduceat(np.where(ok, py, np.inf), seg)
        y1 = np.maximum.reduceat(np.where(ok, py, -np.inf), seg)
        has = np.isfinite(x0)
        # one representative pin of macro i per net
        mine = arr.pin_node[idx] == i
        first = np.full(len(nets), -1)
        pos = np.flatnonzero(mine)
        netpos = np.searchsorted(seg, pos, side="right") - 1
        first[netpos[::-1]] = idx[pos[::-1]]
        sel = has & (first >= 0)
        if not sel.any():
            return np.zeros((G, G))
        w = arr.net_weight[nets][sel]
        dx, dy = self.pin_dx[first[sel]], self.pin_dy[first[sel]]
        X = self.cand_x[i] + self.w[i] / 2  # (G,) centres
        Y = self.cand_y[i] + self.h[i] / 2
        PX = X[None, :] + dx[:, None]
        PY = Y[None, :] + dy[:, None]
        fx = (w[:, None] * (np.maximum(x0[sel][:, None] - PX, 0) + np.maximum(PX - x1[sel][:, None], 0))).sum(0)
        fy = (w[:, None] * (np.maximum(y0[sel][:, None] - PY, 0) + np.maximum(PY - y1[sel][:, None], 0))).sum(0)
        return fy[:, None] + fx[None, :]  # (Gy, Gx)

    def _rudy(self) -> np.ndarray:
        arr = self.arr
        if arr.n_nets == 0:
            return np.zeros((self.G, self.G))
        ok = self.node_placed[arr.pin_node]
        px = self.cx[arr.pin_node] + self.pin_dx
        py = self.cy[arr.pin_node] + self.pin_dy
        s = arr.net_ptr[:-1]
        x0 = np.minimum.reduceat(np.where(ok, px, np.inf), s)
        x1 = np.maximum.reduceat(np.where(ok, px, -np.inf), s)
        y0 = np.minimum.reduceat(np.where(ok, py, np.inf), s)
        y1 = np.maximum.reduceat(np.where(ok, py, -np.inf), s)
        cnt = np.add.reduceat(ok.astype(np.int64), s)
        k = cnt >= 2
        h, v = rudy_maps(x0[k], x1[k], y0[k], y1[k], arr.net_weight[k], self.grid)
        return (0.5 * (h / self.h_cap + v / self.v_cap)).T

    def observe(self) -> dict:
        G, g = self.G, self.grid
        img = np.zeros((N_CHANNELS, G, G), dtype=np.float32)
        k = np.flatnonzero(self.placed)
        if len(k):
            occ = axis_overlap(self.x[k], self.x[k] + self.w[k], g.x_edges).T @ axis_overlap(self.y[k], self.y[k] + self.h[k], g.y_edges)
            img[0] = np.minimum(occ.T / g.bin_area, 1.0)
        img[1] = np.minimum(self.cell_density, 2.0)
        img[2] = np.minimum(self._rudy() / 2.0, 3.0)
        i = self.current
        scal = np.zeros(N_SCALARS, dtype=np.float32)
        if i >= 0:
            mask = self.valid_mask(i)
            if not mask.any():
                mask = np.ones_like(mask)
            fw, fh = self.w[i] / g.bin_w, self.h[i] / g.bin_h
            xs = axis_overlap(np.array([G / 2 - fw / 2]), np.array([G / 2 + fw / 2]), np.arange(G + 1.0))[0]
            ys = axis_overlap(np.array([G / 2 - fh / 2]), np.array([G / 2 + fh / 2]), np.arange(G + 1.0))[0]
            img[3] = ys[:, None] * xs[None, :]
            img[4] = mask.reshape(G, G)
            wm = self._wiremask(i)
            valid = mask.reshape(G, G)
            lo, hi = wm[valid].min(), wm[valid].max()
            img[5] = np.where(valid, (wm - lo) / (hi - lo) if hi > lo else 0.0, 1.0)
            c = self.design.core
            scal[:] = [self.t / max(self.n_steps, 1), self.w[i] / c.width, self.h[i] / c.height, np.sqrt(self.w[i] * self.h[i] / c.area), self.n_steps / 50]
        else:
            mask = np.ones(G * G, dtype=bool)
        return {"image": img, "mask": mask, "macro": i, "placed": self.placed.copy(), "scalars": scal}

    # ----------------------------------------------------------------- step
    def decode(self, action: int, i: int | None = None) -> tuple[int, int]:
        i = self.current if i is None else i
        r, col = divmod(int(action), self.G)
        return int(self.cand_x[i, col]), int(self.cand_y[i, r])

    def step(self, action: int):
        if self.done:
            raise RuntimeError("episode is done; call reset()")
        i = self.current
        mask = self.valid_mask(i)
        info: dict = {}
        if mask.any():
            if not mask[int(action)]:
                raise ValueError(f"action {action} is masked for macro {i}")
            x, y = self.decode(action, i)
        else:
            spot = self._fallback(i)
            info["fallback"] = True
            self.fallbacks += 1
            x, y = spot if spot is not None else self.decode(action, i)
        self.x[i], self.y[i] = x, y
        self.placed[i] = True
        self.node_placed[i] = True
        self.cx[i], self.cy[i] = x + self.w[i] / 2, y + self.h[i] / 2
        self.t += 1
        reward = 0.0
        if self.t >= self.n_steps:
            self.done = True
            reward = self._terminal(info)
        return self.observe(), reward, self.done, info

    def _fallback(self, i: int):
        halo = self.design.halo
        k = np.flatnonzero(self.placed)
        obs = np.stack([self.x[k] - halo, self.y[k] - halo, self.x[k] + self.w[k] + halo, self.y[k] + self.h[k] + halo], 1)
        b = tuple(int(v[i]) for v in self.bounds)
        c = self.design.core
        return _find_spot(c.cx - self.w[i] / 2, c.cy - self.h[i] / 2, int(self.w[i]), int(self.h[i]), b, obs, self.snap)

    def placement(self) -> Placement:
        return Placement(self.x.copy(), self.y.copy(), np.zeros(self.n_macros, dtype=np.int8) + np.array([m.orient if m.fixed else 0 for m in self.design.macros], dtype=np.int8))

    def _terminal(self, info: dict) -> float:
        try:
            p = legalize(self.design, self.placement())
        except LegalizationError:
            info["illegal"] = True
            self.final_placement = self.placement()
            self.result = None
            return -self.cfg.reward_clip
        res, cost = terminal_cost(self.design, p, self.cfg, info)
        self.final_placement = p
        self.result = res
        info["proxy"] = res
        info["cost"] = cost
        r = (self.reference - cost) / self.reference
        return float(np.clip(r, -self.cfg.reward_clip, self.cfg.reward_clip))
