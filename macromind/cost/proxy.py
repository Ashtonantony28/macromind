"""Circuit-Training/TILOS-style proxy cost and its incremental evaluator.

``proxy = w_wl * WL + w_d * density + w_c * congestion`` with

* ``WL = weighted HPWL / ((core_w + core_h) * sum(net weights))``
  (Circuit Training's wirelength normalisation),
* ``density``: either the area above ``target_density`` normalised by total
  node area (``density_mode="overflow"``, default) or the mean of the top
  ``density_topk`` bin utilisations (``"topk"``, Circuit Training style),
* ``congestion``: mean of the top ``congestion_topk`` fraction of RUDY
  demand/capacity ratios over both routing directions.

:class:`IncrementalProxy` keeps per-net bounding boxes and the density and
RUDY maps so that moving one or two macros costs ``O(pins of touched nets +
bins)`` instead of a full re-evaluation; this is what simulated annealing uses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..core.design import Design, DesignArrays, Placement
from ..core.geometry import oriented_dims, transform_offsets
from .congestion import capacity_per_bin, expand_bboxes
from .density import area_nodes, node_rects, topk_mean
from .grid import GridSpec, axis_overlap
from .wirelength import net_bboxes, pin_positions


@dataclass
class ProxyConfig:
    """Proxy-cost parameters."""

    grid_nx: int = 32
    grid_ny: int | None = None
    target_density: float = 0.6
    density_mode: str = "overflow"  # "overflow" | "topk"
    density_topk: float = 0.1
    congestion_topk: float = 0.05
    w_wirelength: float = 1.0
    w_density: float = 0.5
    w_congestion: float = 0.5

    def grid(self, design_or_arrays: Design | DesignArrays) -> GridSpec:
        return GridSpec.over(design_or_arrays.core, self.grid_nx, self.grid_ny)


@dataclass
class ProxyResult:
    """Proxy cost components."""

    hpwl: float  # weighted HPWL, DBU
    wirelength: float  # normalised
    density: float
    congestion: float
    proxy: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"proxy={self.proxy:.5f} (wl={self.wirelength:.5f} density={self.density:.5f} "
            f"congestion={self.congestion:.5f}; hpwl={self.hpwl:.4g} DBU)"
        )


def _caps(design: Design, grid: GridSpec) -> tuple[float, float]:
    dbu = float(design.dbu_per_micron)
    return capacity_per_bin(grid, design.hroutes_per_micron / dbu, design.vroutes_per_micron / dbu)


class IncrementalProxy:
    """Proxy cost with cheap incremental updates for moving a few nodes.

    Usage::

        inc = IncrementalProxy(design, placement)
        new = inc.propose([i], [cx], [cy], [orient])   # does not change state
        if accept: inc.accept() else: inc.reject()

    Exactly one proposal may be pending at a time; :meth:`propose` must be
    followed by :meth:`accept` or :meth:`reject`.
    """

    def __init__(self, design: Design, placement: Placement | None = None, config: ProxyConfig | None = None):
        self.design = design
        self.arr = arr = design.arrays()
        self.cfg = cfg = config or ProxyConfig()
        self.grid = grid = cfg.grid(design)
        self.h_cap, self.v_cap = _caps(design, grid)
        self.min_extent = (0.5 * grid.bin_w, 0.5 * grid.bin_h)
        self.xe, self.ye = grid.x_edges, grid.y_edges
        self.wsum = float(arr.net_weight.sum()) if arr.n_nets else 1.0
        self.wl_norm = (design.core.width + design.core.height) * max(self.wsum, 1e-12)
        an = area_nodes(arr)
        self.is_area = np.zeros(arr.n_nodes, dtype=bool)
        self.is_area[an] = True
        self.total_area = float((arr.node_w[an] * arr.node_h[an]).sum())
        self.use_density = cfg.w_density != 0
        self.use_cong = cfg.w_congestion != 0
        self._cache: dict[int, tuple] = {}
        self.cx, self.cy, self.orient = arr.node_state(placement if placement is not None else design.placement())
        self._pending: tuple | None = None
        self.resync()

    # ------------------------------------------------------------------ full
    def resync(self) -> ProxyResult:
        """Recompute every cached quantity from scratch (removes float drift)."""
        arr = self.arr
        self.px, self.py = pin_positions(arr, self.cx, self.cy, self.orient)
        self.bx0, self.bx1, self.by0, self.by1 = net_bboxes(arr, self.px, self.py)
        self.net_wl = (self.bx1 - self.bx0) + (self.by1 - self.by0)
        self.hpwl = float(np.dot(arr.net_weight, self.net_wl))
        an = np.flatnonzero(self.is_area)
        xl, yl, xh, yh = node_rects(arr, self.cx, self.cy, self.orient, an)
        self.area_map = (axis_overlap(xl, xh, self.xe)).T @ axis_overlap(yl, yh, self.ye)
        self.rh, self.rv = self._rudy(self.bx0, self.bx1, self.by0, self.by1, arr.net_weight)
        self._pending = None
        self.result = self._components(self.hpwl, self.area_map, self.rh, self.rv)
        return self.result

    def _rudy(self, x0, x1, y0, y1, w):
        x0, x1, y0, y1 = expand_bboxes(x0, x1, y0, y1, *self.min_extent)
        ox = axis_overlap(x0, x1, self.xe)
        oy = axis_overlap(y0, y1, self.ye)
        return (ox * (w / (y1 - y0))[:, None]).T @ oy, (ox * (w / (x1 - x0))[:, None]).T @ oy

    def _components(self, hpwl: float, area_map, rh, rv) -> ProxyResult:
        cfg, g = self.cfg, self.grid
        wl = hpwl / self.wl_norm
        util = area_map / g.bin_area
        if cfg.density_mode == "overflow":
            dens = float(np.maximum(util - cfg.target_density, 0.0).sum() * g.bin_area / max(self.total_area, 1e-12))
        elif cfg.density_mode == "topk":
            dens = topk_mean(util, cfg.density_topk)
        else:
            raise ValueError(f"unknown density_mode {cfg.density_mode!r}")
        cong = topk_mean(np.concatenate([(rh / self.h_cap).ravel(), (rv / self.v_cap).ravel()]), cfg.congestion_topk)
        proxy = cfg.w_wirelength * wl + cfg.w_density * dens + cfg.w_congestion * cong
        return ProxyResult(hpwl, wl, dens, cong, proxy)

    @property
    def cost(self) -> float:
        return self.result.proxy

    # ----------------------------------------------------------- incremental
    def _node_info(self, i: int):
        info = self._cache.get(i)
        if info is None:
            arr = self.arr
            nets = arr.node_net_idx[arr.node_net_ptr[i] : arr.node_net_ptr[i + 1]]
            idx, seg = arr.pins_of_nets(nets)
            pins = arr.node_pin_idx[arr.node_pin_ptr[i] : arr.node_pin_ptr[i + 1]]
            info = (nets, idx, seg, pins, arr.pin_dx[pins], arr.pin_dy[pins])
            self._cache[i] = info
        return info

    def propose(self, nodes, cx, cy, orient) -> float:
        """Evaluate moving ``nodes`` to centres ``(cx, cy)`` / ``orient``; returns the new proxy."""
        if self._pending is not None:
            raise RuntimeError("a proposal is already pending; call accept() or reject()")
        arr = self.arr
        nodes = np.atleast_1d(np.asarray(nodes, dtype=np.int64))
        cx = np.atleast_1d(np.asarray(cx, dtype=np.float64))
        cy = np.atleast_1d(np.asarray(cy, dtype=np.float64))
        orient = np.atleast_1d(np.asarray(orient, dtype=np.int8))
        if len(nodes) == 1:
            nets, idx, seg, pins, pdx, pdy = self._node_info(int(nodes[0]))
            ox, oy = transform_offsets(pdx, pdy, int(orient[0]))
            npx, npy = ox + cx[0], oy + cy[0]
        else:
            nets = arr.nets_of_nodes(nodes)
            idx, seg = arr.pins_of_nets(nets)
            infos = [self._node_info(int(i)) for i in nodes]
            pins = np.concatenate([f[3] for f in infos])
            pl = [len(f[3]) for f in infos]
            o = np.repeat(orient, pl)
            ox, oy = transform_offsets(np.concatenate([f[4] for f in infos]), np.concatenate([f[5] for f in infos]), o)
            npx, npy = ox + np.repeat(cx, pl), oy + np.repeat(cy, pl)

        old_px, old_py = self.px[pins].copy(), self.py[pins].copy()
        self.px[pins] = npx
        self.py[pins] = npy
        if len(nets):
            gx, gy = self.px[idx], self.py[idx]
            nx0 = np.minimum.reduceat(gx, seg)
            nx1 = np.maximum.reduceat(gx, seg)
            ny0 = np.minimum.reduceat(gy, seg)
            ny1 = np.maximum.reduceat(gy, seg)
            w = arr.net_weight[nets]
            nwl = (nx1 - nx0) + (ny1 - ny0)
            new_hpwl = self.hpwl + float(np.dot(w, nwl - self.net_wl[nets]))
        else:
            nx0 = nx1 = ny0 = ny1 = nwl = w = np.zeros(0)
            new_hpwl = self.hpwl

        rh, rv = self.rh, self.rv
        if self.use_cong and len(nets):
            sx0 = np.concatenate([self.bx0[nets], nx0])
            sx1 = np.concatenate([self.bx1[nets], nx1])
            sy0 = np.concatenate([self.by0[nets], ny0])
            sy1 = np.concatenate([self.by1[nets], ny1])
            dh, dv = self._rudy(sx0, sx1, sy0, sy1, np.concatenate([-w, w]))
            rh, rv = rh + dh, rv + dv

        amap = self.area_map
        if self.use_density:
            mov = nodes[self.is_area[nodes]]
            if len(mov):
                ww, hh = oriented_dims(arr.node_w[mov], arr.node_h[mov], self.orient[mov])
                nsel = self.is_area[nodes]
                nw, nh = oriented_dims(arr.node_w[mov], arr.node_h[mov], orient[nsel])
                ocx, ocy, ncx, ncy = self.cx[mov], self.cy[mov], cx[nsel], cy[nsel]
                lo_x = np.concatenate([ocx - 0.5 * ww, ncx - 0.5 * nw])
                hi_x = np.concatenate([ocx + 0.5 * ww, ncx + 0.5 * nw])
                lo_y = np.concatenate([ocy - 0.5 * hh, ncy - 0.5 * nh])
                hi_y = np.concatenate([ocy + 0.5 * hh, ncy + 0.5 * nh])
                sgn = np.concatenate([-np.ones(len(mov)), np.ones(len(mov))])
                amap = amap + (axis_overlap(lo_x, hi_x, self.xe) * sgn[:, None]).T @ axis_overlap(lo_y, hi_y, self.ye)

        res = self._components(new_hpwl, amap, rh, rv)
        self._pending = (nodes, cx, cy, orient, pins, old_px, old_py, nets, nx0, nx1, ny0, ny1, nwl, new_hpwl, amap, rh, rv, res)
        return res.proxy

    @property
    def pending_result(self) -> ProxyResult:
        if self._pending is None:
            raise RuntimeError("no pending proposal")
        return self._pending[-1]

    def accept(self) -> ProxyResult:
        """Commit the pending proposal."""
        p = self._pending
        if p is None:
            raise RuntimeError("no pending proposal")
        nodes, cx, cy, orient, _pins, _opx, _opy, nets, nx0, nx1, ny0, ny1, nwl, new_hpwl, amap, rh, rv, res = p
        self.cx[nodes], self.cy[nodes], self.orient[nodes] = cx, cy, orient
        if len(nets):
            self.bx0[nets], self.bx1[nets], self.by0[nets], self.by1[nets] = nx0, nx1, ny0, ny1
            self.net_wl[nets] = nwl
        self.hpwl, self.area_map, self.rh, self.rv = new_hpwl, amap, rh, rv
        self.result = res
        self._pending = None
        return res

    def reject(self) -> None:
        """Discard the pending proposal."""
        p = self._pending
        if p is None:
            raise RuntimeError("no pending proposal")
        pins, opx, opy = p[4], p[5], p[6]
        self.px[pins] = opx
        self.py[pins] = opy
        self._pending = None

    # --------------------------------------------------------------- helpers
    def placement(self) -> Placement:
        """Current macro placement (rounded to DBU)."""
        return self.arr.macro_centres_to_placement(self.cx, self.cy, self.orient)


def proxy_cost(design: Design, placement: Placement | None = None, config: ProxyConfig | None = None) -> ProxyResult:
    """Full proxy-cost evaluation of ``design`` (macros taken from ``placement`` if given)."""
    return IncrementalProxy(design, placement, config).result


def proxy_maps(design: Design, placement: Placement | None = None, config: ProxyConfig | None = None) -> dict[str, np.ndarray]:
    """Utilisation, horizontal and vertical congestion maps, each ``(nx, ny)``."""
    inc = IncrementalProxy(design, placement, config)
    return {
        "utilization": inc.area_map / inc.grid.bin_area,
        "congestion_h": inc.rh / inc.h_cap,
        "congestion_v": inc.rv / inc.v_cap,
    }
