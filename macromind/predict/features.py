"""Placement -> predictor inputs (image channels + node features on the clustered netlist).

Everything is computed from what a macro placer knows at decision time: the
clustered netlist and the macro placement. Std-cell clusters are placed by the
same quadratic cluster pass the proxy uses (:func:`macromind.graph.place_clusters`),
so in the RL loop these features come almost for free (the env already runs it).

Image channels (``IMG_CHANNELS``, each ``(G, G)`` over the core, row = y):

* ``macro_occ``  fraction of the bin covered by macros
* ``cell_den``   log1p(std-cell cluster area / bin area)
* ``rudy_h`` / ``rudy_v``  log1p(RUDY demand / routing capacity), as in the proxy
* ``pin_den``    log1p(pins per bin / mean over bins)
* ``macro_pin``  log1p(macro pins per bin / mean), likewise
* ``coord_x`` / ``coord_y``  bin centre in [0, 1] (CoordConv)

Node features (``NODE_FEATURES``): the static graph features of
:func:`macromind.graph.build_graph` (kind, size, degrees, fixed) plus the
node's normalised position for *every* node (macros included), plus the
node's distance to the nearest macro edge (core-normalised). The models also
gather the image channels at each node's position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..core.design import Design, Placement

IMG_CHANNELS = ["macro_occ", "cell_den", "rudy_h", "rudy_v", "pin_den", "macro_pin", "coord_x", "coord_y"]
NODE_FEATURES = ["is_macro", "is_cell", "is_port", "w", "h", "sqrt_area", "log_pins", "log_degree", "fixed", "x", "y",
                 "d_macro"]


@dataclass
class Features:
    """Model inputs for one placement (torch tensors, CPU)."""

    img: torch.Tensor  #: (C, G, G)
    node_x: torch.Tensor  #: (N, F)
    pos: torch.Tensor  #: (N, 2) in [0, 1] (x, y) over the core
    edge_index: torch.Tensor  #: (2, E)
    edge_weight: torch.Tensor  #: (E,) row-normalised
    n_macros: int
    hpwl_um: float  #: cluster-level HPWL of the scored design (WL normaliser)
    proxy: dict  #: proxy components on the scored design

    def flipped(self, fx: bool, fy: bool) -> "Features":
        """Mirror the whole layout (inputs only; labels must be flipped the same way)."""
        img, pos, nx = self.img, self.pos.clone(), self.node_x.clone()
        if fx:
            img = img.flip(-1)
            pos[:, 0] = 1 - pos[:, 0]
            nx[:, 9] = 1 - nx[:, 9]
        if fy:
            img = img.flip(-2)
            pos[:, 1] = 1 - pos[:, 1]
            nx[:, 10] = 1 - nx[:, 10]
        img = img.clone()
        # coordinate channels stay absolute
        G = img.shape[-1]
        img[6] = _coord(G)[0]
        img[7] = _coord(G)[1]
        return Features(img, nx, pos, self.edge_index, self.edge_weight, self.n_macros, self.hpwl_um, self.proxy)


def _coord(G: int) -> tuple[torch.Tensor, torch.Tensor]:
    c = (torch.arange(G, dtype=torch.float32) + 0.5) / G
    return c[None, :].expand(G, G).clone(), c[:, None].expand(G, G).clone()


class GraphCache:
    """Static (placement-independent) graph tensors of a clustered design, built once."""

    def __init__(self, design: Design):
        from ..graph.tensors import build_graph

        g = build_graph(design, net_model="clique", clique_max=16)
        n = g.n_nodes
        ei, ew = g.edge_index, g.edge_weight
        deg = torch.zeros(n).index_add_(0, ei[0], ew)
        self.edge_index = ei
        self.edge_weight = (ew / deg.clamp_min(1e-9)[ei[0]]).float()  # row-normalised mean aggregation
        self.static = g.x[:, [0, 1, 2, 4, 5, 6, 7, 8, 9]].clone()
        self.n_macros = g.n_macros
        self.n = n


def featurize(design: Design, placement: Placement, *, G: int = 32, scored: Design | None = None,
              graph: GraphCache | None = None) -> Features:
    """Features of ``placement`` on the clustered ``design`` (runs the cluster pass unless ``scored`` is given)."""
    from ..cost.density import node_rects
    from ..cost.grid import splat_rects
    from ..cost.proxy import IncrementalProxy, ProxyConfig
    from ..cost.wirelength import pin_positions
    from ..graph.quadratic import place_clusters

    if scored is None:
        scored = place_clusters(design, placement)
    graph = graph or GraphCache(design)
    cfg = ProxyConfig(grid_nx=G, grid_ny=G)
    inc = IncrementalProxy(scored, placement, cfg)
    arr, grid = inc.arr, inc.grid
    nm = arr.n_macros
    cx, cy, orient = inc.cx, inc.cy, inc.orient
    ba = grid.bin_area
    mx = np.arange(nm)
    if nm:
        xl, yl, xh, yh = node_rects(arr, cx, cy, orient, mx)
        macro_occ = splat_rects(xl, yl, xh, yh, np.ones(nm), grid) / ba
    else:
        macro_occ = np.zeros((G, G))
    cells = np.arange(nm, nm + arr.n_cells)
    xl2, yl2, xh2, yh2 = node_rects(arr, cx, cy, orient, cells)
    cell_den = splat_rects(xl2, yl2, xh2, yh2, np.ones(len(cells)), grid) / ba
    px, py = inc.px, inc.py
    core = scored.core
    bx = np.clip(((px - core.xl) / core.width * G).astype(int), 0, G - 1)
    by = np.clip(((py - core.yl) / core.height * G).astype(int), 0, G - 1)
    pin = np.zeros((G, G))
    np.add.at(pin, (bx, by), 1.0)
    mp = arr.pin_node < nm
    mpin = np.zeros((G, G))
    np.add.at(mpin, (bx[mp], by[mp]), 1.0)
    pin /= max(pin.mean(), 1e-9)
    mpin /= max(mpin.mean(), 1e-9) if mp.any() else 1.0
    # heavy-tailed channels are log-compressed
    chans = [macro_occ, np.log1p(cell_den), np.log1p(inc.rh / inc.h_cap), np.log1p(inc.rv / inc.v_cap), np.log1p(pin),
             np.log1p(mpin)]
    img = np.stack([c.T for c in chans]).astype(np.float32)  # (C, ny, nx)
    cxg, cyg = _coord(G)
    img_t = torch.cat([torch.from_numpy(img), cxg[None], cyg[None]], 0)

    n = graph.n
    posx = ((cx - core.xl) / core.width).astype(np.float32)
    posy = ((cy - core.yl) / core.height).astype(np.float32)
    pos = torch.from_numpy(np.stack([posx, posy], 1)).clamp(0, 1)
    # distance from each node centre to the nearest macro rectangle (core-normalised)
    if nm:
        dx = np.maximum(0, np.maximum(xl[None, :] - cx[:, None], cx[:, None] - xh[None, :])) / core.width
        dy = np.maximum(0, np.maximum(yl[None, :] - cy[:, None], cy[:, None] - yh[None, :])) / core.height
        dmac = np.sqrt(dx ** 2 + dy ** 2).min(1).astype(np.float32)
    else:
        dmac = np.ones(n, np.float32)
    nx = torch.cat([graph.static, pos, torch.from_numpy(dmac)[:, None]], 1)
    assert nx.shape == (n, len(NODE_FEATURES)), nx.shape
    res = inc.result
    return Features(img_t, nx, pos, graph.edge_index, graph.edge_weight, nm, float(inc.hpwl) / scored.dbu_per_micron,
                    res.as_dict())
