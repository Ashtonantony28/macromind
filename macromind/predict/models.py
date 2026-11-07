"""Congestion and timing predictors (plain PyTorch, CPU-sized).

* :class:`CongestionNet` -- GNN encoder on the clustered netlist, node
  embeddings scattered (mean) onto the ``G x G`` grid, concatenated with the
  image channels (RUDY, pin density, macro occupancy, ...) -> small U-Net ->
  per-bin GRT utilisation (H, V) and usage relative to the mean bin capacity
  (H, V; its sum over bins is a wirelength estimate). A scalar head on the U-Net bottleneck
  predicts ``[log(GRT WL / cluster HPWL), hot_frac]``.
* :class:`TimingNet` -- GNN on the clustered netlist whose node inputs are the
  node features plus the image channels gathered at the node's position;
  per-node head = worst endpoint slack / clock period (masked where a node
  has no endpoint), graph head (mean + max pooling) = ``[WNS/T, TNS/(T * #endpoints)]``.
* :class:`Ensemble` -- K independently seeded models; mean and std of the outputs.

The GNN layer is a weighted GraphSAGE (mean aggregation with the row-normalised
clique weights from :class:`macromind.predict.features.GraphCache`).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import IMG_CHANNELS, NODE_FEATURES, Features


def spmm(edge_index: torch.Tensor, edge_weight: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """``out[i] = sum_j w_ij h[j]`` over edges ``(i, j)``."""
    out = torch.zeros_like(h)
    return out.index_add_(0, edge_index[0], h[edge_index[1]] * edge_weight[:, None])


class SAGELayer(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.self_lin = nn.Linear(d_in, d_out)
        self.nbr_lin = nn.Linear(d_in, d_out, bias=False)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, h, ei, ew):
        return F.gelu(self.norm(self.self_lin(h) + self.nbr_lin(spmm(ei, ew, h))))


class GNN(nn.Module):
    def __init__(self, d_in: int, d: int, layers: int, dropout: float = 0.0):
        super().__init__()
        self.inp = nn.Linear(d_in, d)
        self.layers = nn.ModuleList([SAGELayer(d, d) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, x, ei, ew):
        h = F.gelu(self.inp(x))
        for lay in self.layers:
            h = h + self.drop(lay(h, ei, ew))
        return h


N_GLOBAL = 3
N_MAP = 4  #: util_h, util_v, use_h, use_v


def global_feats(f: Features) -> torch.Tensor:
    """Placement-level inputs to the scalar heads: the proxy's wirelength, density and congestion terms."""
    p = f.proxy
    return torch.tensor([p["wirelength"] * 10.0, p["density"], math.log1p(p["congestion"])], dtype=torch.float32)


def gather_img(img: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Image values at node positions: ``img (C, G, G)``, ``pos (N, 2)`` in [0, 1] -> ``(N, C)``."""
    G = img.shape[-1]
    ix = (pos[:, 0] * G).long().clamp(0, G - 1)
    iy = (pos[:, 1] * G).long().clamp(0, G - 1)
    return img[:, iy, ix].T


def scatter_grid(h: torch.Tensor, pos: torch.Tensor, G: int) -> torch.Tensor:
    """Mean of node embeddings per grid bin: ``(N, D)`` -> ``(D, G, G)``."""
    ix = (pos[:, 0] * G).long().clamp(0, G - 1)
    iy = (pos[:, 1] * G).long().clamp(0, G - 1)
    flat = iy * G + ix
    s = torch.zeros(G * G, h.shape[1]).index_add_(0, flat, h)
    c = torch.zeros(G * G).index_add_(0, flat, torch.ones_like(flat, dtype=h.dtype))
    return (s / c.clamp_min(1)[:, None]).T.reshape(h.shape[1], G, G)


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.GELU())


@dataclass
class CongestionConfig:
    gnn_dim: int = 32
    gnn_layers: int = 3
    emb_dim: int = 8  #: node-embedding channels scattered onto the grid
    base: int = 16  #: U-Net width
    dropout: float = 0.1


class CongestionNet(nn.Module):
    n_scalar = 2

    def __init__(self, cfg: CongestionConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or CongestionConfig()
        C = len(IMG_CHANNELS)
        self.gnn = GNN(len(NODE_FEATURES) + C, cfg.gnn_dim, cfg.gnn_layers, cfg.dropout)
        self.emb = nn.Linear(cfg.gnn_dim, cfg.emb_dim)
        b = cfg.base
        self.e1 = _block(C + cfg.emb_dim, b)
        self.e2 = _block(b, 2 * b)
        self.e3 = _block(2 * b, 4 * b)
        self.u2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, stride=2)
        self.d2 = _block(4 * b, 2 * b)
        self.u1 = nn.ConvTranspose2d(2 * b, b, 2, stride=2)
        self.d1 = _block(2 * b, b)
        self.out = nn.Conv2d(b, N_MAP, 1)
        self.scalar = nn.Sequential(nn.Linear(4 * b + N_GLOBAL, 32), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(32, self.n_scalar))

    def forward(self, f: Features) -> tuple[torch.Tensor, torch.Tensor]:
        """``(map (4, G, G), scalars (2,))``: H/V utilisation (~[0, 1]) and H/V usage / mean bin capacity."""
        img = f.img
        G = img.shape[-1]
        x = torch.cat([f.node_x, gather_img(img, f.pos)], 1)
        h = self.gnn(x, f.edge_index, f.edge_weight)
        grid = scatter_grid(self.emb(h), f.pos, G)
        z1 = self.e1(torch.cat([img, grid], 0)[None])
        z2 = self.e2(F.max_pool2d(z1, 2))
        z3 = self.e3(F.max_pool2d(z2, 2))
        y2 = self.d2(torch.cat([self.u2(z3), z2], 1))
        y1 = self.d1(torch.cat([self.u1(y2), z1], 1))
        m = self.out(y1)[0]
        s = self.scalar(torch.cat([z3.mean((2, 3))[0], global_feats(f)]))
        return m, s


@dataclass
class TimingConfig:
    gnn_dim: int = 48
    gnn_layers: int = 4
    dropout: float = 0.1


class TimingNet(nn.Module):
    n_scalar = 2

    def __init__(self, cfg: TimingConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or TimingConfig()
        C = len(IMG_CHANNELS)
        self.gnn = GNN(len(NODE_FEATURES) + C, cfg.gnn_dim, cfg.gnn_layers, cfg.dropout)
        self.node_head = nn.Sequential(nn.Linear(cfg.gnn_dim, 32), nn.GELU(), nn.Linear(32, 1))
        self.graph_head = nn.Sequential(nn.Linear(2 * cfg.gnn_dim + N_GLOBAL, 32), nn.GELU(), nn.Dropout(cfg.dropout),
                                        nn.Linear(32, self.n_scalar))

    def forward(self, f: Features) -> tuple[torch.Tensor, torch.Tensor]:
        """``(node slack / T (N,), [WNS/T, TNS/(T*n_ep)] (2,))``."""
        x = torch.cat([f.node_x, gather_img(f.img, f.pos)], 1)
        h = self.gnn(x, f.edge_index, f.edge_weight)
        node = self.node_head(h)[:, 0]
        g = torch.cat([h.mean(0), h.max(0).values, global_feats(f)])
        return node, self.graph_head(g)


class Ensemble(nn.Module):
    """K models of the same kind; ``forward`` returns per-output (mean, std) over members."""

    def __init__(self, members: list[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)

    @torch.no_grad()
    def forward(self, f: Features):
        self.eval()  # members are always evaluated without dropout
        outs = [m(f) for m in self.members]
        res = []
        for parts in zip(*outs):
            st = torch.stack(parts)
            res.append((st.mean(0), st.std(0, unbiased=len(parts) > 1) if len(parts) > 1 else torch.zeros_like(st[0])))
        return res


def model_config_dict(m: nn.Module) -> dict:
    return {"class": type(m).__name__, "config": asdict(m.cfg)}


def build_model(kind: str, config: dict | None = None) -> nn.Module:
    if kind == "congestion":
        return CongestionNet(CongestionConfig(**(config or {})))
    if kind == "timing":
        return TimingNet(TimingConfig(**(config or {})))
    raise ValueError(kind)
