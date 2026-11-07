"""Policy/value network: plain-torch GNN encoder + CNN over the grid state.

* **Graph encoder**: an input MLP and ``gnn_layers`` weighted GraphSAGE layers
  (``h <- h + ReLU(LN(W_s h + W_n mean_w(h_nbr)))``) over the static netlist
  graph. It runs once per design per forward pass, not per state.
* **Context**: current macro embedding, mean graph embedding, mean embedding of
  the macros placed so far, and the env scalars -> MLP -> context vector.
* **Grid trunk** (small U-Net): 3x3 conv stem at G x G, a stride-2 conv to
  G/2, dilated 3x3 residual blocks (dilations 1, 2, 4: receptive field covers
  the grid) FiLM-modulated by the context, a transposed conv back to G x G
  concatenated with the stem features.
* **Policy head**: 1x1 convs -> one logit per cell, plus a learnable prior
  ``-wm_scale * wiremask`` (MaskPlace-style; initialised at 2, so an
  untrained policy already leans towards short wires). Masked cells get
  ``-1e9`` (finite, so softmax stays NaN-free and their probability is 0).
* **Value head**: global average + max pool of the trunk plus context -> MLP.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..graph.tensors import N_FEATURES, GraphTensors
from .env import N_CHANNELS, N_SCALARS

MASK_VALUE = -1e9


@dataclass
class PolicyConfig:
    grid: int = 32
    hidden: int = 64
    gnn_layers: int = 3
    conv_ch: int = 32
    dilations: tuple[int, ...] = (1, 2, 4)
    wiremask_prior: float = 2.0

    def as_dict(self) -> dict:
        return asdict(self)


class SAGELayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.self_lin = nn.Linear(d, d)
        self.nbr_lin = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        agg = torch.zeros_like(h).index_add_(0, dst, h[src] * w[:, None])
        deg = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype).index_add_(0, dst, w)
        agg = agg / deg.clamp_min(1e-6)[:, None]
        return h + F.relu(self.norm(self.self_lin(h) + self.nbr_lin(agg)))


class GraphEncoder(nn.Module):
    def __init__(self, hidden: int, layers: int):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(N_FEATURES, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList(SAGELayer(hidden) for _ in range(layers))

    def forward(self, g: GraphTensors) -> torch.Tensor:
        # log-scale edge weights (bus weights span 1..32+)
        w = torch.log1p(g.edge_weight)
        h = self.inp(g.x)
        for layer in self.layers:
            h = layer(h, g.edge_index, w)
        return h


class FiLMBlock(nn.Module):
    def __init__(self, ch: int, ctx: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.film = nn.Linear(ctx, 2 * ch)
        self.norm = nn.GroupNorm(4, ch)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(ctx).chunk(2, dim=-1)
        y = F.relu(self.norm(self.conv1(x)))
        y = y * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return x + self.conv2(F.relu(y))


class PolicyNet(nn.Module):
    def __init__(self, config: PolicyConfig | None = None):
        super().__init__()
        self.cfg = cfg = config or PolicyConfig()
        H, C = cfg.hidden, cfg.conv_ch
        self.gnn = GraphEncoder(H, cfg.gnn_layers)
        self.ctx = nn.Sequential(nn.Linear(3 * H + N_SCALARS, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
        S = max(8, C // 2)
        self.stem = nn.Conv2d(N_CHANNELS, S, 3, padding=1)
        self.down = nn.Conv2d(S, C, 3, stride=2, padding=1)
        self.blocks = nn.ModuleList(FiLMBlock(C, H, d) for d in cfg.dilations)
        self.up = nn.ConvTranspose2d(C, S, 2, stride=2)
        self.head = nn.Sequential(nn.Conv2d(2 * S, S, 3, padding=1), nn.ReLU(), nn.Conv2d(S, 1, 1))
        self.wm_scale = nn.Parameter(torch.tensor(float(cfg.wiremask_prior)))
        self.value = nn.Sequential(nn.Linear(2 * C + H, H), nn.ReLU(), nn.Linear(H, 1))

    def encode(self, g: GraphTensors) -> tuple[torch.Tensor, torch.Tensor]:
        node = self.gnn(g)
        return node, node.mean(0)

    def forward(self, enc: tuple[torch.Tensor, torch.Tensor], image: torch.Tensor, mask: torch.Tensor,
                macro: torch.Tensor, placed: torch.Tensor, scalars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Masked logits ``(B, G*G)`` and values ``(B,)``. ``placed`` is ``(B, n_macros)`` bool."""
        node, gmean = enc
        B = image.shape[0]
        nm = placed.shape[1]
        cur = node[macro.clamp_min(0)]
        pm = placed.float()
        pmean = (pm @ node[:nm]) / pm.sum(1, keepdim=True).clamp_min(1.0) if nm else torch.zeros_like(cur)
        ctx = self.ctx(torch.cat([cur, gmean.expand(B, -1), pmean, scalars], dim=-1))
        s0 = F.relu(self.stem(image))
        x = F.relu(self.down(s0))
        for blk in self.blocks:
            x = blk(x, ctx)
        u = F.relu(self.up(x))
        logits = self.head(torch.cat([s0, u], 1)).flatten(1) - self.wm_scale * image[:, 5].flatten(1)
        logits = logits.masked_fill(~mask, MASK_VALUE)
        v = self.value(torch.cat([x.mean((2, 3)), x.amax((2, 3)), ctx], dim=-1)).squeeze(-1)
        return logits, v


def obs_to_tensors(obs_list: list[dict]) -> dict[str, torch.Tensor]:
    import numpy as np

    return {
        "image": torch.from_numpy(np.stack([o["image"] for o in obs_list])),
        "mask": torch.from_numpy(np.stack([o["mask"] for o in obs_list])),
        "macro": torch.tensor([o["macro"] for o in obs_list], dtype=torch.long),
        "placed": torch.from_numpy(np.stack([o["placed"] for o in obs_list])),
        "scalars": torch.from_numpy(np.stack([o["scalars"] for o in obs_list])),
    }
