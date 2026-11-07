"""Design -> PyTorch graph tensors (plain torch, no PyG dependency).

Nodes are ``[macros | cells/clusters | ports]`` in :class:`DesignArrays` order,
followed (``net_model="star"``) by one node per net. Edges are undirected and
stored in both directions with weights:

* ``clique``: nets with ``k <= clique_max`` pins become cliques of weight
  ``w / (k - 1)``; larger nets connect each pin to the net's first pin (the
  driver) with weight ``w / (k - 1)``. Parallel edges are summed.
* ``star``: every pin connects to its net node with weight ``w``.

Node features (``FEATURES``): kind one-hot (macro, cell, port, net), width and
height normalised by the core, sqrt of area / core area, log1p pin count,
log1p net degree, fixed flag, and a known-position block (x, y normalised to
the core plus a flag) that is zero for movable macros, since their position
is what the placer decides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..core.design import KIND_MACRO, KIND_PORT, Design

FEATURES = ["is_macro", "is_cell", "is_port", "is_net", "w", "h", "sqrt_area", "log_pins", "log_degree", "fixed", "x", "y", "pos_known"]
N_FEATURES = len(FEATURES)


@dataclass
class GraphTensors:
    x: torch.Tensor  #: (N, F) float32 node features
    edge_index: torch.Tensor  #: (2, E) int64, both directions
    edge_weight: torch.Tensor  #: (E,) float32
    node_kind: torch.Tensor  #: (N,) int64 (0 macro, 1 cell, 2 port, 3 net)
    n_macros: int
    n_cells: int
    n_ports: int
    n_net_nodes: int = 0

    @property
    def n_nodes(self) -> int:
        return int(self.x.shape[0])

    def to(self, device) -> "GraphTensors":
        return GraphTensors(self.x.to(device), self.edge_index.to(device), self.edge_weight.to(device), self.node_kind.to(device),
                            self.n_macros, self.n_cells, self.n_ports, self.n_net_nodes)


def build_graph(design: Design, net_model: str = "clique", clique_max: int = 16) -> GraphTensors:
    arr = design.arrays()
    n = arr.n_nodes
    core = design.core
    deg = np.diff(arr.net_ptr)
    src: list[np.ndarray] = []
    dst: list[np.ndarray] = []
    wts: list[np.ndarray] = []
    n_net = 0
    if net_model == "clique":
        for k in np.unique(deg):
            nets = np.flatnonzero(deg == k)
            nodes = arr.pin_node[arr.net_ptr[nets][:, None] + np.arange(k)[None, :]]
            w = arr.net_weight[nets] / (k - 1)
            if k <= clique_max:
                iu, ju = np.triu_indices(k, 1)
                src.append(nodes[:, iu].ravel())
                dst.append(nodes[:, ju].ravel())
                wts.append(np.repeat(w, len(iu)))
            else:
                src.append(np.repeat(nodes[:, 0], k - 1))
                dst.append(nodes[:, 1:].ravel())
                wts.append(np.repeat(w, k - 1))
    elif net_model == "star":
        n_net = arr.n_nets
        src.append(arr.pin_node)
        dst.append(n + arr.pin_net)
        wts.append(arr.net_weight[arr.pin_net])
    else:
        raise ValueError(f"unknown net_model {net_model!r}")
    N = n + n_net
    if src:
        s, d, w = np.concatenate(src), np.concatenate(dst), np.concatenate(wts)
        keep = s != d
        s, d, w = s[keep], d[keep], w[keep]
        a, b = np.concatenate([s, d]), np.concatenate([d, s])
        key = a * N + b
        uk, inv = np.unique(key, return_inverse=True)
        ww = np.bincount(inv, weights=np.concatenate([w, w]))
        ei = np.stack([uk // N, uk % N])
    else:
        ei, ww = np.zeros((2, 0), np.int64), np.zeros(0)

    kind = np.concatenate([arr.node_kind.astype(np.int64), np.full(n_net, 3)])
    X = np.zeros((N, N_FEATURES), dtype=np.float32)
    X[np.arange(N), kind] = 1.0
    X[:n, 4] = arr.node_w / core.width
    X[:n, 5] = arr.node_h / core.height
    X[:n, 6] = np.sqrt(arr.node_w * arr.node_h / core.area)
    X[:n, 7] = np.log1p(np.diff(arr.node_pin_ptr))
    X[:n, 8] = np.log1p(np.diff(arr.node_net_ptr))
    if n_net:
        X[n:, 7] = np.log1p(deg)
        X[n:, 8] = np.log1p(arr.net_weight)
    X[:n, 9] = arr.node_fixed
    known = (arr.node_kind != KIND_MACRO) | arr.node_fixed
    X[:n, 10] = np.where(known, (arr.node_cx - core.xl) / core.width, 0.0)
    X[:n, 11] = np.where(known, (arr.node_cy - core.yl) / core.height, 0.0)
    X[:n, 12] = known
    return GraphTensors(
        torch.from_numpy(X), torch.from_numpy(ei.astype(np.int64)), torch.from_numpy(ww.astype(np.float32)),
        torch.from_numpy(kind), arr.n_macros, arr.n_cells, arr.n_ports, n_net,
    )


__all__ = ["FEATURES", "N_FEATURES", "GraphTensors", "build_graph", "KIND_PORT"]
