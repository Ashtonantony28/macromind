"""Bin utilisation and density-overflow cost."""

from __future__ import annotations

import numpy as np

from ..core.design import KIND_PORT, DesignArrays
from ..core.geometry import oriented_dims
from .grid import GridSpec, splat_rects


def node_rects(arr: DesignArrays, cx: np.ndarray, cy: np.ndarray, orient: np.ndarray, nodes: np.ndarray | None = None):
    """Oriented rectangles ``(xl, yl, xh, yh)`` of ``nodes`` (default: all)."""
    if nodes is None:
        nodes = np.arange(arr.n_nodes)
    w, h = oriented_dims(arr.node_w[nodes], arr.node_h[nodes], orient[nodes])
    x, y = cx[nodes], cy[nodes]
    return x - 0.5 * w, y - 0.5 * h, x + 0.5 * w, y + 0.5 * h


def area_nodes(arr: DesignArrays) -> np.ndarray:
    """Indices of nodes with area (macros and cells)."""
    return np.flatnonzero((arr.node_kind != KIND_PORT) & (arr.node_w * arr.node_h > 0))


def density_map(arr: DesignArrays, cx, cy, orient, grid: GridSpec) -> np.ndarray:
    """Bin utilisation (occupied area / bin area), shape ``(nx, ny)``."""
    nodes = area_nodes(arr)
    xl, yl, xh, yh = node_rects(arr, cx, cy, orient, nodes)
    return splat_rects(xl, yl, xh, yh, np.ones(len(nodes)), grid) / grid.bin_area


def density_overflow(util: np.ndarray, target: float, bin_area: float, total_area: float) -> float:
    """Area above ``target`` utilisation, normalised by the total node area."""
    over = np.maximum(util - target, 0.0).sum() * bin_area
    return float(over / max(total_area, 1e-12))


def topk_mean(values: np.ndarray, frac: float) -> float:
    """Mean of the top ``frac`` fraction of ``values`` (at least one element)."""
    v = np.asarray(values).ravel()
    k = max(1, int(np.ceil(frac * v.size)))
    if k >= v.size:
        return float(v.mean())
    return float(np.partition(v, v.size - k)[v.size - k :].mean())
