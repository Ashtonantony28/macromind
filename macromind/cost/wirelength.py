"""Vectorised half-perimeter wirelength (HPWL)."""

from __future__ import annotations

import numpy as np

from ..core.design import DesignArrays
from ..core.geometry import transform_offsets


def pin_positions(arr: DesignArrays, cx: np.ndarray, cy: np.ndarray, orient: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Absolute pin coordinates given node centres and orientations."""
    o = orient[arr.pin_node]
    dx, dy = transform_offsets(arr.pin_dx, arr.pin_dy, o)
    return cx[arr.pin_node] + dx, cy[arr.pin_node] + dy


def net_bboxes(arr: DesignArrays, px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-net bounding boxes ``(xmin, xmax, ymin, ymax)``."""
    if arr.n_nets == 0:
        z = np.zeros(0)
        return z, z, z, z
    s = arr.net_ptr[:-1]
    return (
        np.minimum.reduceat(px, s),
        np.maximum.reduceat(px, s),
        np.minimum.reduceat(py, s),
        np.maximum.reduceat(py, s),
    )


def net_hpwl(arr: DesignArrays, cx: np.ndarray, cy: np.ndarray, orient: np.ndarray) -> np.ndarray:
    """Unweighted HPWL of every (kept) net."""
    px, py = pin_positions(arr, cx, cy, orient)
    x0, x1, y0, y1 = net_bboxes(arr, px, py)
    return (x1 - x0) + (y1 - y0)


def hpwl(arr: DesignArrays, cx: np.ndarray, cy: np.ndarray, orient: np.ndarray) -> float:
    """Total weighted HPWL (DBU)."""
    return float(np.dot(arr.net_weight, net_hpwl(arr, cx, cy, orient)))
