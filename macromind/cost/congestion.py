"""RUDY (Rectangular Uniform wire DensitY) congestion estimation.

Each net spreads its horizontal wire length ``w * bw`` uniformly over its
bounding box (``bw x bh``), so per unit area the horizontal demand is
``w / bh`` and the vertical demand ``w / bw``. Degenerate boxes are expanded
symmetrically to a minimum extent (half a bin by default) so that two-pin nets
on a line do not produce infinite density. Demand is compared with routing
capacity (``routes_per_micron`` tracks per micron in each direction).
"""

from __future__ import annotations

import numpy as np

from .density import topk_mean
from .grid import GridSpec, axis_overlap


def expand_bboxes(x0, x1, y0, y1, min_w: float, min_h: float):
    """Expand boxes narrower than ``min_w``/``min_h`` about their centre."""
    bw = x1 - x0
    bh = y1 - y0
    px = np.maximum(min_w - bw, 0.0) * 0.5
    py = np.maximum(min_h - bh, 0.0) * 0.5
    return x0 - px, x1 + px, y0 - py, y1 + py


def rudy_maps(x0, x1, y0, y1, weight, grid: GridSpec, min_extent: tuple[float, float] | None = None):
    """Horizontal and vertical RUDY demand maps (wire length per bin), each ``(nx, ny)``."""
    if min_extent is None:
        min_extent = (0.5 * grid.bin_w, 0.5 * grid.bin_h)
    x0, x1, y0, y1 = expand_bboxes(x0, x1, y0, y1, *min_extent)
    ox = axis_overlap(x0, x1, grid.x_edges)
    oy = axis_overlap(y0, y1, grid.y_edges)
    w = np.asarray(weight, dtype=np.float64)
    h_map = (ox * (w / (y1 - y0))[:, None]).T @ oy
    v_map = (ox * (w / (x1 - x0))[:, None]).T @ oy
    return h_map, v_map


def capacity_per_bin(grid: GridSpec, hroutes_per_dbu: float, vroutes_per_dbu: float) -> tuple[float, float]:
    """Routing capacity (wire length) of one bin in each direction."""
    return grid.bin_area * hroutes_per_dbu, grid.bin_area * vroutes_per_dbu


def congestion_score(h_map, v_map, h_cap: float, v_cap: float, topk: float = 0.05) -> float:
    """Mean of the top ``topk`` fraction of all H and V demand/capacity ratios."""
    return topk_mean(np.concatenate([(h_map / h_cap).ravel(), (v_map / v_cap).ravel()]), topk)
