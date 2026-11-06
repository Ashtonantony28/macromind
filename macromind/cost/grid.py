"""Uniform bin grid over the core used by density and congestion maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.geometry import Rect


@dataclass(frozen=True)
class GridSpec:
    """``nx`` x ``ny`` bins covering ``[x0, x0+width] x [y0, y0+height]``."""

    nx: int
    ny: int
    x0: float
    y0: float
    width: float
    height: float

    @staticmethod
    def over(rect: Rect, nx: int = 32, ny: int | None = None) -> "GridSpec":
        """Grid over ``rect``. With ``ny=None`` bins are made roughly square."""
        if ny is None:
            ny = max(1, int(round(nx * rect.height / max(rect.width, 1e-12))))
        return GridSpec(int(nx), int(ny), float(rect.xl), float(rect.yl), float(rect.width), float(rect.height))

    @property
    def bin_w(self) -> float:
        return self.width / self.nx

    @property
    def bin_h(self) -> float:
        return self.height / self.ny

    @property
    def bin_area(self) -> float:
        return self.bin_w * self.bin_h

    @property
    def x_edges(self) -> np.ndarray:
        return self.x0 + self.bin_w * np.arange(self.nx + 1)

    @property
    def y_edges(self) -> np.ndarray:
        return self.y0 + self.bin_h * np.arange(self.ny + 1)


def axis_overlap(lo: np.ndarray, hi: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Overlap lengths of intervals ``[lo_k, hi_k]`` with bins defined by ``edges``.

    Returns an array of shape ``(len(lo), len(edges) - 1)``.
    """
    lo = np.asarray(lo, dtype=np.float64)[:, None]
    hi = np.asarray(hi, dtype=np.float64)[:, None]
    ov = np.minimum(hi, edges[None, 1:]) - np.maximum(lo, edges[None, :-1])
    np.maximum(ov, 0.0, out=ov)
    return ov


def splat_rects(xl, yl, xh, yh, weight, grid: GridSpec) -> np.ndarray:
    """Sum ``weight_k * overlap_area(rect_k, bin)`` over rects; shape ``(nx, ny)``."""
    ox = axis_overlap(xl, xh, grid.x_edges)
    oy = axis_overlap(yl, yh, grid.y_edges)
    return (ox * np.asarray(weight, dtype=np.float64)[:, None]).T @ oy
