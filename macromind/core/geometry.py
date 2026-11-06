"""Basic geometry: rectangles and LEF/DEF orientations.

All coordinates are in database units (DBU) unless stated otherwise.

Orientation codes follow OpenDB's ``dbOrientType`` ordering so that odd codes
are exactly the 90-degree rotated variants (which swap width and height)::

    code  DEF  OpenDB   transform of a centre-relative offset (x, y)
    0     N    R0       ( x,  y)
    1     W    R90      (-y,  x)
    2     S    R180     (-x, -y)
    3     E    R270     ( y, -x)
    4     FN   MY       (-x,  y)
    5     FE   MYR90    (-y, -x)
    6     FS   MX       ( x, -y)
    7     FW   MXR90    ( y,  x)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Orient(IntEnum):
    """Placement orientation (DEF names, OpenDB ordering)."""

    N = 0
    W = 1
    S = 2
    E = 3
    FN = 4
    FE = 5
    FS = 6
    FW = 7

    @classmethod
    def parse(cls, s: str | int | "Orient") -> "Orient":
        """Parse a DEF name (``N``, ``FS``...) or OpenDB name (``R0``, ``MX``...)."""
        if isinstance(s, (int, np.integer)):
            return cls(int(s))
        s = str(s).strip().upper()
        if s in _OPENDB_NAMES:
            return cls(_OPENDB_NAMES[s])
        return cls[s]

    @property
    def swaps_dims(self) -> bool:
        """True for 90/270-degree variants (width and height swap)."""
        return bool(self.value & 1)


_OPENDB_NAMES = {"R0": 0, "R90": 1, "R180": 2, "R270": 3, "MY": 4, "MYR90": 5, "MX": 6, "MXR90": 7}

#: The four orientations that keep a macro's footprint (no rotation).
FLIP_ORIENTS: tuple[int, ...] = (Orient.N, Orient.S, Orient.FN, Orient.FS)

#: ORIENT_MATRIX[o] = [[a, b], [c, d]] so that (x', y') = (a x + b y, c x + d y)
#: for a centre-relative offset (x, y).
ORIENT_MATRIX: np.ndarray = np.array(
    [
        [[1, 0], [0, 1]],  # N
        [[0, -1], [1, 0]],  # W  (R90)
        [[-1, 0], [0, -1]],  # S  (R180)
        [[0, 1], [-1, 0]],  # E  (R270)
        [[-1, 0], [0, 1]],  # FN (MY)
        [[0, -1], [-1, 0]],  # FE (MYR90)
        [[1, 0], [0, -1]],  # FS (MX)
        [[0, 1], [1, 0]],  # FW (MXR90)
    ],
    dtype=np.float64,
)
ORIENT_A = ORIENT_MATRIX[:, 0, 0].copy()
ORIENT_B = ORIENT_MATRIX[:, 0, 1].copy()
ORIENT_C = ORIENT_MATRIX[:, 1, 0].copy()
ORIENT_D = ORIENT_MATRIX[:, 1, 1].copy()


def transform_offsets(dx: np.ndarray, dy: np.ndarray, orient: np.ndarray | int) -> tuple[np.ndarray, np.ndarray]:
    """Apply orientation(s) to centre-relative offsets (vectorised)."""
    o = np.asarray(orient, dtype=np.intp)
    a, b, c, d = ORIENT_A[o], ORIENT_B[o], ORIENT_C[o], ORIENT_D[o]
    return a * dx + b * dy, c * dx + d * dy


def oriented_dims(w: np.ndarray | float, h: np.ndarray | float, orient: np.ndarray | int):
    """Width/height of the oriented bounding box."""
    o = np.asarray(orient)
    swap = (o & 1).astype(bool)
    return np.where(swap, h, w), np.where(swap, w, h)


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle ``[xl, xh] x [yl, yh]``."""

    xl: float
    yl: float
    xh: float
    yh: float

    @property
    def width(self) -> float:
        return self.xh - self.xl

    @property
    def height(self) -> float:
        return self.yh - self.yl

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return 0.5 * (self.xl + self.xh)

    @property
    def cy(self) -> float:
        return 0.5 * (self.yl + self.yh)

    def contains(self, other: "Rect") -> bool:
        return other.xl >= self.xl and other.yl >= self.yl and other.xh <= self.xh and other.yh <= self.yh

    def overlaps(self, other: "Rect") -> bool:
        """Strict overlap (touching edges do not overlap)."""
        return self.xl < other.xh and other.xl < self.xh and self.yl < other.yh and other.yl < self.yh

    def expanded(self, m: float) -> "Rect":
        return Rect(self.xl - m, self.yl - m, self.xh + m, self.yh + m)
