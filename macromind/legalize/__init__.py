"""Shared macro legaliser: grid snap, halo, overlap removal, orientation choice."""

from .legalizer import (
    LegalizationError,
    SnapGrid,
    check_legality,
    choose_orientations,
    is_legal,
    legalize,
    macro_bounds,
)

__all__ = ["LegalizationError", "SnapGrid", "check_legality", "choose_orientations", "is_legal", "legalize", "macro_bounds"]
