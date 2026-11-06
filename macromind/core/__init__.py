"""Core data model: designs, placements, geometry and the synthetic generator."""

from .design import (
    KIND_CELL,
    KIND_MACRO,
    KIND_PORT,
    Cell,
    Design,
    DesignArrays,
    Macro,
    Net,
    PinRef,
    Placement,
    Port,
    Row,
    Site,
)
from .geometry import FLIP_ORIENTS, Orient, Rect, oriented_dims, transform_offsets
from .synthetic import generate_synthetic

__all__ = [
    "KIND_CELL", "KIND_MACRO", "KIND_PORT", "Cell", "Design", "DesignArrays", "Macro", "Net",
    "PinRef", "Placement", "Port", "Row", "Site", "FLIP_ORIENTS", "Orient", "Rect",
    "oriented_dims", "transform_offsets", "generate_synthetic",
]
