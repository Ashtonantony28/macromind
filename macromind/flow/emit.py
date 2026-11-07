"""Write and read macro placements in the formats OpenROAD/ORFS consume.

Two formats are supported:

* **TCL** (preferred): one ``place_macro -macro_name <inst> -location {x y}
  -orientation <R0|MX|...>`` line per macro, microns, lower-left corner of the
  oriented bounding box. This is exactly what ORFS's own
  ``write_macro_placement`` emits after RTLMP (``2_2_floorplan_macro.tcl``), so
  the same file can be replayed with ``MACRO_PLACEMENT_TCL``. ``place_macro``
  snaps to the manufacturing grid / pin-track alignment and sets the instance
  status to ``LOCKED`` (FIRM-equivalent: global placement will not move it).
* **DEF**: a copy of the input DEF with the macros' ``COMPONENTS`` entries
  rewritten as ``FIXED`` (see :func:`macromind.io.write_def_placement`).
"""

from __future__ import annotations

import re
from pathlib import Path

from ..core.design import Design, Placement
from ..core.geometry import Orient

#: DEF/MacroMind orientation code -> OpenDB name used by ``place_macro``.
OPENDB_ORIENT = {0: "R0", 1: "R90", 2: "R180", 3: "R270", 4: "MY", 5: "MYR90", 6: "MX", 7: "MXR90"}

_PLACE_MACRO_RE = re.compile(
    r"^\s*place_macro\s+-macro_name\s+(\{[^}]*\}|\S+)\s+-location\s+\{\s*(-?[\d.eE+]+)\s+(-?[\d.eE+]+)\s*\}"
    r"(?:\s+-orientation\s+(\S+))?"
)


def _fmt(v: float) -> str:
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def placement_to_tcl(design: Design, placement: Placement | None = None, *, include_fixed: bool = False) -> str:
    """``place_macro`` commands for every (movable) macro, in microns."""
    p = placement if placement is not None else design.placement()
    dbu = float(design.dbu_per_micron)
    lines = [f"# MacroMind macro placement for {design.name}: {len(design.macros)} macros"]
    for i, m in enumerate(design.macros):
        if m.fixed and not include_fixed:
            continue
        name = m.name
        if any(c in name for c in " {}[]$\\\""):
            name = "{" + name + "}"
        lines.append(
            f"place_macro -macro_name {name} -location {{{_fmt(p.x[i] / dbu)} {_fmt(p.y[i] / dbu)}}} "
            f"-orientation {OPENDB_ORIENT[int(p.orient[i])]}"
        )
    return "\n".join(lines) + "\n"


def write_tcl(path: str | Path, design: Design, placement: Placement | None = None) -> None:
    Path(path).write_text(placement_to_tcl(design, placement))


def parse_place_macro_tcl(text: str) -> dict[str, tuple[float, float, int]]:
    """Parse ``place_macro`` lines -> ``{inst: (x_um, y_um, orient_code)}``."""
    out: dict[str, tuple[float, float, int]] = {}
    for line in text.splitlines():
        m = _PLACE_MACRO_RE.match(line)
        if not m:
            continue
        name = m.group(1)
        if name.startswith("{"):
            name = name[1:-1]
        orient = int(Orient.parse(m.group(4))) if m.group(4) else 0
        out[name] = (float(m.group(2)), float(m.group(3)), orient)
    return out


def apply_um_placement(design: Design, placed: dict[str, tuple[float, float, int]], *, strict: bool = True) -> Placement:
    """Placement for ``design`` from ``{inst: (x_um, y_um, orient)}`` (others keep their position)."""
    p = design.placement()
    dbu = float(design.dbu_per_micron)
    idx = design.macro_index()
    missing = [n for n in placed if n not in idx]
    if missing and strict:
        raise KeyError(f"unknown macros: {missing[:5]}")
    for n, (x, y, o) in placed.items():
        i = idx.get(n)
        if i is None:
            continue
        p.x[i] = int(round(x * dbu))
        p.y[i] = int(round(y * dbu))
        p.orient[i] = o
    return p


def write_placement(path: str | Path, design: Design, placement: Placement, *, src_def: str | Path | None = None) -> str:
    """Write ``placement`` as TCL (``.tcl``) or DEF (``.def``, needs ``src_def``). Returns the format."""
    path = Path(path)
    if path.suffix.lower() == ".def":
        if src_def is None:
            raise ValueError("DEF output needs the source DEF")
        from ..io.defio import write_def_placement

        write_def_placement(src_def, path, design, placement, status="FIXED")
        return "def"
    write_tcl(path, design, placement)
    return "tcl"
