"""Pure-Python DEF reader and macro-placement writer.

Reads the subset of DEF 5.8 that OpenROAD/ORFS emits at floorplan stage:
``UNITS``, ``DIEAREA``, ``ROW``, ``COMPONENTS`` (with placement status and
orientation), ``PINS`` (with layer shapes and placement) and ``NETS`` (pin
lists only; routing is ignored). All other sections are skipped.

The writer (:func:`update_def_components` / :func:`write_def_placement`)
rewrites only the placement status, location and orientation of selected
components and leaves every other byte of the file untouched, so a placement
can be injected back into a flow.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from ..core.design import Cell, Design, Macro, Net, PinRef, Placement, Port, Row, Site
from ..core.geometry import Orient, Rect, transform_offsets
from .lef import LefLibrary, read_lef

_DEF_TOKEN_RE = re.compile(r'"[^"]*"|[();]|[^\s();]+')


def _tokenize_def(text: str) -> list[str]:
    toks: list[str] = []
    for line in text.splitlines():
        s = line.lstrip()
        if s.startswith("#"):
            continue
        toks.extend(_DEF_TOKEN_RE.findall(line))
    return toks


@dataclass
class DefComponent:
    name: str
    master: str
    status: str = "UNPLACED"  # PLACED | FIXED | COVER | UNPLACED
    x: int = 0
    y: int = 0
    orient: int = Orient.N
    halo: tuple[int, int, int, int] | None = None


@dataclass
class DefPin:
    name: str
    net: str = ""
    direction: str = ""
    use: str = ""
    layer: str = ""
    rect: tuple[int, int, int, int] | None = None
    status: str = ""
    x: int = 0
    y: int = 0
    orient: int = Orient.N

    def centre(self) -> tuple[float, float]:
        """Absolute centre of the pin shape (or placement point)."""
        if self.rect is None:
            return float(self.x), float(self.y)
        cx = 0.5 * (self.rect[0] + self.rect[2])
        cy = 0.5 * (self.rect[1] + self.rect[3])
        dx, dy = transform_offsets(np.array(cx), np.array(cy), self.orient)
        return self.x + float(dx), self.y + float(dy)


@dataclass
class DefNet:
    name: str
    pins: list[tuple[str, str]] = field(default_factory=list)  # (component or 'PIN', pin)
    use: str = "SIGNAL"
    weight: float = 1.0


@dataclass
class DefData:
    design: str = ""
    dbu_per_micron: int = 1000
    die: Rect | None = None
    rows: list[Row] = field(default_factory=list)
    components: list[DefComponent] = field(default_factory=list)
    pins: list[DefPin] = field(default_factory=list)
    nets: list[DefNet] = field(default_factory=list)


_SKIP_SECTIONS = {
    "VIAS", "NONDEFAULTRULES", "REGIONS", "BLOCKAGES", "SPECIALNETS", "SCANCHAINS", "GROUPS",
    "FILLS", "SLOTS", "STYLES", "PINPROPERTIES", "PROPERTYDEFINITIONS",
}


class _S:
    def __init__(self, toks: list[str]):
        self.t, self.i, self.n = toks, 0, len(toks)

    def peek(self, k: int = 0) -> str | None:
        j = self.i + k
        return self.t[j] if j < self.n else None

    def next(self) -> str:
        tok = self.t[self.i]
        self.i += 1
        return tok

    def stmt(self) -> list[str]:
        """Tokens up to (not including) the next ';'."""
        j = self.i
        t = self.t
        while j < self.n and t[j] != ";":
            j += 1
        out = t[self.i : j]
        self.i = j + 1
        return out

    def skip_section(self, name: str) -> None:
        while self.i < self.n:
            if self.t[self.i] == "END" and self.peek(1) == name:
                self.i += 2
                return
            self.i += 1


def _parse_point(t: list[str], k: int) -> tuple[int, int, int]:
    """Parse ``( x y )`` starting at t[k] == '('; returns (x, y, next index)."""
    assert t[k] == "(", t[k : k + 4]
    return int(round(float(t[k + 1]))), int(round(float(t[k + 2]))), k + 4


def _parse_component(t: list[str]) -> DefComponent:
    c = DefComponent(t[1], t[2])
    k = 3
    while k < len(t):
        if t[k] != "+":
            k += 1
            continue
        key = t[k + 1] if k + 1 < len(t) else ""
        if key in ("PLACED", "FIXED", "COVER"):
            x, y, k2 = _parse_point(t, k + 2)
            c.status, c.x, c.y = key, x, y
            c.orient = int(Orient.parse(t[k2]))
            k = k2 + 1
        elif key == "UNPLACED":
            c.status = "UNPLACED"
            k += 2
        elif key == "HALO":
            j = k + 2
            if t[j] == "SOFT":
                j += 1
            c.halo = tuple(int(float(v)) for v in t[j : j + 4])  # type: ignore[assignment]
            k = j + 4
        else:
            k += 2
    return c


def _parse_pin(t: list[str]) -> DefPin:
    p = DefPin(t[1])
    k = 2
    got_place = False
    got_layer = False
    while k < len(t):
        if t[k] != "+":
            k += 1
            continue
        key = t[k + 1]
        if key == "NET":
            p.net = t[k + 2]
            k += 3
        elif key == "DIRECTION":
            p.direction = t[k + 2]
            k += 3
        elif key == "USE":
            p.use = t[k + 2]
            k += 3
        elif key == "LAYER" and not got_layer:
            got_layer = True
            p.layer = t[k + 2]
            j = k + 3
            while j < len(t) and t[j] != "(":
                j += 2 if t[j] in ("MASK", "SPACING", "DESIGNRULEWIDTH") else 1
            x0, y0, j = _parse_point(t, j)
            x1, y1, j = _parse_point(t, j)
            p.rect = (x0, y0, x1, y1)
            k = j
        elif key in ("PLACED", "FIXED", "COVER") and not got_place:
            got_place = True
            x, y, j = _parse_point(t, k + 2)
            p.status, p.x, p.y = key, x, y
            p.orient = int(Orient.parse(t[j]))
            k = j + 1
        else:
            k += 2
    return p


def _parse_net(t: list[str]) -> DefNet | None:
    name = t[1]
    if name == "MUSTJOIN":
        return None
    net = DefNet(name)
    k = 2
    n = len(t)
    while k < n and t[k] == "(":
        # ( comp pin [+ SYNTHESIZED] )
        j = k + 1
        while j < n and t[j] != ")":
            j += 1
        group = t[k + 1 : j]
        if len(group) >= 2 and group[0] != "*":
            net.pins.append((group[0], group[1]))
        k = j + 1
    while k < n:
        if t[k] == "+" and k + 1 < n:
            key = t[k + 1]
            if key == "USE" and k + 2 < n:
                net.use = t[k + 2]
            elif key == "WEIGHT" and k + 2 < n:
                net.weight = float(t[k + 2])
        k += 1
    return net


def parse_def(text: str) -> DefData:
    """Parse DEF text into a :class:`DefData` record (no LEF needed)."""
    d = DefData()
    s = _S(_tokenize_def(text))
    while s.peek() is not None:
        tok = s.next()
        if tok == ";":
            continue
        if tok == "DESIGN":
            d.design = s.stmt()[0]
        elif tok == "UNITS":
            a = s.stmt()  # DISTANCE MICRONS n
            d.dbu_per_micron = int(float(a[-1]))
        elif tok == "DIEAREA":
            a = s.stmt()
            pts = []
            k = 0
            while k < len(a):
                if a[k] == "(":
                    x, y, k = _parse_point(a, k)
                    pts.append((x, y))
                else:
                    k += 1
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            d.die = Rect(min(xs), min(ys), max(xs), max(ys))
        elif tok == "ROW":
            a = s.stmt()
            row = Row(a[0], a[1], int(float(a[2])), int(float(a[3])), 1, 1, 0, 0, int(Orient.parse(a[4])))
            if "DO" in a:
                k = a.index("DO")
                row.num_x, row.num_y = int(a[k + 1]), int(a[k + 3])
            if "STEP" in a:
                k = a.index("STEP")
                row.step_x, row.step_y = int(float(a[k + 1])), int(float(a[k + 2]))
            d.rows.append(row)
        elif tok in ("COMPONENTS", "PINS", "NETS"):
            s.stmt()  # count ;
            while True:
                if s.peek() == "END" and s.peek(1) == tok:
                    s.i += 2
                    break
                st = s.stmt()
                if not st or st[0] != "-":
                    continue
                if tok == "COMPONENTS":
                    d.components.append(_parse_component(st))
                elif tok == "PINS":
                    d.pins.append(_parse_pin(st))
                else:
                    net = _parse_net(st)
                    if net is not None:
                        d.nets.append(net)
        elif tok in _SKIP_SECTIONS:
            s.skip_section(tok)
        elif tok == "BEGINEXT":
            while s.peek() is not None and s.next() != "ENDEXT":
                pass
        elif tok == "END":
            if s.peek() == "DESIGN":
                break
        else:
            s.stmt()
    return d


def def_to_design(
    data: DefData,
    lef: LefLibrary | None = None,
    *,
    keep_cells: bool = True,
    skip_power_nets: bool = True,
    halo: int = 0,
) -> Design:
    """Build a :class:`Design` from parsed DEF + LEF.

    Components whose LEF master has ``CLASS BLOCK*`` become macros, all others
    become cells (``keep_cells=False`` drops them and their net pins).
    Components with an unknown master get zero size (counted in
    ``design.meta['unknown_masters']``).
    """
    lef = lef or LefLibrary()
    dbu = data.dbu_per_micron
    die = data.die or Rect(0, 0, 0, 0)
    if data.rows:
        xl = min(r.x for r in data.rows)
        yl = min(r.y for r in data.rows)
        site_name = data.rows[0].site
        ls = lef.sites.get(site_name)
        if ls is not None and ls.width > 0:
            sw, sh = int(round(ls.width * dbu)), int(round(ls.height * dbu))
        else:
            sw = max(1, data.rows[0].step_x or 1)
            ys = sorted({r.y for r in data.rows})
            sh = int(min(np.diff(ys))) if len(ys) > 1 else sw
        xh = max(r.x + (r.num_x * (r.step_x or sw) if r.num_y == 1 else sw) for r in data.rows)
        yh = max(r.y + (r.num_y * r.step_y if r.num_y > 1 and r.step_y else sh) for r in data.rows)
        core = Rect(xl, yl, xh, yh)
        site: Site | None = Site(site_name, sw, sh)
    else:
        core = die
        site = None

    macros: list[Macro] = []
    cells: list[Cell] = []
    unknown = 0
    names_kept: set[str] = set()
    for c in data.components:
        lm = lef.macros.get(c.master)
        if lm is None:
            unknown += 1
            w = h = 0
            pins: dict[str, tuple[float, float]] = {}
            block = False
        else:
            w, h = int(round(lm.width * dbu)), int(round(lm.height * dbu))
            pins = {k: (v[0] * dbu, v[1] * dbu) for k, v in lm.pin_offsets().items()}
            block = lm.is_block
        placed = c.status != "UNPLACED"
        fixed = c.status in ("FIXED", "COVER")
        if block:
            macros.append(Macro(c.name, w, h, c.x, c.y, c.orient, fixed, placed, c.master, pins))
            names_kept.add(c.name)
        elif keep_cells:
            cells.append(Cell(c.name, w, h, c.x, c.y, c.orient, fixed, placed, c.master, pins))
            names_kept.add(c.name)

    ports = []
    port_names = set()
    for p in data.pins:
        if skip_power_nets and p.use in ("POWER", "GROUND"):
            continue
        x, y = p.centre()
        ports.append(Port(p.name, x, y, p.direction or "INOUT", p.layer))
        port_names.add(p.name)

    nets = []
    for n in data.nets:
        if skip_power_nets and n.use in ("POWER", "GROUND"):
            continue
        refs = []
        for comp, pin in n.pins:
            if comp == "PIN":
                if pin in port_names:
                    refs.append(PinRef(pin, "", True))
            elif comp in names_kept:
                refs.append(PinRef(comp, pin))
        nets.append(Net(n.name, refs, n.weight))

    return Design(
        name=data.design,
        dbu_per_micron=dbu,
        die=die,
        core=core,
        site=site,
        rows=list(data.rows),
        macros=macros,
        cells=cells,
        ports=ports,
        nets=nets,
        halo=halo,
        meta={"source": "def", "unknown_masters": unknown},
    )


def read_def(
    def_path: str | Path,
    lef: LefLibrary | str | Path | Iterable[str | Path] | None = None,
    **kwargs,
) -> Design:
    """Read a DEF (plus LEF files or a parsed :class:`LefLibrary`) into a :class:`Design`."""
    if lef is not None and not isinstance(lef, LefLibrary):
        lef = read_lef(lef)
    return def_to_design(parse_def(Path(def_path).read_text()), lef, **kwargs)


# --------------------------------------------------------------------- writer
_COMP_HDR = re.compile(r"^[ \t]*COMPONENTS[ \t]+\d+[ \t]*;", re.M)
_COMP_END = re.compile(r"^[ \t]*END[ \t]+COMPONENTS\b", re.M)
_COMP_STMT = re.compile(r"-\s+(\S+)\s+(\S+)([^;]*);")
_PLACE_RE = re.compile(r"\+\s*(PLACED|FIXED|COVER)\s*\(\s*(-?[\d.]+)\s+(-?[\d.]+)\s*\)\s*(\w+)|\+\s*UNPLACED\b")


def update_def_components(text: str, placements: Mapping[str, tuple[int, int, int]], status: str | None = None) -> str:
    """Rewrite placement of the named components in DEF ``text``.

    ``placements`` maps component name -> ``(x, y, orient)``. ``status`` forces
    ``PLACED``/``FIXED``; ``None`` keeps an existing ``PLACED``/``FIXED``/``COVER``
    status and uses ``PLACED`` for previously unplaced components. Raises
    ``KeyError`` if a name is not found.
    """
    h = _COMP_HDR.search(text)
    if h is None:
        raise ValueError("DEF has no COMPONENTS section")
    e = _COMP_END.search(text, h.end())
    if e is None:
        raise ValueError("unterminated COMPONENTS section")
    section = text[h.end() : e.start()]
    seen: set[str] = set()

    def repl(m: re.Match) -> str:
        name = m.group(1)
        if name not in placements:
            return m.group(0)
        seen.add(name)
        x, y, o = placements[name]
        body = m.group(3)
        pm = _PLACE_RE.search(body)
        st = status
        if st is None:
            st = pm.group(1) if (pm is not None and pm.group(1)) else "PLACED"
        new = f"+ {st} ( {int(x)} {int(y)} ) {Orient(int(o)).name}"
        if pm is not None:
            body = body[: pm.start()] + new + body[pm.end() :]
        else:
            body = body.rstrip() + " " + new + " "
        head = m.group(0)[: m.start(3) - m.start(0)]
        return f"{head}{body};"

    new_section = _COMP_STMT.sub(repl, section)
    missing = set(placements) - seen
    if missing:
        raise KeyError(f"components not found in DEF: {sorted(missing)[:5]}")
    return text[: h.end()] + new_section + text[e.start() :]


def write_def_placement(
    src: str | Path,
    out: str | Path,
    design: Design,
    placement: Placement | None = None,
    *,
    status: str | None = None,
) -> None:
    """Copy DEF ``src`` to ``out`` with the macros' placement from ``placement``
    (default: the design's current macro positions). Fixed macros are left alone."""
    p = placement if placement is not None else design.placement()
    mapping = {
        m.name: (int(p.x[i]), int(p.y[i]), int(p.orient[i])) for i, m in enumerate(design.macros) if not m.fixed
    }
    text = Path(src).read_text()
    Path(out).write_text(update_def_components(text, mapping, status))
