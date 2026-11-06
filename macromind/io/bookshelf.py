"""Bookshelf reader (ICCAD04 IBM mixed-size: ``.aux .nodes .nets .wts .pl .scl``).

Classification:

* movable nodes taller than one row -> :class:`Macro`; other movable nodes -> :class:`Cell`;
* terminals inside the core with non-zero area -> fixed :class:`Macro`;
* other terminals (pads on/outside the boundary, zero-size) -> :class:`Port`.

Bookshelf pin offsets are relative to the node centre; they are converted to
the lower-left-relative convention of :class:`Design`. Coordinates are
multiplied by ``scale`` and rounded to integer DBU (``dbu_per_micron`` is set
to ``scale``, i.e. one Bookshelf unit = one "micron").
"""

from __future__ import annotations

from pathlib import Path

from ..core.design import Cell, Design, Macro, Net, PinRef, Port, Row, Site
from ..core.geometry import Orient, Rect


def _lines(path: Path):
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("UCLA"):
            continue
        yield line


def _kv(line: str) -> tuple[str, str] | None:
    if ":" in line:
        k, v = line.split(":", 1)
        return k.strip(), v.strip()
    return None


def _parse_nodes(path: Path):
    nodes = {}
    for line in _lines(path):
        if line.startswith(("NumNodes", "NumTerminals")):
            continue
        t = line.split()
        name, w, h = t[0], float(t[1]), float(t[2])
        term = len(t) > 3 and t[3].startswith("terminal")
        nodes[name] = (w, h, term)
    return nodes


def _parse_nets(path: Path):
    nets = []
    cur = None
    for line in _lines(path):
        if line.startswith(("NumNets", "NumPins")):
            continue
        if line.startswith("NetDegree"):
            kv = _kv(line)
            parts = kv[1].split() if kv else line.split()[2:]
            name = parts[1] if len(parts) > 1 else f"net{len(nets)}"
            cur = (name, [])
            nets.append(cur)
            continue
        t = line.replace(":", " : ").split()
        node = t[0]
        dx = dy = 0.0
        if ":" in t:
            k = t.index(":")
            if len(t) >= k + 3:
                dx, dy = float(t[k + 1]), float(t[k + 2])
        if cur is not None:
            cur[1].append((node, dx, dy))
    return nets


def _parse_pl(path: Path):
    pl = {}
    for line in _lines(path):
        t = line.replace(":", " : ").split()
        if len(t) < 3:
            continue
        orient = "N"
        if ":" in t:
            k = t.index(":")
            if k + 1 < len(t) and not t[k + 1].startswith("/"):
                orient = t[k + 1]
        fixed = any(x.startswith("/FIXED") for x in t)
        pl[t[0]] = (float(t[1]), float(t[2]), orient, fixed)
    return pl


def _parse_scl(path: Path):
    rows = []
    cur: dict | None = None
    for line in _lines(path):
        if line.startswith("NumRows"):
            continue
        if line.startswith("CoreRow"):
            cur = {}
            continue
        if line.startswith("End"):
            if cur is not None:
                rows.append(cur)
            cur = None
            continue
        if cur is None:
            continue
        # e.g. "SubrowOrigin : 0 NumSites : 100" has two pairs
        toks = line.replace(":", " : ").split()
        k = 0
        while k < len(toks):
            if k + 2 < len(toks) and toks[k + 1] == ":":
                cur[toks[k]] = toks[k + 2]
                k += 3
            else:
                k += 1
    return rows


def read_bookshelf(aux_path: str | Path, *, scale: int = 1, name: str | None = None) -> Design:
    """Read a Bookshelf benchmark given its ``.aux`` file."""
    aux = Path(aux_path)
    files: dict[str, Path] = {}
    for line in _lines(aux):
        kv = _kv(line)
        names = (kv[1] if kv else line).split()
        for f in names:
            files[Path(f).suffix.lower()] = aux.parent / f
    for req in (".nodes", ".nets", ".pl", ".scl"):
        if req not in files:
            raise ValueError(f"{aux}: missing {req} file")

    nodes = _parse_nodes(files[".nodes"])
    nets_raw = _parse_nets(files[".nets"])
    pl = _parse_pl(files[".pl"])
    rows_raw = _parse_scl(files[".scl"])
    weights: dict[str, float] = {}
    if ".wts" in files and files[".wts"].exists():
        for line in _lines(files[".wts"]):
            t = line.split()
            if len(t) >= 2:
                try:
                    weights[t[0]] = float(t[1])
                except ValueError:
                    pass

    s = float(scale)

    def q(v: float) -> int:
        return int(round(v * s))

    rows: list[Row] = []
    row_heights: list[int] = []
    for k, r in enumerate(rows_raw):
        y = q(float(r.get("Coordinate", 0)))
        h = q(float(r.get("Height", 1)))
        sw = q(float(r.get("Sitespacing", r.get("Sitewidth", 1))))
        x = q(float(r.get("SubrowOrigin", 0)))
        n = int(float(r.get("NumSites", 0)))
        rows.append(Row(f"row{k}", "bookshelf_site", x, y, n, 1, sw, 0, Orient.N))
        row_heights.append(h)
    if rows:
        row_h = row_heights[0]
        core = Rect(
            min(r.x for r in rows),
            min(r.y for r in rows),
            max(r.x + r.num_x * r.step_x for r in rows),
            max(r.y + rh for r, rh in zip(rows, row_heights)),
        )
        site = Site("bookshelf_site", rows[0].step_x, row_h)
    else:
        row_h = 1
        core = Rect(0, 0, 0, 0)
        site = None

    macros: list[Macro] = []
    cells: list[Cell] = []
    ports: list[Port] = []
    kind: dict[str, str] = {}
    objs: dict[str, Macro] = {}
    for nname, (w, h, term) in nodes.items():
        x, y, o, fixed_pl = pl.get(nname, (0.0, 0.0, "N", False))
        W, H, X, Y = q(w), q(h), q(x), q(y)
        orient = int(Orient.parse(o))
        cxn, cyn = X + W / 2, Y + H / 2
        inside = core.xl <= cxn <= core.xh and core.yl <= cyn <= core.yh
        if term and (W * H == 0 or not inside):
            ports.append(Port(nname, float(cxn), float(cyn)))
            kind[nname] = "port"
        elif term or fixed_pl:
            m = Macro(nname, W, H, X, Y, orient, True, True, nname, {})
            macros.append(m)
            objs[nname] = m
            kind[nname] = "macro"
        elif H > row_h:
            m = Macro(nname, W, H, X, Y, orient, False, nname in pl, nname, {})
            macros.append(m)
            objs[nname] = m
            kind[nname] = "macro"
        else:
            c = Cell(nname, W, H, X, Y, orient, False, nname in pl, nname, {})
            cells.append(c)
            objs[nname] = c
            kind[nname] = "cell"

    nets: list[Net] = []
    for netname, pins in nets_raw:
        refs: list[PinRef] = []
        for node, dx, dy in pins:
            kd = kind.get(node)
            if kd is None:
                continue
            if kd == "port":
                refs.append(PinRef(node, "", True))
            else:
                obj = objs[node]
                pname = f"p{len(obj.pins)}"
                obj.pins[pname] = (dx * s + obj.width / 2, dy * s + obj.height / 2)
                refs.append(PinRef(node, pname))
        nets.append(Net(netname, refs, weights.get(netname, 1.0)))

    xs = [core.xl, core.xh] + [p.x for p in ports]
    ys = [core.yl, core.yh] + [p.y for p in ports]
    die = Rect(min(xs), min(ys), max(xs), max(ys))
    return Design(
        name=name or aux.stem,
        dbu_per_micron=int(scale),
        die=die,
        core=core,
        site=site,
        rows=rows,
        macros=macros,
        cells=cells,
        ports=ports,
        nets=nets,
        meta={"source": "bookshelf"},
    )
