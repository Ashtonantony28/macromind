"""Minimal pure-Python LEF reader.

Extracts what macro placement needs: database units, ``SITE`` definitions and
``MACRO`` class, size, origin and pin centres (centre of the bounding box of
all of a pin's port shapes). Everything else (layers, vias, rules, properties)
is skipped. The reader is robust to the subset of LEF that ORFS platforms
(NanGate45, ASAP7, sky130) and fakeram generators emit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

_TOKEN_RE = re.compile(r'"[^"]*"|;|[^\s;]+')


def tokenize(text: str) -> list[str]:
    """Split LEF/DEF text into tokens, dropping ``#`` comments (outside quotes)."""
    out: list[str] = []
    for line in text.splitlines():
        if "#" in line:
            # strip comments that are not inside a quoted string
            q = False
            for k, ch in enumerate(line):
                if ch == '"':
                    q = not q
                elif ch == "#" and not q:
                    line = line[:k]
                    break
        out.extend(_TOKEN_RE.findall(line))
    return out


@dataclass
class LefSite:
    name: str
    width: float  # microns
    height: float
    site_class: str = ""


@dataclass
class LefPin:
    name: str
    direction: str = ""
    use: str = ""
    #: (xl, yl, xh, yh) in microns, macro-origin relative (ORIGIN applied)
    rects: list[tuple[float, float, float, float]] = field(default_factory=list)

    def centre(self) -> tuple[float, float] | None:
        if not self.rects:
            return None
        xl = min(r[0] for r in self.rects)
        yl = min(r[1] for r in self.rects)
        xh = max(r[2] for r in self.rects)
        yh = max(r[3] for r in self.rects)
        return 0.5 * (xl + xh), 0.5 * (yl + yh)


@dataclass
class LefMacro:
    name: str
    macro_class: str = ""
    width: float = 0.0  # microns
    height: float = 0.0
    origin: tuple[float, float] = (0.0, 0.0)
    site: str = ""
    pins: dict[str, LefPin] = field(default_factory=dict)

    @property
    def is_block(self) -> bool:
        return self.macro_class.upper().startswith("BLOCK")

    def pin_offsets(self) -> dict[str, tuple[float, float]]:
        """Pin centre offsets from the lower-left corner, in microns."""
        out = {}
        for n, p in self.pins.items():
            c = p.centre()
            if c is not None:
                out[n] = c
        return out


@dataclass
class LefLibrary:
    dbu_per_micron: int | None = None
    sites: dict[str, LefSite] = field(default_factory=dict)
    macros: dict[str, LefMacro] = field(default_factory=dict)

    def merge(self, other: "LefLibrary") -> "LefLibrary":
        if self.dbu_per_micron is None:
            self.dbu_per_micron = other.dbu_per_micron
        self.sites.update(other.sites)
        self.macros.update(other.macros)
        return self


class _Stream:
    def __init__(self, toks: list[str]):
        self.t = toks
        self.i = 0

    def peek(self) -> str | None:
        return self.t[self.i] if self.i < len(self.t) else None

    def next(self) -> str:
        if self.i >= len(self.t):
            raise ValueError("unexpected end of LEF")
        tok = self.t[self.i]
        self.i += 1
        return tok

    def until_semicolon(self) -> list[str]:
        out = []
        while True:
            tok = self.next()
            if tok == ";":
                return out
            out.append(tok)

    def skip_to_end(self, name: str) -> None:
        """Skip until ``END name``."""
        while True:
            tok = self.next()
            if tok == "END" and self.peek() == name:
                self.next()
                return


_NAMED_BLOCKS = {"LAYER", "VIA", "VIARULE", "NONDEFAULTRULE", "SITE", "MACRO"}
_KEYWORD_BLOCKS = {"UNITS", "PROPERTYDEFINITIONS", "SPACING", "NOISETABLE", "CORRECTIONTABLE", "IRDROP", "ARRAY"}


def _rect_from(nums: list[str], origin: tuple[float, float]) -> tuple[float, float, float, float] | None:
    vals = []
    for s in nums:
        try:
            vals.append(float(s))
        except ValueError:
            continue  # e.g. MASK n
    if len(vals) < 4:
        return None
    ox, oy = origin
    if len(vals) > 4 and len(vals) % 2 == 0:  # MASK value already removed; may be polygon
        xs, ys = vals[0::2], vals[1::2]
    else:
        vals = vals[-4:]
        xs, ys = [vals[0], vals[2]], [vals[1], vals[3]]
    return min(xs) + ox, min(ys) + oy, max(xs) + ox, max(ys) + oy


def _parse_pin(s: _Stream, name: str, origin: tuple[float, float]) -> LefPin:
    pin = LefPin(name)
    while True:
        tok = s.next()
        if tok == "END":
            if s.peek() == name:
                s.next()
                return pin
            continue
        if tok == "DIRECTION":
            pin.direction = " ".join(s.until_semicolon())
        elif tok == "USE":
            pin.use = " ".join(s.until_semicolon())
        elif tok == "PORT":
            while True:
                t2 = s.next()
                if t2 == "END":
                    break
                if t2 in ("RECT", "POLYGON"):
                    args = s.until_semicolon()
                    if args and args[0] == "MASK":
                        args = args[2:]
                    if args and args[0] == "ITERATE":
                        args = args[1:]
                    r = _rect_from(args, origin)
                    if r is not None:
                        pin.rects.append(r)
                elif t2 == ";":
                    continue
                else:
                    s.until_semicolon()
        elif tok == ";":
            continue
        else:
            s.until_semicolon()


def _parse_macro(s: _Stream, name: str) -> LefMacro:
    m = LefMacro(name)
    while True:
        tok = s.next()
        if tok == "END":
            if s.peek() == name:
                s.next()
                break
            continue
        if tok == "CLASS":
            m.macro_class = " ".join(s.until_semicolon())
        elif tok == "SIZE":
            a = s.until_semicolon()  # w BY h
            m.width, m.height = float(a[0]), float(a[2])
        elif tok == "ORIGIN":
            a = s.until_semicolon()
            m.origin = (float(a[0]), float(a[1]))
        elif tok == "SITE":
            a = s.until_semicolon()
            m.site = a[0] if a else ""
        elif tok == "PIN":
            pname = s.next()
            m.pins[pname] = _parse_pin(s, pname, m.origin)
        elif tok == "OBS":
            while s.next() != "END":
                pass
        elif tok == ";":
            continue
        else:
            s.until_semicolon()
    return m


def parse_lef(text: str) -> LefLibrary:
    """Parse LEF text."""
    lib = LefLibrary()
    s = _Stream(tokenize(text))
    while s.peek() is not None:
        tok = s.next()
        if tok == "END":
            if s.peek() == "LIBRARY":
                s.next()
            continue
        if tok == "UNITS":
            while True:
                t2 = s.next()
                if t2 == "END" and s.peek() == "UNITS":
                    s.next()
                    break
                if t2 == "DATABASE":
                    a = s.until_semicolon()  # MICRONS n
                    lib.dbu_per_micron = int(float(a[-1]))
                elif t2 != ";":
                    s.until_semicolon()
        elif tok == "SITE":
            name = s.next()
            site = LefSite(name, 0.0, 0.0)
            while True:
                t2 = s.next()
                if t2 == "END" and s.peek() == name:
                    s.next()
                    break
                if t2 == "SIZE":
                    a = s.until_semicolon()
                    site.width, site.height = float(a[0]), float(a[2])
                elif t2 == "CLASS":
                    site.site_class = " ".join(s.until_semicolon())
                elif t2 != ";":
                    s.until_semicolon()
            lib.sites[name] = site
        elif tok == "MACRO":
            name = s.next()
            lib.macros[name] = _parse_macro(s, name)
        elif tok in _NAMED_BLOCKS:
            name = s.next()
            if tok == "VIA" and s.peek() in (";",):
                s.next()
                continue
            # 'LAYER name' at top level is a block; skip to 'END name'
            s.skip_to_end(name)
        elif tok == "BEGINEXT":
            while s.next() != "ENDEXT":
                pass
        elif tok in _KEYWORD_BLOCKS:
            s.skip_to_end(tok)
        elif tok == ";":
            continue
        else:
            s.until_semicolon()
    return lib


def read_lef(paths: str | Path | Iterable[str | Path]) -> LefLibrary:
    """Read one or several LEF files (later files override earlier definitions)."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    lib = LefLibrary()
    for p in paths:
        lib.merge(parse_lef(Path(p).read_text()))
    return lib


def iter_block_macros(lib: LefLibrary) -> Iterator[LefMacro]:
    return (m for m in lib.macros.values() if m.is_block)
