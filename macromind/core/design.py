"""Design data model.

A :class:`Design` is the in-memory netlist + floorplan that every other module
works on. It is deliberately simple (plain dataclasses) and can be converted to
flat numpy arrays (:class:`DesignArrays`) for vectorised cost evaluation.

Conventions
-----------
* All geometry is in DBU (``dbu_per_micron`` DBU = 1 um).
* ``(x, y)`` of a macro/cell is the lower-left corner of its *oriented*
  bounding box (the DEF placement point).
* Pin offsets on a master are stored relative to the lower-left corner in the
  ``N`` (R0) orientation, as in LEF.
* Node indices in :class:`DesignArrays` are ``[macros | cells | ports]``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import NamedTuple, Sequence

import numpy as np

from .geometry import Orient, Rect, oriented_dims

KIND_MACRO = 0
KIND_CELL = 1
KIND_PORT = 2


@dataclass
class Site:
    """Placement site (LEF ``SITE``)."""

    name: str
    width: int
    height: int


@dataclass
class Row:
    """Placement row (DEF ``ROW``)."""

    name: str
    site: str
    x: int
    y: int
    num_x: int
    num_y: int = 1
    step_x: int = 0
    step_y: int = 0
    orient: int = Orient.N


@dataclass
class Macro:
    """A hard macro (or, with ``Cell``, any rectangular placeable instance)."""

    name: str
    width: int
    height: int
    x: int = 0
    y: int = 0
    orient: int = Orient.N
    fixed: bool = False
    placed: bool = True
    master: str = ""
    #: pin name -> (dx, dy) offset from lower-left corner in orientation N.
    pins: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def area(self) -> float:
        return float(self.width) * float(self.height)

    def oriented_size(self) -> tuple[int, int]:
        return (self.height, self.width) if Orient(self.orient).swaps_dims else (self.width, self.height)

    def rect(self) -> Rect:
        w, h = self.oriented_size()
        return Rect(self.x, self.y, self.x + w, self.y + h)


@dataclass
class Cell(Macro):
    """A standard cell, or a soft cluster of standard cells (``is_cluster``)."""

    is_cluster: bool = False


@dataclass
class Port:
    """A top-level I/O pin (DEF ``PINS``), treated as a fixed point."""

    name: str
    x: float
    y: float
    direction: str = "INOUT"
    layer: str = ""


class PinRef(NamedTuple):
    """Reference from a net to a pin: ``(instance name, pin name)``, or a port."""

    node: str
    pin: str = ""
    is_port: bool = False


@dataclass
class Net:
    name: str
    pins: list[PinRef] = field(default_factory=list)
    weight: float = 1.0


@dataclass
class Placement:
    """Macro placement: one entry per macro of the design, in design order."""

    x: np.ndarray  # int64, lower-left of oriented bbox (DBU)
    y: np.ndarray
    orient: np.ndarray  # int8 Orient codes
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.int64)
        self.y = np.asarray(self.y, dtype=np.int64)
        self.orient = np.asarray(self.orient, dtype=np.int8)
        if not (len(self.x) == len(self.y) == len(self.orient)):
            raise ValueError("placement arrays must have equal length")

    def __len__(self) -> int:
        return len(self.x)

    def copy(self) -> "Placement":
        return Placement(self.x.copy(), self.y.copy(), self.orient.copy(), dict(self.meta))


@dataclass
class Design:
    """Netlist + floorplan."""

    name: str
    dbu_per_micron: int
    die: Rect
    core: Rect
    site: Site | None = None
    rows: list[Row] = field(default_factory=list)
    macros: list[Macro] = field(default_factory=list)
    cells: list[Cell] = field(default_factory=list)
    ports: list[Port] = field(default_factory=list)
    nets: list[Net] = field(default_factory=list)
    #: Minimum spacing (DBU) required between macros.
    halo: int = 0
    #: Routing tracks per micron, used to normalise RUDY congestion.
    hroutes_per_micron: float = 10.0
    vroutes_per_micron: float = 10.0
    meta: dict = field(default_factory=dict)

    _arrays: "DesignArrays | None" = field(default=None, init=False, repr=False, compare=False)

    # ------------------------------------------------------------------ helpers
    @property
    def n_macros(self) -> int:
        return len(self.macros)

    @property
    def row_height(self) -> int:
        if self.site is not None:
            return int(self.site.height)
        return 1

    @property
    def site_width(self) -> int:
        if self.site is not None:
            return int(self.site.width)
        return 1

    def macro_index(self) -> dict[str, int]:
        return {m.name: i for i, m in enumerate(self.macros)}

    def placement(self) -> Placement:
        """Current macro placement."""
        return Placement(
            np.array([m.x for m in self.macros], dtype=np.int64),
            np.array([m.y for m in self.macros], dtype=np.int64),
            np.array([m.orient for m in self.macros], dtype=np.int8),
        )

    def apply_placement(self, p: Placement, *, inplace: bool = False) -> "Design":
        """Return a design with macro positions taken from ``p`` (topology shared)."""
        if len(p) != len(self.macros):
            raise ValueError(f"placement has {len(p)} macros, design has {len(self.macros)}")
        d = self if inplace else copy.copy(self)
        if not inplace:
            d.macros = [copy.copy(m) for m in self.macros]
            d._arrays = self._arrays  # topology unchanged
        for m, x, y, o in zip(d.macros, p.x, p.y, p.orient):
            m.x, m.y, m.orient, m.placed = int(x), int(y), int(o), True
        return d

    def arrays(self) -> "DesignArrays":
        """Flat numpy view of the topology (cached; invalidate with :meth:`touch`)."""
        if self._arrays is None:
            self._arrays = DesignArrays.build(self)
        return self._arrays

    def touch(self) -> None:
        """Invalidate cached arrays after editing topology (nets, pins, sizes)."""
        self._arrays = None

    def summary(self) -> str:
        return (
            f"Design {self.name}: {len(self.macros)} macros, {len(self.cells)} cells, "
            f"{len(self.ports)} ports, {len(self.nets)} nets, core {self.core.width:.0f}x{self.core.height:.0f} DBU "
            f"(dbu/um={self.dbu_per_micron})"
        )


def _csr(groups: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (ptr, idx) so that idx[ptr[g]:ptr[g+1]] are the positions with groups == g."""
    order = np.argsort(groups, kind="stable")
    counts = np.bincount(groups, minlength=n_groups)
    ptr = np.zeros(n_groups + 1, dtype=np.int64)
    np.cumsum(counts, out=ptr[1:])
    return ptr, order.astype(np.int64)


@dataclass
class DesignArrays:
    """Flat, vectorisation-friendly view of a design's topology.

    Nodes are ``[macros | cells | ports]``. Pins are stored in net order, so
    ``net_ptr`` is a CSR index: pins of net ``k`` are ``net_ptr[k]:net_ptr[k+1]``.
    Only nets with at least 2 pins are kept (``net_orig`` maps back).
    """

    n_macros: int
    n_cells: int
    n_ports: int
    node_w: np.ndarray  # R0 width (float64), 0 for ports
    node_h: np.ndarray
    node_kind: np.ndarray  # int8 KIND_*
    node_fixed: np.ndarray  # bool
    # default positions from the design (centres, orient)
    node_cx: np.ndarray
    node_cy: np.ndarray
    node_orient: np.ndarray
    pin_node: np.ndarray  # int64
    pin_dx: np.ndarray  # centre-relative R0 offset
    pin_dy: np.ndarray
    pin_net: np.ndarray
    net_ptr: np.ndarray
    net_weight: np.ndarray
    net_orig: np.ndarray
    node_pin_ptr: np.ndarray
    node_pin_idx: np.ndarray
    node_net_ptr: np.ndarray
    node_net_idx: np.ndarray
    core: Rect
    node_names: list[str]

    @property
    def n_nodes(self) -> int:
        return len(self.node_w)

    @property
    def n_pins(self) -> int:
        return len(self.pin_node)

    @property
    def n_nets(self) -> int:
        return len(self.net_weight)

    @staticmethod
    def build(design: Design) -> "DesignArrays":
        nodes: list[Macro] = list(design.macros) + list(design.cells)
        nm, nc, npo = len(design.macros), len(design.cells), len(design.ports)
        n = nm + nc + npo
        names = [x.name for x in nodes] + [p.name for p in design.ports]
        inst_index = {x.name: i for i, x in enumerate(nodes)}
        port_index = {p.name: nm + nc + i for i, p in enumerate(design.ports)}

        w = np.zeros(n)
        h = np.zeros(n)
        cx = np.zeros(n)
        cy = np.zeros(n)
        orient = np.zeros(n, dtype=np.int8)
        fixed = np.zeros(n, dtype=bool)
        kind = np.full(n, KIND_PORT, dtype=np.int8)
        kind[:nm] = KIND_MACRO
        kind[nm : nm + nc] = KIND_CELL
        for i, x in enumerate(nodes):
            w[i], h[i] = x.width, x.height
            ow, oh = oriented_dims(x.width, x.height, x.orient)
            cx[i], cy[i] = x.x + 0.5 * float(ow), x.y + 0.5 * float(oh)
            orient[i] = x.orient
            fixed[i] = x.fixed
        for j, p in enumerate(design.ports):
            cx[nm + nc + j], cy[nm + nc + j] = p.x, p.y
            fixed[nm + nc + j] = True

        pin_node: list[int] = []
        pin_dx: list[float] = []
        pin_dy: list[float] = []
        pin_net: list[int] = []
        weights: list[float] = []
        orig: list[int] = []
        k = 0
        for ni, net in enumerate(design.nets):
            refs = []
            for r in net.pins:
                if r.is_port:
                    idx = port_index.get(r.node)
                    if idx is None:
                        continue
                    refs.append((idx, 0.0, 0.0))
                else:
                    idx = inst_index.get(r.node)
                    if idx is None:
                        continue
                    inst = nodes[idx]
                    off = inst.pins.get(r.pin)
                    if off is None:
                        refs.append((idx, 0.0, 0.0))
                    else:
                        refs.append((idx, off[0] - 0.5 * inst.width, off[1] - 0.5 * inst.height))
            if len(refs) < 2:
                continue
            for idx, dx, dy in refs:
                pin_node.append(idx)
                pin_dx.append(dx)
                pin_dy.append(dy)
                pin_net.append(k)
            weights.append(net.weight)
            orig.append(ni)
            k += 1

        pin_node_a = np.asarray(pin_node, dtype=np.int64)
        pin_net_a = np.asarray(pin_net, dtype=np.int64)
        net_ptr = np.zeros(k + 1, dtype=np.int64)
        np.cumsum(np.bincount(pin_net_a, minlength=k), out=net_ptr[1:])
        node_pin_ptr, node_pin_idx = _csr(pin_node_a, n)
        # node -> unique nets
        pairs = np.unique(np.stack([pin_node_a, pin_net_a], axis=1), axis=0) if len(pin_node_a) else np.zeros((0, 2), np.int64)
        node_net_ptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(pairs[:, 0], minlength=n), out=node_net_ptr[1:])
        node_net_idx = pairs[:, 1].astype(np.int64)

        return DesignArrays(
            n_macros=nm,
            n_cells=nc,
            n_ports=npo,
            node_w=w,
            node_h=h,
            node_kind=kind,
            node_fixed=fixed,
            node_cx=cx,
            node_cy=cy,
            node_orient=orient,
            pin_node=pin_node_a,
            pin_dx=np.asarray(pin_dx, dtype=np.float64),
            pin_dy=np.asarray(pin_dy, dtype=np.float64),
            pin_net=pin_net_a,
            net_ptr=net_ptr,
            net_weight=np.asarray(weights, dtype=np.float64),
            net_orig=np.asarray(orig, dtype=np.int64),
            node_pin_ptr=node_pin_ptr,
            node_pin_idx=node_pin_idx,
            node_net_ptr=node_net_ptr,
            node_net_idx=node_net_idx,
            core=design.core,
            node_names=names,
        )

    # ---------------------------------------------------------------- positions
    def node_state(self, placement: Placement | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Centres and orientations of all nodes, with macros taken from ``placement``."""
        cx, cy, orient = self.node_cx.copy(), self.node_cy.copy(), self.node_orient.copy()
        if placement is not None:
            nm = self.n_macros
            if len(placement) != nm:
                raise ValueError("placement size mismatch")
            ow, oh = oriented_dims(self.node_w[:nm], self.node_h[:nm], placement.orient)
            cx[:nm] = placement.x + 0.5 * ow
            cy[:nm] = placement.y + 0.5 * oh
            orient[:nm] = placement.orient
        return cx, cy, orient

    def macro_centres_to_placement(self, cx: np.ndarray, cy: np.ndarray, orient: np.ndarray) -> Placement:
        """Inverse of :meth:`node_state` for the macro block (rounds to DBU)."""
        nm = self.n_macros
        o = np.asarray(orient[:nm], dtype=np.int8)
        ow, oh = oriented_dims(self.node_w[:nm], self.node_h[:nm], o)
        return Placement(np.rint(cx[:nm] - 0.5 * ow), np.rint(cy[:nm] - 0.5 * oh), o)

    def nets_of_nodes(self, nodes: Sequence[int] | np.ndarray) -> np.ndarray:
        """Sorted unique nets touching any of ``nodes``."""
        nodes = np.asarray(nodes, dtype=np.int64)
        if len(nodes) == 1:
            i = int(nodes[0])
            return self.node_net_idx[self.node_net_ptr[i] : self.node_net_ptr[i + 1]]
        parts = [self.node_net_idx[self.node_net_ptr[i] : self.node_net_ptr[i + 1]] for i in nodes]
        return np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)

    def pins_of_nets(self, nets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Concatenated pin indices of ``nets`` and segment start offsets (for reduceat)."""
        starts = self.net_ptr[nets]
        lens = self.net_ptr[nets + 1] - starts
        seg = np.zeros(len(nets), dtype=np.int64)
        if len(nets) > 1:
            np.cumsum(lens[:-1], out=seg[1:])
        idx = np.repeat(starts - seg, lens) + np.arange(int(lens.sum()), dtype=np.int64)
        return idx, seg
