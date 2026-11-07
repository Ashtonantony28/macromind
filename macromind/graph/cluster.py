"""Standard-cell clustering into soft macros ("clusters").

:func:`cluster_cells` reduces a flat design (macros + thousands of std cells)
to macros + a few hundred soft clusters, following the AlphaChip/TILOS idea of
placing macros against a coarse netlist. The algorithm is multilevel
heavy-edge matching (HEM, as in METIS/hMETIS coarsening) on the clique-expanded
cell graph, with two twists:

* **Hierarchy affinity.** Cell names are split on ``/`` and ``.`` into a
  logical hierarchy path. Edges between cells of the same hierarchy group (at
  the deepest level that still has at most ``n_clusters`` groups) are boosted
  by ``1 + hier_weight``, so clusters prefer to follow module boundaries.
* **Area balance.** A cluster may not exceed ``area_balance`` times the mean
  cluster area, and matches are rated ``w_uv / (a_u + a_v)`` so small clusters
  merge first.

Cells without any net (tap cells, fillers, end caps) are dropped from the
reduced design (``assignment == -1``); fixed std cells that *are* connected
are clustered like the others. Clusters get an initial position from the
area-weighted centroid of their placed members; clusters whose members are all
unplaced (DEF ``UNPLACED`` cells sitting at (0, 0)) are put at the core centre
and, with ``initial_place=True``, moved by the quadratic placer
(:func:`macromind.graph.quadratic.place_clusters`) using ports and placed
macros as anchors.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ..core.design import Cell, Design, Net, PinRef
from ..core.geometry import Orient

_HIER_SPLIT = re.compile(r"[/.]")


@dataclass
class Clustering:
    """Result of :func:`cluster_cells`."""

    design: Design  #: reduced design: same macros/ports, clusters as cells
    assignment: np.ndarray  #: cluster index of every original cell (-1 = dropped)
    cluster_area: np.ndarray  #: exact summed member area of each cluster (DBU^2)
    method: str = "hem"
    stats: dict = field(default_factory=dict)

    @property
    def n_clusters(self) -> int:
        return len(self.cluster_area)

    def members(self, k: int) -> np.ndarray:
        return np.flatnonzero(self.assignment == k)


def hierarchy_path(name: str) -> tuple[str, ...]:
    """Logical hierarchy of an instance name (the leaf name is dropped)."""
    parts = [p for p in _HIER_SPLIT.split(name.lstrip("\\")) if p]
    return tuple(parts[:-1])


def hierarchy_groups(names: list[str], max_groups: int) -> np.ndarray:
    """Group id per name at the deepest hierarchy level with at most ``max_groups`` groups."""
    paths = [hierarchy_path(n) for n in names]
    depth = max((len(p) for p in paths), default=0)
    best = np.zeros(len(names), dtype=np.int64)
    for d in range(1, depth + 1):
        keys: dict[tuple[str, ...], int] = {}
        gid = np.fromiter((keys.setdefault(p[:d], len(keys)) for p in paths), dtype=np.int64, count=len(paths))
        if len(keys) > max_groups:
            break
        best = gid
    return best


def cell_adjacency(design: Design, cell_index: dict[str, int], max_net_degree: int = 64) -> sp.csr_matrix:
    """Symmetric clique-expanded cell-to-cell adjacency (weights ``w / (k - 1)``)."""
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    n = len(cell_index)
    for net in design.nets:
        idx = {cell_index[r.node] for r in net.pins if not r.is_port and r.node in cell_index}
        k = len(idx)
        if k < 2 or k > max_net_degree:
            continue
        a = np.fromiter(idx, dtype=np.int64, count=k)
        iu, ju = np.triu_indices(k, 1)
        rows.append(a[iu])
        cols.append(a[ju])
        vals.append(np.full(len(iu), net.weight / (k - 1)))
    if not rows:
        return sp.csr_matrix((n, n))
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    v = np.concatenate(vals)
    A = sp.coo_matrix((np.concatenate([v, v]), (np.concatenate([r, c]), np.concatenate([c, r]))), shape=(n, n)).tocsr()
    A.sum_duplicates()
    return A


def _hem_level(A: sp.csr_matrix, area: np.ndarray, max_area: float, need: int, rng: np.random.Generator, gid: np.ndarray | None = None) -> np.ndarray:
    """One level of heavy-edge matching; returns ``match`` (partner index or self)."""
    n = A.shape[0]
    match = np.full(n, -1, dtype=np.int64)
    indptr, indices, data = A.indptr, A.indices, A.data
    merged = 0
    for u in rng.permutation(n):
        if merged >= need:
            break
        if match[u] >= 0:
            continue
        s, e = indptr[u], indptr[u + 1]
        if s == e:
            continue
        nb = indices[s:e]
        ok = (match[nb] < 0) & (nb != u) & (area[nb] + area[u] <= max_area)
        if gid is not None:
            ok &= gid[nb] == gid[u]
        if not ok.any():
            continue
        nb = nb[ok]
        rating = data[s:e][ok] / (area[nb] + area[u])
        v = int(nb[int(np.argmax(rating))])
        match[u], match[v] = v, u
        merged += 1
    un = match < 0
    match[un] = np.flatnonzero(un)
    return match


def _contract(match: np.ndarray) -> np.ndarray:
    """Map each node to its coarse node index given a matching."""
    rep = np.minimum(np.arange(len(match)), match)
    _, inv = np.unique(rep, return_inverse=True)
    return inv


def cluster_cells(
    design: Design,
    n_clusters: int | None = None,
    *,
    max_net_degree: int = 64,
    hier_weight: float = 1.0,
    area_balance: float = 2.5,
    seed: int = 0,
    initial_place: bool = True,
    merge_parallel_nets: bool = True,
) -> Clustering:
    """Cluster ``design.cells`` into about ``n_clusters`` soft clusters (see module doc).

    ``n_clusters`` defaults to ``clip(n_cells // 40, 16, 1000)``. If the design
    has no more connected cells than ``n_clusters`` every cell becomes its own
    cluster (dimensions kept).
    """
    rng = np.random.default_rng(seed)
    cells = design.cells
    nc = len(cells)
    cell_index = {c.name: i for i, c in enumerate(cells)}

    connected = np.zeros(nc, dtype=bool)
    for net in design.nets:
        if len(net.pins) < 2:
            continue
        for r in net.pins:
            if not r.is_port:
                i = cell_index.get(r.node)
                if i is not None:
                    connected[i] = True
    keep = np.flatnonzero(connected)
    if n_clusters is None:
        n_clusters = int(np.clip(len(keep) // 40, 16, 1000))
    target = max(1, min(int(n_clusters), len(keep)))

    area0 = np.array([float(c.width) * float(c.height) for c in cells])
    sub_index = {cells[i].name: k for k, i in enumerate(keep)}
    A = cell_adjacency(design, sub_index, max_net_degree)
    area = np.maximum(area0[keep], 1.0)
    gid = hierarchy_groups([cells[i].name for i in keep], target)
    if hier_weight and len(keep) and gid.max() > 0:
        A = A.tocoo()
        same = gid[A.row] == gid[A.col]
        A = sp.coo_matrix((A.data * np.where(same, 1.0 + hier_weight, 1.0), (A.row, A.col)), shape=A.shape).tocsr()

    # ---- multilevel coarsening: phase 0 within hierarchy groups, phase 1 global, then fallback
    to_coarse = np.arange(len(keep))
    levels = 0
    relax = 1.0
    phase = 0 if (hier_weight and len(keep) and gid.max() > 0) else 1
    max_area = area_balance * area.sum() / target
    while A.shape[0] > target:
        n = A.shape[0]
        match = _hem_level(A, area, max_area * relax, n - target, rng, gid if phase == 0 else None)
        merged = int((match != np.arange(n)).sum() // 2)
        if merged < max(1, 0.02 * (n - target)):
            if relax < 4:
                relax *= 2  # stall: let clusters grow
            elif phase == 0:
                phase, relax = 1, 1.0
            else:
                match = _fallback_pairs(A, area, gid, n - target)
                merged = 1
            if merged == 0:
                continue
        inv = _contract(match)
        m = int(inv.max()) + 1
        P = sp.csr_matrix((np.ones(n), (np.arange(n), inv)), shape=(n, m))
        A = (P.T @ A @ P).tocsr()
        A.setdiag(0)
        A.eliminate_zeros()
        area = P.T @ area
        g2 = np.zeros(m, dtype=np.int64)
        g2[inv] = gid  # any member's group
        gid = g2
        to_coarse = inv[to_coarse]
        levels += 1

    assignment = np.full(nc, -1, dtype=np.int64)
    assignment[keep] = to_coarse
    k = int(to_coarse.max()) + 1 if len(keep) else 0
    reduced, cl_area = _reduced_design(design, assignment, k, merge_parallel_nets)
    if initial_place and k:
        from .quadratic import place_clusters

        unplaced = reduced.meta["clustering"]["unplaced_clusters"]
        if unplaced:
            reduced = place_clusters(reduced, movable=np.asarray(unplaced, dtype=np.int64))
    stats = {
        "n_cells": nc, "n_connected": int(len(keep)), "n_clusters": k, "levels": levels,
        "dropped": int(nc - len(keep)), "n_nets": len(reduced.nets), "hier_groups": int(gid.max() + 1) if len(gid) else 0,
    }
    reduced.meta["clustering"].update(stats)
    return Clustering(reduced, assignment, cl_area, "hem", stats)


def _fallback_pairs(A: sp.csr_matrix, area: np.ndarray, gid: np.ndarray, need: int) -> np.ndarray:
    """Merge unconnected leftovers: pair the smallest clusters in (group, index) order."""
    n = A.shape[0]
    match = np.arange(n)
    order = np.lexsort((np.arange(n), gid))
    pairs = 0
    small = set(np.argsort(area)[: 2 * need].tolist())
    seq = [int(i) for i in order if int(i) in small]
    for a, b in zip(seq[::2], seq[1::2]):
        if pairs >= need:
            break
        match[a], match[b] = b, a
        pairs += 1
    return match


def _reduced_design(design: Design, assignment: np.ndarray, k: int, merge_parallel: bool) -> tuple[Design, np.ndarray]:
    cells = design.cells
    site_w, row_h = max(1, design.site_width), max(1, design.row_height)
    cl_area = np.zeros(k)
    sx = np.zeros(k)
    sy = np.zeros(k)
    sw = np.zeros(k)
    count = np.zeros(k, dtype=np.int64)
    fixed_all = np.ones(k, dtype=bool)
    for i, c in enumerate(cells):
        g = assignment[i]
        if g < 0:
            continue
        a = float(c.width) * float(c.height)
        cl_area[g] += a
        count[g] += 1
        fixed_all[g] &= bool(c.fixed)
        if c.placed:
            w, h = c.oriented_size()
            wt = max(a, 1.0)
            sx[g] += wt * (c.x + 0.5 * w)
            sy[g] += wt * (c.y + 0.5 * h)
            sw[g] += wt
    single = {int(assignment[i]): i for i in range(len(cells)) if assignment[i] >= 0}
    core = design.core
    clusters: list[Cell] = []
    unplaced: list[int] = []
    for g in range(k):
        if count[g] == 1:
            src = cells[single[g]]
            w, h = int(src.width), int(src.height)
        else:
            side = np.sqrt(max(cl_area[g], 1.0))
            w = max(site_w, int(round(side / site_w)) * site_w)
            h = max(row_h, int(round(cl_area[g] / w / row_h)) * row_h)
        if sw[g] > 0:
            cx, cy = sx[g] / sw[g], sy[g] / sw[g]
        else:
            cx, cy = core.cx, core.cy
            unplaced.append(g)
        x = int(np.clip(cx - w / 2, core.xl, max(core.xl, core.xh - w)))
        y = int(np.clip(cy - h / 2, core.yl, max(core.yl, core.yh - h)))
        name = cells[single[g]].name if count[g] == 1 else f"__cluster_{g}"
        clusters.append(Cell(name, w, h, x, y, Orient.N, bool(fixed_all[g] and count[g] == 1), True, "CLUSTER", {}, is_cluster=True))

    cname = [c.name for c in clusters]
    cell_index = {c.name: i for i, c in enumerate(cells)}
    merged: dict[tuple, int] = {}
    nets: list[Net] = []
    for net in design.nets:
        refs: list[PinRef] = []
        seen: set = set()
        for r in net.pins:
            if not r.is_port:
                i = cell_index.get(r.node)
                if i is not None:
                    g = assignment[i]
                    if g < 0:
                        continue
                    r = PinRef(cname[g], "", False)
            if r in seen:
                continue
            seen.add(r)
            refs.append(r)
        if len({(r.node, r.is_port) for r in refs}) < 2:
            continue
        if merge_parallel:
            key = tuple(sorted(refs))
            j = merged.get(key)
            if j is not None:
                nets[j].weight += net.weight
                continue
            merged[key] = len(nets)
        nets.append(Net(net.name, refs, float(net.weight)))

    reduced = Design(
        name=design.name, dbu_per_micron=design.dbu_per_micron, die=design.die, core=design.core, site=design.site,
        rows=list(design.rows), macros=[_copy(m) for m in design.macros], cells=clusters, ports=list(design.ports),
        nets=nets, halo=design.halo, hroutes_per_micron=design.hroutes_per_micron,
        vroutes_per_micron=design.vroutes_per_micron,
        meta={**design.meta, "clustering": {"unplaced_clusters": unplaced}},
    )
    return reduced, cl_area


def _copy(m):
    import copy

    return copy.copy(m)


def group_sizes(assignment: np.ndarray) -> dict[int, int]:
    d: dict[int, int] = defaultdict(int)
    for a in assignment:
        if a >= 0:
            d[int(a)] += 1
    return dict(d)
