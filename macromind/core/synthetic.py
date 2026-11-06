"""Seeded synthetic mixed-size designs.

Used for RL training and CI when real designs are unavailable. The generator
produces:

* ``n_macros`` hard macros with log-normally distributed sizes and aspect
  ratios, each with pins spread along its edges;
* ``n_clusters`` soft standard-cell clusters, pre-placed by recursive
  bisection so that logical locality maps to spatial locality (mimicking the
  initial cell placement Circuit Training clusters come from);
* ``n_ports`` I/O ports evenly spread on the die boundary;
* nets with Rent's-rule-like locality: nodes are leaves of an implicit binary
  hierarchy (a random permutation), and a net draws its pins from a block of
  size ``2**l`` around its driver, with ``P(l) ~ 2**(l*(p-1))`` where ``p`` is
  the Rent exponent. Net degrees follow a truncated power law.

Geometry defaults are NanGate45-like (site 0.19 x 1.4 um, 2000 DBU/um).
"""

from __future__ import annotations

import numpy as np

from .design import Cell, Design, Macro, Net, PinRef, Port, Row, Site
from .geometry import Orient, Rect


def _bisect_positions(order: np.ndarray, areas: np.ndarray, rect: tuple[float, float, float, float]) -> dict[int, tuple[float, float]]:
    """Place items (in ``order``) by recursive area-proportional bisection; returns centres."""
    out: dict[int, tuple[float, float]] = {}
    stack = [(order, rect)]
    while stack:
        items, (xl, yl, xh, yh) = stack.pop()
        if len(items) == 1:
            out[int(items[0])] = ((xl + xh) / 2, (yl + yh) / 2)
            continue
        half = len(items) // 2
        a, b = items[:half], items[half:]
        fa = areas[a].sum() / max(areas[items].sum(), 1e-12)
        if (xh - xl) >= (yh - yl):
            xm = xl + fa * (xh - xl)
            stack.append((a, (xl, yl, xm, yh)))
            stack.append((b, (xm, yl, xh, yh)))
        else:
            ym = yl + fa * (yh - yl)
            stack.append((a, (xl, yl, xh, ym)))
            stack.append((b, (xl, ym, xh, yh)))
    return out


def generate_synthetic(
    n_macros: int = 20,
    n_clusters: int = 200,
    n_ports: int = 32,
    seed: int = 0,
    *,
    utilization: float = 0.6,
    macro_area_fraction: float = 0.4,
    nets_per_node: float = 1.3,
    rent_exponent: float = 0.6,
    max_degree: int = 24,
    degree_alpha: float = 2.2,
    aspect_ratio: float = 1.0,
    dbu_per_micron: int = 2000,
    site_width_um: float = 0.19,
    row_height_um: float = 1.4,
    macro_size_um: float = 60.0,
    halo_um: float = 2.0,
    n_fixed_macros: int = 0,
    bus_weights: bool = True,
    routing_demand: float = 0.5,
    name: str | None = None,
) -> Design:
    """Generate a reproducible synthetic design.

    Parameters
    ----------
    n_macros, n_clusters, n_ports:
        Counts of hard macros, soft std-cell clusters and I/O ports.
    seed:
        RNG seed; the same arguments always produce the same design.
    utilization:
        (macro area + cluster area) / core area.
    macro_area_fraction:
        Fraction of placeable area taken by macros.
    nets_per_node:
        Number of nets ~= ``nets_per_node * (n_macros + n_clusters)``.
    rent_exponent:
        Locality of connectivity (lower = more local).
    macro_size_um:
        Median macro edge length in microns.
    n_fixed_macros:
        The first ``n_fixed_macros`` macros are marked fixed (pre-placed on a
        legal shelf at the bottom of the core).
    bus_weights:
        Give nets bus-width weights (macro nets 1..32, cluster nets 1..8), since
        a cluster/macro net stands for many real wires.
    routing_demand:
        Routing capacity is calibrated so that the mean RUDY demand/capacity
        ratio of the generator's initial (bisection) placement equals this.

    Macros are returned unplaced (``placed=False``) at their bisection
    positions, which overlap; run a placer/legaliser before evaluating.
    """
    rng = np.random.default_rng(seed)
    dbu = int(dbu_per_micron)
    site_w = max(1, int(round(site_width_um * dbu)))
    row_h = max(1, int(round(row_height_um * dbu)))

    # ---- macro sizes
    med = macro_size_um * dbu
    if n_macros:
        area = np.exp(rng.normal(np.log(med * med), 0.6, size=n_macros))
        ar = np.exp(rng.uniform(np.log(0.4), np.log(2.5), size=n_macros))
        mw = np.maximum(site_w, np.round(np.sqrt(area * ar) / site_w) * site_w).astype(np.int64)
        mh = np.maximum(row_h, np.round(np.sqrt(area / ar) / row_h) * row_h).astype(np.int64)
        macro_area = float((mw * mh).sum())
    else:
        mw = mh = np.zeros(0, np.int64)
        macro_area = 0.0
    if n_macros and macro_area_fraction > 0:
        cluster_total = macro_area * (1 - macro_area_fraction) / macro_area_fraction
    else:
        cluster_total = float(max(n_clusters, 1)) * (20 * dbu) ** 2
    core_area = (macro_area + cluster_total) / utilization
    core_w = np.sqrt(core_area * aspect_ratio)
    core_h = core_area / core_w
    # make sure the largest macro fits comfortably
    if n_macros:
        core_w = max(core_w, 1.5 * float(mw.max()))
        core_h = max(core_h, 1.5 * float(mh.max()))
    n_sites = int(np.ceil(core_w / site_w))
    n_rows = int(np.ceil(core_h / row_h))
    margin = 10 * row_h
    core = Rect(margin, margin, margin + n_sites * site_w, margin + n_rows * row_h)
    die = Rect(0, 0, core.xh + margin, core.yh + margin)
    rows = [Row(f"ROW_{r}", "core", int(core.xl), int(core.yl + r * row_h), n_sites, 1, site_w, 0, Orient.N if r % 2 == 0 else Orient.FS) for r in range(n_rows)]

    # ---- macros with edge pins
    macros: list[Macro] = []
    for i in range(n_macros):
        w, h = int(mw[i]), int(mh[i])
        npins = int(rng.integers(8, 33))
        pins: dict[str, tuple[float, float]] = {}
        for p in range(npins):
            side = rng.integers(4)
            t = rng.uniform(0.05, 0.95)
            off = {0: (t * w, 0.0), 1: (w, t * h), 2: (t * w, h), 3: (0.0, t * h)}[int(side)]
            pins[f"p{p}"] = (float(round(off[0])), float(round(off[1])))
        macros.append(Macro(f"m{i}", w, h, int(core.xl), int(core.yl), Orient.N, False, False, f"RAM{i}", pins))
    # (positions of non-fixed macros are set from the bisection below)

    # fixed macros: shelf at the bottom-left
    x_cur = core.xl
    for i in range(min(n_fixed_macros, n_macros)):
        m = macros[i]
        if x_cur + m.width > core.xh:
            break
        m.x, m.y, m.fixed, m.placed = int(x_cur), int(core.yl), True, True
        x_cur += m.width + 2 * int(halo_um * dbu)

    # ---- clusters
    n_nodes = n_macros + n_clusters
    cl_area = rng.lognormal(0.0, 0.5, size=n_clusters) if n_clusters else np.zeros(0)
    if n_clusters:
        cl_area = cl_area / cl_area.sum() * cluster_total
    order = rng.permutation(n_nodes)  # leaf order of the implicit hierarchy
    pos_in_order = np.empty(n_nodes, dtype=np.int64)
    pos_in_order[order] = np.arange(n_nodes)
    all_area = np.concatenate([(mw * mh).astype(float), cl_area])
    centres = _bisect_positions(order, np.maximum(all_area, 1.0), (core.xl, core.yl, core.xh, core.yh))
    for i, m in enumerate(macros):
        if not m.fixed:
            cxm, cym = centres[i]
            m.x = int(np.clip(cxm - m.width / 2, core.xl, core.xh - m.width))
            m.y = int(np.clip(cym - m.height / 2, core.yl, core.yh - m.height))
    cells: list[Cell] = []
    for j in range(n_clusters):
        side = np.sqrt(cl_area[j])
        w = max(site_w, int(round(side / site_w)) * site_w)
        h = max(row_h, int(round(cl_area[j] / w / row_h)) * row_h)
        cxc, cyc = centres[n_macros + j]
        x = int(np.clip(cxc - w / 2, core.xl, core.xh - w))
        y = int(np.clip(cyc - h / 2, core.yl, core.yh - h))
        cells.append(Cell(f"c{j}", w, h, x, y, Orient.N, False, True, "CLUSTER", {}, is_cluster=True))

    node_names = [m.name for m in macros] + [c.name for c in cells]
    macro_pin_cursor = np.zeros(n_macros, dtype=np.int64)
    macro_pin_names = [list(m.pins) for m in macros]

    def ref(node: int) -> PinRef:
        if node < n_macros:
            names = macro_pin_names[node]
            k = int(macro_pin_cursor[node] % len(names))
            macro_pin_cursor[node] += 1
            return PinRef(node_names[node], names[k])
        return PinRef(node_names[node], "")

    # ---- nets with hierarchical locality
    levels = max(1, int(np.ceil(np.log2(max(n_nodes, 2)))))
    lv = np.arange(1, levels + 1)
    p_level = 2.0 ** (lv * (rent_exponent - 1.0))
    p_level /= p_level.sum()
    degs = np.arange(2, max_degree + 1)
    p_deg = degs.astype(float) ** (-degree_alpha)
    p_deg /= p_deg.sum()

    nets: list[Net] = []
    touched = np.zeros(n_nodes, dtype=bool)
    n_nets = int(round(nets_per_node * n_nodes)) if n_nodes >= 2 else 0

    def block_members(src: int, size: int) -> np.ndarray:
        p = pos_in_order[src]
        lo = (p // size) * size
        return order[lo : min(lo + size, n_nodes)]

    for k in range(n_nets):
        src = int(rng.integers(n_nodes))
        lvl = int(rng.choice(lv, p=p_level))
        members = block_members(src, 2**lvl)
        while len(members) < 2:
            lvl += 1
            members = block_members(src, 2**lvl)
        d = int(min(rng.choice(degs, p=p_deg), len(members)))
        others = members[members != src]
        sinks = rng.choice(others, size=min(d - 1, len(others)), replace=False)
        nodes = [src, *map(int, sinks)]
        touched[nodes] = True
        weight = 1.0
        if bus_weights:
            if min(nodes) < n_macros:
                weight = float(rng.choice([1, 2, 4, 8, 16, 32], p=[0.2, 0.15, 0.2, 0.2, 0.15, 0.1]))
            else:
                weight = float(rng.integers(1, 9))
        nets.append(Net(f"n{k}", [ref(x) for x in nodes], weight))

    # connect isolated nodes to their hierarchy neighbour
    for v in np.flatnonzero(~touched):
        members = block_members(int(v), 2)
        nb = [int(x) for x in members if x != v] or [int(order[(pos_in_order[v] + 1) % n_nodes])]
        nets.append(Net(f"n{len(nets)}", [ref(int(v)), ref(nb[0])]))

    # ---- ports evenly on the die boundary, each driving a small local net
    ports: list[Port] = []
    perim = 2 * (die.width + die.height)
    for q in range(n_ports):
        t = (q + 0.5) / n_ports * perim
        if t < die.width:
            px, py = t, die.yl
        elif t < die.width + die.height:
            px, py = die.xh, t - die.width
        elif t < 2 * die.width + die.height:
            px, py = die.xh - (t - die.width - die.height), die.yh
        else:
            px, py = die.xl, die.yh - (t - 2 * die.width - die.height)
        pname = f"io{q}"
        ports.append(Port(pname, float(round(px)), float(round(py)), "INPUT" if q % 2 == 0 else "OUTPUT", "metal5"))
        if n_nodes:
            # attach to the node nearest the port among a random sample (spatially plausible)
            cand = rng.choice(n_nodes, size=min(8, n_nodes), replace=False)
            dist = [abs(centres[int(c)][0] - px) + abs(centres[int(c)][1] - py) for c in cand]
            tgt = int(cand[int(np.argmin(dist))])
            nets.append(Net(f"n{len(nets)}", [PinRef(pname, "", True), ref(tgt)]))

    design = Design(
        name=name or f"synth_m{n_macros}_c{n_clusters}_s{seed}",
        dbu_per_micron=dbu,
        die=die,
        core=core,
        site=Site("core", site_w, row_h),
        rows=rows,
        macros=macros,
        cells=cells,
        ports=ports,
        nets=nets,
        halo=int(round(halo_um * dbu)),
        meta={"generator": "synthetic", "seed": seed},
    )
    if routing_demand > 0 and nets:
        _calibrate_routing(design, routing_demand)
    return design


def _calibrate_routing(design: Design, target: float) -> None:
    """Set routes/um so that the mean RUDY ratio of the current placement is ``target``."""
    from ..cost.congestion import rudy_maps
    from ..cost.grid import GridSpec
    from ..cost.wirelength import net_bboxes, pin_positions

    arr = design.arrays()
    cx, cy, o = arr.node_state()
    px, py = pin_positions(arr, cx, cy, o)
    grid = GridSpec.over(design.core, 32)
    h, v = rudy_maps(*net_bboxes(arr, px, py), arr.net_weight, grid)
    # demand per bin / (bin_area * routes_per_dbu) == target  =>  routes_per_dbu
    dbu = design.dbu_per_micron
    design.hroutes_per_micron = float(h.mean() / (grid.bin_area * target) * dbu)
    design.vroutes_per_micron = float(v.mean() / (grid.bin_area * target) * dbu)
