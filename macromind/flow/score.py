"""MacroMind proxy score of a real (DEF) floorplan, the same way for every placer.

The flow DEF after macro placement has macros placed and std cells unplaced.
We cluster the std cells (deterministic HEM, seed 0), place the clusters
around the macros with the quadratic cluster pass, and evaluate the TILOS-style
proxy (wirelength + density + congestion). The result is what we correlate
against routed PPA.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

from ..core.design import Design, Placement
from ..cost.proxy import ProxyConfig, proxy_cost


def load_reduced(def_path: str | Path, lefs: Iterable[str | Path], *, halo_um: float = 0.0, n_clusters: int | None = None,
                 cluster_seed: int = 0) -> tuple[Design, Design, dict]:
    """Read DEF+LEF and return ``(flat design, clustered design, stats)``."""
    from ..graph.cluster import cluster_cells
    from ..io.defio import read_def

    t0 = time.perf_counter()
    flat = read_def(def_path, list(lefs))
    flat.halo = int(round(halo_um * flat.dbu_per_micron))
    t1 = time.perf_counter()
    cl = cluster_cells(flat, n_clusters, seed=cluster_seed)
    red = cl.design
    red.halo = flat.halo
    if [m.name for m in red.macros] != [m.name for m in flat.macros]:
        raise RuntimeError("clustering changed the macro order")
    stats = {"read_s": round(t1 - t0, 3), "cluster_s": round(time.perf_counter() - t1, 3), **cl.stats,
             "n_macros": len(flat.macros), "unknown_masters": flat.meta.get("unknown_masters", 0)}
    return flat, red, stats


def score_placement(reduced: Design, placement: Placement, config: ProxyConfig | None = None) -> dict:
    """Proxy of ``placement`` on the clustered design, after the cluster pass."""
    from ..graph.quadratic import place_clusters
    from ..legalize.legalizer import check_legality

    cfg = config or ProxyConfig()
    t = time.perf_counter()
    scored = place_clusters(reduced, placement)
    res = proxy_cost(scored, placement, cfg)
    raw = proxy_cost(reduced, placement, cfg)
    errs = check_legality(reduced, placement, check_grid=False)
    return {**res.as_dict(), "proxy_no_cluster_pass": raw.proxy, "legal": not errs, "legality_errors": errs[:5],
            "score_s": round(time.perf_counter() - t, 3)}


def score_def(def_path: str | Path, lefs: Iterable[str | Path], *, halo_um: float = 0.0) -> dict:
    """Proxy score of the macro placement stored in a DEF."""
    flat, red, stats = load_reduced(def_path, lefs, halo_um=halo_um)
    p = red.placement()
    unplaced = [m.name for m in red.macros if not m.placed]
    out = score_placement(red, p)
    out.update(unplaced_macros=len(unplaced), clustering=stats, macros=placement_record(red, p))
    return out


def placement_record(design: Design, placement: Placement) -> list[dict]:
    """JSON-friendly macro placement (microns, OpenDB orientation names)."""
    from .emit import OPENDB_ORIENT

    dbu = float(design.dbu_per_micron)
    return [
        {"name": m.name, "master": m.master, "x": round(placement.x[i] / dbu, 4), "y": round(placement.y[i] / dbu, 4),
         "orient": OPENDB_ORIENT[int(placement.orient[i])]}
        for i, m in enumerate(design.macros)
    ]
