"""Assemble labelled samples (datagen + legacy harness runs) into model-ready records.

A :class:`Sample` holds one placement of one design with

* inputs: :class:`macromind.predict.features.Features` (image + clustered graph),
* congestion labels: GRT H/V utilisation on the same ``G x G`` grid as the
  image (``cong``: H/V utilisation and H/V usage relative to the mean bin capacity), plus gcell-resolution scalars (``hot_frac`` = fraction of
  gcells at >= 90% utilisation, ``util_top``, overflow),
* timing labels: WNS/TNS (ns) from the ORFS GRT metrics, the clock period, and
  (datagen runs) the per-node worst endpoint slack on the clustered graph
  (``node_slack``, NaN where a node has no endpoint),
* GRT wirelength and the MacroMind proxy (the baseline ranker).

Samples are de-duplicated by macro placement (the harness has GRT and finish
runs of the same placement) and cached per run dir under
``runs/datagen/cache/`` (keyed by a feature version and grid size).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..core.design import Design
from ..flow.orfs import REPO_ROOT, DesignSpec
from .features import Features, GraphCache, featurize
from .labels import congestion_maps, congestion_summary, endpoint_instance

FEATURE_VERSION = 6


@dataclass
class Sample:
    design: str
    sample_id: str
    source: str  #: "datagen" | "harness"
    run_dir: str
    feats: Features
    cong: torch.Tensor  #: (4, G, G) GRT utilisation (H, V), usage / mean bin capacity (H, V)
    scalars: dict  #: grt_wl_um, wns_ns, tns_ns, power_w, hot_frac, util_top, overflow, proxy, ...
    node_slack: torch.Tensor | None = None  #: (N,) worst endpoint slack / clock period, NaN = no endpoint
    meta: dict = field(default_factory=dict)

    @property
    def period(self) -> float:
        return float(self.scalars["clock_period_ns"])


class DesignData:
    """Per-design context: clustered design, cluster assignment, name lookups, graph cache."""

    def __init__(self, name: str):
        from ..graph.cluster import cluster_cells
        from ..io.defio import read_def
        from .datagen import context

        self.name = name
        self.ctx = context(name)
        flat = read_def(self.ctx.def_path, self.ctx.lefs)
        flat.halo = int(round(self.ctx.halo_um * flat.dbu_per_micron))
        cl = cluster_cells(flat, None, seed=0)  # same call as macromind.flow.score.load_reduced
        self.flat = flat
        self.red: Design = cl.design
        self.red.halo = flat.halo
        self.graph = GraphCache(self.red)
        nm = len(flat.macros)
        # DEF escapes bus brackets (``reg\\[3\\]``); OpenSTA reports them unescaped
        self.inst_node: dict[str, int] = {m.name.replace("\\", ""): i for i, m in enumerate(flat.macros)}
        for i, c in enumerate(flat.cells):
            if cl.assignment[i] >= 0:
                self.inst_node[c.name.replace("\\", "")] = nm + int(cl.assignment[i])
        nc = len(self.red.cells)
        self.port_node = {p.name: nm + nc + j for j, p in enumerate(self.red.ports)}
        self.period = _clock_period(self.ctx.spec)

    def node_of_endpoint(self, pin: str) -> int | None:
        inst, is_port = endpoint_instance(pin)
        if is_port:
            return self.port_node.get(inst)
        return self.inst_node.get(inst.replace("\\", ""))


def _clock_period(spec: DesignSpec) -> float:
    for p in sorted((REPO_ROOT / "runs/harness/base").glob(f"{spec.tag}-*/results/*/*/*/clock_period.txt")):
        return float(p.read_text().split()[0])
    raise FileNotFoundError(f"no clock_period.txt for {spec}")


_DD: dict[str, DesignData] = {}


def design_data(name: str) -> DesignData:
    if name not in _DD:
        _DD[name] = DesignData(name)
    return _DD[name]


# ----------------------------------------------------------------------------- discovery
def labelled_runs(designs: list[str] | None = None) -> list[tuple[str, Path, str]]:
    """``(design, run_dir, source)`` of every labelled run (datagen first, then harness legacy)."""
    from .datagen import DESIGNS, datagen_dir

    designs = designs or list(DESIGNS)
    out = []
    for d in designs:
        tag = DesignSpec.parse(DESIGNS[d]).tag
        for src, root in (("datagen", datagen_dir() / "jobs" / tag), ("harness", REPO_ROOT / "runs/harness/jobs" / tag)):
            for rd in sorted(root.glob("*")):
                if (rd / "labels" / "done.json").exists() and (rd / "ppa.json").exists():
                    rec = json.loads((rd / "ppa.json").read_text())
                    if rec.get("status") == "ok" and rec.get("ppa", {}).get("grt_wl_um"):
                        out.append((d, rd, src))
    return out


def _placement_key(macros: list[dict]) -> str:
    s = json.dumps(sorted((m["name"], round(m["x"], 2), round(m["y"], 2), m["orient"]) for m in macros))
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- build
def build_sample(design: str, run_dir: Path, source: str, *, G: int = 32) -> Sample:
    from ..core.geometry import Orient
    from ..flow.emit import apply_um_placement

    dd = design_data(design)
    rec = json.loads((run_dir / "ppa.json").read_text())
    macros = rec["placement"]["macros"]
    p = apply_um_placement(dd.red, {m["name"]: (m["x"], m["y"], int(Orient.parse(m["orient"]))) for m in macros})
    feats = featurize(dd.red, p, G=G, graph=dd.graph)
    g = dict(np.load(run_dir / "labels" / "gcells.npz"))
    core = dd.red.core
    cm = congestion_maps(g, G, G, core=(core.xl, core.yl, core.xh, core.yh))
    cong = torch.from_numpy(np.stack([cm["util_h"], cm["util_v"], cm["use_h"], cm["use_v"]]).astype(np.float32))
    cs = congestion_summary(g)
    ppa = rec["ppa"]
    om = rec.get("orfs_metrics", {})
    pr = rec["placement"]
    # always the GRT-stage values (a harness run that went on to `finish` reports post-route timing in `ppa`)
    scalars = {
        "grt_wl_um": float(om.get("globalroute__global_route__wirelength", ppa["grt_wl_um"])),
        "wns_ns": float(om["globalroute__timing__setup__ws"]), "tns_ns": float(om["globalroute__timing__setup__tns"]),
        "power_w": float(om.get("globalroute__power__total", math.nan)),
        "setup_violations": float(om.get("globalroute__timing__drv__setup_violation_count", 0)),
        "grt_overflow": float(ppa.get("grt_overflow") or 0), "clock_period_ns": dd.period,
        "hot_frac": 0.5 * (cs["hot_frac_h"] + cs["hot_frac_v"]), "util_top": cs["util_top"], "util_mean": cs["util_mean"],
        "overflow": cs["overflow"], "proxy": float(pr["proxy"]), "proxy_wirelength": float(pr["wirelength"]),
        "proxy_density": float(pr["density"]), "proxy_congestion": float(pr["congestion"]), "hpwl_um": feats.hpwl_um,
    }
    node_slack = None
    ep_file = run_dir / "labels" / "endpoints.json"
    n_ep = n_mapped = 0
    if ep_file.exists():
        eps = json.loads(ep_file.read_text())
        ns = torch.full((dd.graph.n,), float("nan"))
        for pin, s in eps.items():
            n_ep += 1
            k = dd.node_of_endpoint(pin)
            if k is None:
                continue
            n_mapped += 1
            v = s / dd.period
            if math.isnan(ns[k]) or v < ns[k]:
                ns[k] = v
        node_slack = ns
        scalars["n_endpoints"] = n_ep
    sid = run_dir.name
    meta = {"placement_key": _placement_key(macros), "placer": rec["job"].get("placer"), "n_endpoints": n_ep,
            "n_endpoints_mapped": n_mapped}
    if source == "datagen":
        from .datagen import datagen_dir

        sj = datagen_dir() / "placements" / run_dir.parent.name / sid.rsplit("-", 1)[0] / "spec.json"
        if sj.exists():
            meta["spec"] = json.loads(sj.read_text())["spec"]
    return Sample(design, sid, source, str(run_dir), feats, cong, scalars, node_slack, meta)


def cache_path(run_dir: Path, G: int) -> Path:
    h = hashlib.sha256(f"{run_dir}|{G}|{FEATURE_VERSION}".encode()).hexdigest()[:16]
    return REPO_ROOT / "runs/datagen/cache" / f"{run_dir.name}-{h}.pt"


def load_samples(designs: list[str] | None = None, *, G: int = 32, dedupe: bool = True, log=None) -> list[Sample]:
    """All labelled samples (cached), de-duplicated by placement (datagen runs preferred)."""
    out: list[Sample] = []
    seen: set[tuple[str, str]] = set()
    for d, rd, src in labelled_runs(designs):
        cp = cache_path(rd, G)
        s = None
        if cp.exists():
            try:
                s = torch.load(cp, weights_only=False)
            except Exception:  # noqa: BLE001 -- rebuild a corrupt cache entry
                s = None
        if s is None:
            s = build_sample(d, rd, src, G=G)
            cp.parent.mkdir(parents=True, exist_ok=True)
            torch.save(s, cp)
            if log:
                log(f"[dataset] built {d} {rd.name}")
        k = (d, s.meta["placement_key"])
        if dedupe and k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def dataset_stats(samples: list[Sample]) -> dict:
    by: dict[str, dict] = {}
    for s in samples:
        b = by.setdefault(s.design, {"n": 0, "with_endpoints": 0, "sources": {}, "kinds": {}, "grt_wl_um": [], "wns_ns": [],
                                     "tns_ns": [], "hot_frac": [], "proxy": []})
        b["n"] += 1
        b["with_endpoints"] += s.node_slack is not None
        b["sources"][s.source] = b["sources"].get(s.source, 0) + 1
        kind = s.meta.get("spec", {}).get("kind", s.meta.get("placer"))
        b["kinds"][kind] = b["kinds"].get(kind, 0) + 1
        for k in ("grt_wl_um", "wns_ns", "tns_ns", "hot_frac", "proxy"):
            b[k].append(s.scalars[k])
    for b in by.values():
        for k in ("grt_wl_um", "wns_ns", "tns_ns", "hot_frac", "proxy"):
            v = np.asarray(b[k])
            b[k] = {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()), "std": float(v.std())}
    return by
