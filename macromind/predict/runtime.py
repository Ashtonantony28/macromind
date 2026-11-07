"""Inference-time wrapper: trained predictor ensembles -> penalties for the RL reward / candidate ranking.

    from macromind.predict.runtime import Predictor
    pred = Predictor.load("runs/predict/bundle.pt")
    out = pred.predict(design, placement)          # hot_frac, wns_ns, grt_wl_um, ... with ensemble std
    mu, sigma = pred.penalty(design, placement)    # standardised congestion + timing badness

    env_cfg = EnvConfig(predictor=pred, pred_weight=0.05, unc_weight=0.05)   # RL reward hook

``design`` is the clustered design the placer works on (e.g. the RL env's design,
or ``load_reduced``'s second value). The penalty is in units of training-set
standard deviations: ``w_congestion * z(hot_frac) + w_timing * z(-WNS/T) + w_wl * z(log WL/HPWL)``;
``sigma`` combines the ensembles' member spread the same way. Inputs are
re-featurised per call (the cluster pass is reused if ``scored`` is given).
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from ..core.design import Design, Placement
from .features import GraphCache, featurize


class Predictor:
    def __init__(self, ensembles: dict, *, G: int = 32, w_congestion: float = 1.0, w_timing: float = 1.0, w_wl: float = 0.0):
        self.ensembles = ensembles  # kind -> (Ensemble, Normalizer)
        self.G = G
        self.w = {"congestion": w_congestion, "timing": w_timing, "wl": w_wl}
        self._graphs: dict[int, tuple[Design, GraphCache]] = {}

    @classmethod
    def load(cls, path: str | Path, **kw) -> "Predictor":
        from .train import load_bundle

        return cls(load_bundle(path), **kw)

    def _graph(self, design: Design) -> GraphCache:
        k = id(design)
        hit = self._graphs.get(k)
        if hit is None or hit[0] is not design:
            hit = (design, GraphCache(design))
            self._graphs[k] = hit
        return hit[1]

    @torch.no_grad()
    def raw(self, design: Design, placement: Placement, scored: Design | None = None) -> dict:
        """Per kind: ``(main mean, main std, scalars mean (normalised), scalars std (normalised))``."""
        f = featurize(design, placement, G=self.G, scored=scored, graph=self._graph(design))
        out = {"_features": f}
        for kind, (ens, _) in self.ensembles.items():
            (a, a_sd), (s, s_sd) = ens(f)
            out[kind] = (a, a_sd, s, s_sd)
        return out

    def predict(self, design: Design, placement: Placement, scored: Design | None = None) -> dict:
        t = time.perf_counter()
        r = self.raw(design, placement, scored)
        f = r["_features"]
        res: dict = {"infer_s": None}
        if "congestion" in r:
            m, m_sd, s, s_sd = r["congestion"]
            norm = self.ensembles["congestion"][1]
            v, sd = norm.inv(s), s_sd * norm.std
            res.update(grt_wl_um=float(math.exp(v[0]) * f.hpwl_um), grt_wl_rel_std=float(sd[0]), hot_frac=float(v[1]),
                       hot_frac_std=float(sd[1]), congestion_map=m, congestion_map_std=m_sd)
        if "timing" in r:
            node, node_sd, s, s_sd = r["timing"]
            norm = self.ensembles["timing"][1]
            v, sd = norm.inv(s), s_sd * norm.std
            res.update(wns_over_t=float(v[0]), wns_over_t_std=float(sd[0]), tns_log=float(v[1]), node_slack_over_t=node,
                       node_slack_std=node_sd)
        res["infer_s"] = round(time.perf_counter() - t, 4)
        return res

    def penalty(self, design: Design, placement: Placement, scored: Design | None = None) -> tuple[float, float]:
        """``(mu, sigma)`` of the standardised predicted badness (higher = worse)."""
        r = self.raw(design, placement, scored)
        mu, var = 0.0, 0.0
        if "congestion" in r:
            _, _, s, s_sd = r["congestion"]
            for w, k in ((self.w["congestion"], 1), (self.w["wl"], 0)):
                mu += w * float(s[k])
                var += (w * float(s_sd[k])) ** 2
        if "timing" in r:
            _, _, s, s_sd = r["timing"]
            w = self.w["timing"]
            mu += -w * float(s[0])  # higher WNS is better
            var += (w * float(s_sd[0])) ** 2
        return mu, math.sqrt(var)
