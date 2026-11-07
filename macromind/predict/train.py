"""Training, cross-validation and ranking evaluation of the predictors.

Targets (see :mod:`macromind.predict.dataset`):

* congestion model: per-bin GRT utilisation map (H, V) + scalars
  ``[log(GRT WL / cluster HPWL), hot_frac]``;
* timing model: per-node worst endpoint slack / T (datagen samples) + scalars
  ``[WNS / T, -log1p(-TNS / T)]``.

Scalar targets are standardised with training-set statistics (kept in the
bundle). Augmentation: random mirror of the whole layout in x and/or y
(inputs and the congestion map flipped together; timing labels are invariant).

CLI::

    python -m macromind.predict.train cv    [--folds 5] [--seeds 3] [--epochs 150] [--out runs/predict/cv.json]
    python -m macromind.predict.train lodo  [--seeds 3]
    python -m macromind.predict.train fit   [--seeds 3] --out runs/predict/bundle.pt     # train on everything
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import Sample, load_samples
from .models import Ensemble, build_model

TARGETS = {
    "congestion": ["log_wl_ratio", "hot_frac"],
    "timing": ["wns_t", "tns_t"],
}


def scalar_targets(s: Sample, kind: str) -> torch.Tensor:
    sc = s.scalars
    if kind == "congestion":
        return torch.tensor([math.log(sc["grt_wl_um"] / sc["hpwl_um"]), sc["hot_frac"]], dtype=torch.float32)
    T = sc["clock_period_ns"]
    return torch.tensor([sc["wns_ns"] / T, -math.log1p(max(0.0, -sc["tns_ns"]) / T)], dtype=torch.float32)


@dataclass
class TrainConfig:
    epochs: int = 150
    lr: float = 2e-3
    weight_decay: float = 1e-4
    map_weight: float = 1.0
    node_weight: float = 1.0
    scalar_weight: float = 1.0
    augment: bool = True
    seed: int = 0
    model: dict = field(default_factory=dict)


@dataclass
class Normalizer:
    """Target standardisation: global statistics plus per-design ones.

    Rankings are within a design, so the scalar heads are trained on targets
    standardised *per design* (a design's mean/std over its training samples);
    a design without training samples (leave-one-design-out, or a new design in
    the RL loop) falls back to the global statistics, which only matters for the
    absolute (denormalised) values, not for the ranking.
    """

    mean: torch.Tensor
    std: torch.Tensor
    per_design: dict = field(default_factory=dict)  #: design -> (mean, std)

    @classmethod
    def fit(cls, ys: torch.Tensor, designs: list[str] | None = None) -> "Normalizer":
        g_std = ys.std(0, unbiased=False).clamp_min(1e-6) if len(ys) > 1 else torch.ones(ys.shape[1])
        n = cls(ys.mean(0), g_std)
        for d in sorted(set(designs or [])):
            yd = ys[[i for i, x in enumerate(designs) if x == d]]
            if len(yd) >= 3:
                n.per_design[d] = (yd.mean(0), torch.maximum(yd.std(0, unbiased=False), 0.05 * g_std).clamp_min(1e-6))
        return n

    def stats(self, design: str | None = None):
        return self.per_design.get(design, (self.mean, self.std))

    def __call__(self, y, design: str | None = None):
        m, s = self.stats(design)
        return (y - m) / s

    def inv(self, z, design: str | None = None):
        m, s = self.stats(design)
        return z * s + m

    def state(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist(),
                "per_design": {d: [m.tolist(), s.tolist()] for d, (m, s) in self.per_design.items()}}

    @classmethod
    def from_state(cls, d):
        n = cls(torch.tensor(d["mean"]), torch.tensor(d["std"]))
        n.per_design = {k: (torch.tensor(m), torch.tensor(s)) for k, (m, s) in d.get("per_design", {}).items()}
        return n


def _loss(model, kind: str, s: Sample, y_norm: torch.Tensor, cfg: TrainConfig, fx: bool, fy: bool) -> torch.Tensor:
    f = s.feats.flipped(fx, fy) if (fx or fy) else s.feats
    if kind == "congestion":
        m, sc = model(f)
        tgt = s.cong
        if fx:
            tgt = tgt.flip(-1)
        if fy:
            tgt = tgt.flip(-2)
        return cfg.map_weight * F.mse_loss(m, tgt) / 0.01 + cfg.scalar_weight * F.mse_loss(sc, y_norm)
    node, sc = model(f)
    loss = cfg.scalar_weight * F.mse_loss(sc, y_norm)
    if s.node_slack is not None:
        ok = ~torch.isnan(s.node_slack)
        if ok.any():
            loss = loss + cfg.node_weight * F.mse_loss(node[ok], s.node_slack[ok]) / 0.01
    return loss


def train_model(kind: str, samples: list[Sample], cfg: TrainConfig | None = None, *, norm: Normalizer | None = None,
                log=None) -> tuple[torch.nn.Module, Normalizer, dict]:
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    ys = torch.stack([scalar_targets(s, kind) for s in samples])
    norm = norm or Normalizer.fit(ys, [s.design for s in samples])
    yn = torch.stack([norm(y, s.design) for y, s in zip(ys, samples)])
    model = build_model(kind, cfg.model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
    t0 = time.perf_counter()
    hist = []
    idx = list(range(len(samples)))
    for ep in range(cfg.epochs):
        model.train()
        rng.shuffle(idx)
        tot = 0.0
        for i in idx:
            fx = cfg.augment and rng.random() < 0.5
            fy = cfg.augment and rng.random() < 0.5
            loss = _loss(model, kind, samples[i], yn[i], cfg, fx, fy)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
        sched.step()
        hist.append(tot / max(1, len(idx)))
        if log and (ep % 25 == 0 or ep == cfg.epochs - 1):
            log(f"[train] {kind} seed={cfg.seed} epoch {ep} loss {hist[-1]:.4f}")
    model.eval()
    return model, norm, {"train_s": round(time.perf_counter() - t0, 1), "final_loss": hist[-1] if hist else None,
                         "loss_curve": hist[:: max(1, len(hist) // 20)]}


def train_ensemble(kind: str, samples: list[Sample], seeds: int = 3, cfg: TrainConfig | None = None, log=None):
    cfg = cfg or TrainConfig()
    ys = torch.stack([scalar_targets(s, kind) for s in samples])
    norm = Normalizer.fit(ys, [s.design for s in samples])
    members, infos = [], []
    for k in range(seeds):
        c = TrainConfig(**{**cfg.__dict__, "seed": cfg.seed + k})
        m, _, info = train_model(kind, samples, c, norm=norm, log=log)
        members.append(m)
        infos.append(info)
    return Ensemble(members), norm, infos


@torch.no_grad()
def predict(ens: Ensemble, norm: Normalizer, kind: str, s: Sample) -> dict:
    """Denormalised predictions with ensemble std for one sample."""
    (a_mean, a_std), (sc_mean, sc_std) = ens(s.feats)
    sc = norm.inv(sc_mean, s.design)
    sc_sd = sc_std * norm.stats(s.design)[1]
    out = {"scalars": sc.tolist(), "scalars_std": sc_sd.tolist()}
    if kind == "congestion":
        out["map"] = a_mean
        out["map_std"] = a_std
        out["grt_wl_um"] = float(math.exp(sc[0]) * s.scalars["hpwl_um"])
        out["hot_frac"] = float(sc[1])
        util = a_mean[:2].flatten()
        out["map_top"] = float(util.topk(max(1, util.numel() // 10)).values.mean())  # mean of the hottest 10% util bins
        out["map_use"] = float(a_mean[2:].sum())  # total predicted routing usage (wirelength-like)
    else:
        out["node"] = a_mean
        out["node_std"] = a_std
        T = s.scalars["clock_period_ns"]
        out["wns_ns"] = float(sc[0] * T)
        out["tns_ns"] = float(-(math.expm1(-sc[1])) * T)
    return out


# ----------------------------------------------------------------------------- metrics
def kendall_tau(a, b) -> float:
    """Kendall tau-b (ties handled), pure numpy."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    if n < 2:
        return float("nan")
    i, j = np.triu_indices(n, 1)
    da, db = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
    num = float((da * db).sum())
    den = math.sqrt(float((da != 0).sum()) * float((db != 0).sum()))
    return num / den if den > 0 else float("nan")


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    c = np.corrcoef(ra, rb)[0, 1]
    return float(c)


#: ranking targets: name -> (true cost from scalars, which model, predicted cost from predictions)
RANK_TARGETS = {
    "grt_wl": (lambda s: s["grt_wl_um"], "congestion", lambda p: p["grt_wl_um"]),
    "grt_wl_map": (lambda s: s["grt_wl_um"], "congestion", lambda p: p.get("map_use", float("nan"))),
    "hot_frac": (lambda s: s["hot_frac"], "congestion", lambda p: p["hot_frac"]),
    "hot_frac_map": (lambda s: s["hot_frac"], "congestion", lambda p: p.get("map_top", float("nan"))),
    "wns": (lambda s: -s["wns_ns"], "timing", lambda p: -p["wns_ns"]),
    "tns": (lambda s: -s["tns_ns"], "timing", lambda p: -p["tns_ns"]),
}


def handcrafted(s: Sample) -> np.ndarray:
    """Placement-level features for the linear baseline: proxy components + image statistics."""
    sc, img = s.scalars, s.feats.img
    v = [sc["proxy"], sc["proxy_wirelength"], sc["proxy_density"], sc["proxy_congestion"]]
    for c in range(6):
        x = img[c].flatten()
        v += [float(x.mean()), float(x.max()), float(torch.quantile(x, 0.9))]
    return np.asarray(v, float)


def ridge_predict(train: list[Sample], test: list[Sample], alpha: float = 1.0) -> list[dict]:
    """Ridge regression (standardised handcrafted features) on the same normalised targets as the networks."""
    X = np.stack([handcrafted(s) for s in train])
    Y = np.stack([np.concatenate([scalar_targets(s, "congestion").numpy(), scalar_targets(s, "timing").numpy()]) for s in train])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    ym = Y.mean(0)
    Z = (X - mu) / sd
    W = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (Y - ym))
    out = []
    for s in test:
        y = ((handcrafted(s) - mu) / sd) @ W + ym
        T = s.scalars["clock_period_ns"]
        out.append({"grt_wl_um": float(math.exp(y[0]) * s.scalars["hpwl_um"]), "hot_frac": float(y[1]),
                    "wns_ns": float(y[2] * T), "tns_ns": float(-math.expm1(-y[3]) * T)})
    return out


def node_metrics(samples: list[Sample], preds: list[dict], baselines: list) -> dict:
    """Per design: Spearman (pooled over nodes with endpoints) of predicted vs true per-node slack,
    and of a placement-independent baseline (per-node mean slack over the training placements)."""
    out = {}
    for d in sorted({s.design for s in samples}):
        rp, rb, n = [], [], 0
        for i, s in enumerate(samples):
            if s.design != d or s.node_slack is None or preds[i] is None or "node" not in preds[i]:
                continue
            ok = ~torch.isnan(s.node_slack)
            t = s.node_slack[ok].numpy()
            rp.append(spearman(preds[i]["node"][ok].numpy(), t))
            b = baselines[i]
            if b is not None:
                bb = b[ok.numpy()]
                good = ~np.isnan(bb)
                if good.sum() > 2:
                    rb.append(spearman(bb[good], t[good]))
            n += 1
        if n:
            out[d] = {"n_samples": n, "rho_node_pred": float(np.nanmean(rp)),
                      "rho_node_train_mean": float(np.nanmean(rb)) if rb else None}
    return out


def _node_mean_baseline(train: list[Sample], s: Sample):
    arrs = [t.node_slack.numpy() for t in train if t.design == s.design and t.node_slack is not None]
    if not arrs or s.node_slack is None:
        return None
    import warnings

    with warnings.catch_warnings():  # all-NaN columns (nodes never an endpoint) stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(arrs), 0)


def ranking_table(samples: list[Sample], preds: dict[str, list[dict]], ridge: list[dict] | None = None) -> dict:
    """Per design and target: Kendall tau / Spearman of predictor vs truth and of the proxy (and HPWL) vs truth."""
    out: dict = {}
    designs = sorted({s.design for s in samples})
    for d in designs:
        ix = [i for i, s in enumerate(samples) if s.design == d]
        row = {"n": len(ix)}
        for t, (truef, kind, predf) in RANK_TARGETS.items():
            if kind not in preds:
                continue
            truth = [truef(samples[i].scalars) for i in ix]
            pr = [predf(preds[kind][i]) for i in ix]
            prox = [samples[i].scalars["proxy"] for i in ix]
            hp = [samples[i].scalars["hpwl_um"] for i in ix]
            row[t] = {"tau_pred": kendall_tau(pr, truth), "tau_proxy": kendall_tau(prox, truth),
                      "tau_ridge": kendall_tau([predf(ridge[i]) for i in ix], truth) if ridge and not t.endswith("_map") else None,
                      "tau_hpwl": kendall_tau(hp, truth), "rho_pred": spearman(pr, truth), "rho_proxy": spearman(prox, truth),
                      "n_distinct_truth": len(set(np.round(truth, 9)))}
            unc = [preds[kind][i]["scalars_std"][0 if t in ("grt_wl", "wns") else 1] for i in ix]
            if t.endswith("_map"):
                unc = [float(preds[kind][i]["map_std"].mean()) if "map_std" in preds[kind][i] else float("nan") for i in ix]
            err = [abs(a - b) for a, b in zip(pr, truth)]
            row[t]["rho_unc_err"] = spearman(unc, err)
        out[d] = row
    return out


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().numpy().astype(float), b.flatten().numpy().astype(float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def map_metrics(samples: list[Sample], preds: list[dict], base: list | None = None) -> dict:
    """Per design: MSE / MAE / pixel correlation of the predicted vs true GRT utilisation map.

    Baselines: the RUDY channels of the input (``pixel_corr_rudy``) and, if ``base``
    is given, the mean training map of the same design (``*_design_mean``: knows
    the design's typical map but nothing about this placement). ``pixel_corr_delta``
    correlates (prediction - design mean) with (truth - design mean), i.e. how well
    the placement-induced change of the map is predicted.
    """
    out = {}
    for d in sorted({s.design for s in samples}):
        ix = [i for i, s in enumerate(samples) if s.design == d]
        acc: dict[str, list] = {}
        for i in ix:
            t = samples[i].cong[:2]  # utilisation channels
            p = preds[i]["map"][:2]
            acc.setdefault("mse", []).append(float(((p - t) ** 2).mean()))
            acc.setdefault("mae", []).append(float((p - t).abs().mean()))
            acc.setdefault("pixel_corr", []).append(_corr(p, t))
            acc.setdefault("pixel_corr_rudy", []).append(_corr(samples[i].feats.img[2:4], t))
            if base is not None and base[i] is not None:
                b = base[i][:2]
                acc.setdefault("mse_design_mean", []).append(float(((b - t) ** 2).mean()))
                acc.setdefault("pixel_corr_design_mean", []).append(_corr(b, t))
                acc.setdefault("pixel_corr_delta", []).append(_corr(p - b, t - b))
        out[d] = {k: float(np.nanmean(v)) if len(v) else None for k, v in acc.items()}
    return out


# ----------------------------------------------------------------------------- protocols
def _strip(p: dict) -> dict:
    return {k: v for k, v in p.items() if not isinstance(v, torch.Tensor)}


def cross_validate(samples: list[Sample], *, folds: int = 5, seeds: int = 3, cfg: TrainConfig | None = None, log=print,
                   kinds=("congestion", "timing")) -> dict:
    """Held-out placements per design: K folds within each design; train on the other folds + all other designs."""
    rng = np.random.default_rng(0)
    fold_of = np.zeros(len(samples), int)
    for d in {s.design for s in samples}:
        ix = [i for i, s in enumerate(samples) if s.design == d]
        perm = rng.permutation(ix)
        for k, i in enumerate(perm):
            fold_of[i] = k % folds
    preds: dict[str, list] = {k: [None] * len(samples) for k in kinds}
    ridge: list = [None] * len(samples)
    node_base: list = [None] * len(samples)
    map_base: list = [None] * len(samples)
    timing: dict[str, list] = {k: [] for k in kinds}
    for k in range(folds):
        tr = [s for i, s in enumerate(samples) if fold_of[i] != k]
        te = [i for i in range(len(samples)) if fold_of[i] == k]
        if not te:
            continue
        for i, r in zip(te, ridge_predict(tr, [samples[i] for i in te])):
            ridge[i] = r
        for i in te:
            node_base[i] = _node_mean_baseline(tr, samples[i])
            same = [t.cong for t in tr if t.design == samples[i].design]
            map_base[i] = torch.stack(same).mean(0) if same else None
        for kind in kinds:
            ens, norm, infos = train_ensemble(kind, tr, seeds, cfg)
            timing[kind].append(sum(i["train_s"] for i in infos))
            for i in te:
                preds[kind][i] = predict(ens, norm, kind, samples[i])
        log(f"[cv] fold {k + 1}/{folds} done ({len(tr)} train / {len(te)} test)")
    res = {"protocol": f"{folds}-fold held-out placements per design (other designs always in train)", "seeds": seeds,
           "ranking": ranking_table(samples, preds, ridge), "train_s_per_fold": {k: v for k, v in timing.items()}}
    if "congestion" in preds:
        res["maps"] = map_metrics(samples, preds["congestion"], map_base)
    if "timing" in preds:
        res["nodes"] = node_metrics(samples, preds["timing"], node_base)
    res["predictions"] = [{"design": s.design, "id": s.sample_id, "truth": s.scalars,
                           **{k: _strip(preds[k][i]) for k in kinds}} for i, s in enumerate(samples)]
    return res


def leave_one_design_out(samples: list[Sample], *, seeds: int = 3, cfg: TrainConfig | None = None, log=print,
                         kinds=("congestion", "timing")) -> dict:
    designs = sorted({s.design for s in samples})
    preds: dict[str, list] = {k: [None] * len(samples) for k in kinds}
    ridge: list = [None] * len(samples)
    for d in designs:
        tr = [s for s in samples if s.design != d]
        te = [i for i, s in enumerate(samples) if s.design == d]
        for i, r in zip(te, ridge_predict(tr, [samples[i] for i in te])):
            ridge[i] = r
        for kind in kinds:
            ens, norm, _ = train_ensemble(kind, tr, seeds, cfg)
            for i in te:
                preds[kind][i] = predict(ens, norm, kind, samples[i])
        log(f"[lodo] held out {d} ({len(te)} samples, trained on {len(tr)})")
    res = {"protocol": "leave one design out", "ranking": ranking_table(samples, preds, ridge)}
    if "congestion" in preds:
        res["maps"] = map_metrics(samples, preds["congestion"])
    if "timing" in preds:
        res["nodes"] = node_metrics(samples, preds["timing"], [None] * len(samples))
    return res


# ----------------------------------------------------------------------------- bundles
def save_bundle(path: str | Path, ensembles: dict, meta: dict | None = None) -> None:
    """``ensembles = {kind: (Ensemble, Normalizer)}`` -> one torch file."""
    from .models import model_config_dict

    blob = {"version": 1, "meta": meta or {}, "models": {}}
    for kind, (ens, norm) in ensembles.items():
        blob["models"][kind] = {"arch": model_config_dict(ens.members[0]), "norm": norm.state(),
                                "states": [m.state_dict() for m in ens.members]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)


def load_bundle(path: str | Path) -> dict:
    blob = torch.load(path, weights_only=False)
    out = {}
    for kind, d in blob["models"].items():
        members = []
        for st in d["states"]:
            m = build_model(kind, d["arch"]["config"])
            m.load_state_dict(st)
            m.eval()
            members.append(m)
        out[kind] = (Ensemble(members), Normalizer.from_state(d["norm"]))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m macromind.predict.train", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["cv", "lodo", "fit", "stats"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--kinds", nargs="+", default=["congestion", "timing"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    samples = load_samples(log=print)
    cfg = TrainConfig(epochs=a.epochs)
    t = time.time()
    if a.cmd == "stats":
        from .dataset import dataset_stats

        res = dataset_stats(samples)
    elif a.cmd == "cv":
        res = cross_validate(samples, folds=a.folds, seeds=a.seeds, cfg=cfg, kinds=a.kinds)
    elif a.cmd == "lodo":
        res = leave_one_design_out(samples, seeds=a.seeds, cfg=cfg, kinds=a.kinds)
    else:
        ens = {k: train_ensemble(k, samples, a.seeds, cfg, log=print)[:2] for k in a.kinds}
        out = a.out or "runs/predict/bundle.pt"
        save_bundle(out, ens, {"n_samples": len(samples), "designs": sorted({s.design for s in samples}),
                               "epochs": a.epochs, "seeds": a.seeds})
        res = {"bundle": out}
    res["wall_s"] = round(time.time() - t, 1)
    res["n_samples"] = len(samples)
    txt = json.dumps(res, indent=1, default=float)
    if a.out and a.cmd != "fit":
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt)
    print(json.dumps({k: v for k, v in res.items() if k != "predictions"}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
