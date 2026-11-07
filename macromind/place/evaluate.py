"""Evaluate RL checkpoints against the baselines on the same designs.

Every method's macro placement goes through the same legaliser and then the
same evaluation protocol (:func:`~macromind.place.env.evaluate_placement`:
cluster placement pass, then proxy cost). Methods:

* ``random@k``: best of ``k`` random legal placements;
* ``grid``: shelf packing;
* ``wiremask``: greedy heuristic: each macro (largest first) goes to the
  valid cell with the smallest wire-mask value (no learning);
* ``sa@N``: simulated annealing with ``N`` proxy evaluations (incremental,
  without the cluster pass during the search);
* ``sa-alt@N``: cluster-pass-aware SA: grid start, then twice (cluster
  pass, SA with ``N/2`` evaluations from the current macros);
* ``rl:NAME`` greedy and ``rl:NAME@k`` (greedy plus ``k`` sampled episodes,
  best kept). One evaluation = one full episode with the cluster pass.

Example::

    python -m macromind.place.evaluate --ckpt pre=runs/rl-v1/policy.pt --synthetic 8-20 \\
        --design-seeds 1001 1002 --k 32 --sa-budgets 32 20000 --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

from ..baselines.placers import SAConfig, grid_place, random_place, sa_place
from ..core.design import Design, Placement
from ..cost.proxy import proxy_cost
from ..legalize.legalizer import check_legality, legalize
from .agent import load_checkpoint, run_policy
from .env import EnvConfig, PlacementEnv, evaluate_placement
from .ppo import DesignTask
from .train import synthetic_designs


def wiremask_place(design: Design, env_cfg: EnvConfig | None = None, reference: float | None = None) -> Placement:
    env = PlacementEnv(design, env_cfg or EnvConfig(), reference=reference if reference is not None else 1.0)
    obs = env.reset()
    while not env.done:
        wm = obs["image"][5].ravel().copy()
        wm[~obs["mask"]] = np.inf
        obs, *_ = env.step(int(np.argmin(wm)))
    return legalize(design, env.placement())


def sa_alt_place(design: Design, seed: int = 0, budget: int = 20000, rounds: int = 2) -> Placement:
    """SA aware of the cluster pass: alternate (cluster pass -> SA from the current macros) ``rounds`` times."""
    from ..graph.quadratic import place_clusters

    p = grid_place(design, seed)
    for r in range(rounds):
        d = place_clusters(design, p)  # clusters follow the current macros; d has macros at p
        p = sa_place(d, seed + r, budget // rounds, SAConfig(init="current"))
    p.meta.update(method="sa-alt", evaluations=budget)
    return p


def _rec(design, method, p, t0, evals, extra=None):
    runtime = time.perf_counter() - t0
    res, _ = evaluate_placement(design, p)
    nocp = proxy_cost(design, p)  # clusters left at the design's own positions
    return {"design": design.name, "n_macros": len(design.macros), "method": method, **res.as_dict(),
            "proxy_nocp": nocp.proxy,
            "legal": not check_legality(design, p), "runtime_s": round(runtime, 3), "evaluations": evals,
            **(extra or {})}


def evaluate_design(design: Design, ckpts: dict[str, str], *, k: int = 32, sa_budgets=(20000,), seed: int = 0, sa_alt: int = 0,
                    log=print) -> list[dict]:
    out = []
    t = time.perf_counter()
    best = None
    for s in range(k):
        p = random_place(design, seed=seed + s)
        r = evaluate_placement(design, p)[0].proxy
        if best is None or r < best[0]:
            best = (r, p)
    out.append(_rec(design, f"random@{k}", best[1], t, k))
    t = time.perf_counter()
    out.append(_rec(design, "grid", grid_place(design, seed), t, 1))
    t = time.perf_counter()
    out.append(_rec(design, "wiremask", wiremask_place(design), t, 1))
    for b in sa_budgets:
        t = time.perf_counter()
        p = sa_place(design, seed, b, SAConfig())
        out.append(_rec(design, f"sa@{b}", p, t, b, {"sa_own_proxy": p.meta.get("best_proxy")}))
    if sa_alt:
        t = time.perf_counter()
        out.append(_rec(design, f"sa-alt@{sa_alt}", sa_alt_place(design, seed, sa_alt), t, sa_alt))
    for name, path in ckpts.items():
        policy, env_cfg, _ = load_checkpoint(path)
        task = DesignTask(design, env_cfg)
        t = time.perf_counter()
        p, _ = run_policy(policy, design, env_cfg, k=0, seed=seed, task=task)
        out.append(_rec(design, f"rl:{name}", p, t, 1))
        t = time.perf_counter()
        p, _ = run_policy(policy, design, env_cfg, k=k, seed=seed, task=task)
        out.append(_rec(design, f"rl:{name}@{k}", p, t, k + 1))
    for r in out:
        log(f"  {r['design']:<22} {r['method']:<18} proxy {r['proxy']:.4f} wl {r['wirelength']:.4f} "
            f"den {r['density']:.4f} cong {r['congestion']:.4f} nocp {r['proxy_nocp']:.4f} {r['runtime_s']:7.2f}s legal={r['legal']}")
    return out


def markdown_table(records: list[dict]) -> str:
    lines = ["| design | method | proxy | wirelength | density | congestion | proxy (no cluster pass) | evals | runtime (s) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        lines.append(f"| {r['design']} | {r['method']} | {r['proxy']:.4f} | {r['wirelength']:.4f} | {r['density']:.4f} | "
                     f"{r['congestion']:.4f} | {r['proxy_nocp']:.4f} | {r['evaluations']} | {r['runtime_s']:.1f} |")
    return "\n".join(lines)


def summary_table(records: list[dict]) -> str:
    """Mean proxy ratio to grid per method, and win counts vs SA."""
    by = {}
    for r in records:
        by.setdefault(r["design"], {})[r["method"]] = r["proxy"]
    methods = list(dict.fromkeys(r["method"] for r in records))
    nocp = {}
    for r in records:
        nocp.setdefault(r["design"], {})[r["method"]] = r["proxy_nocp"]
    lines = ["| method | mean proxy | proxy / grid | proxy / grid (no cluster pass) | designs |", "|---|---|---|---|---|"]
    for m in methods:
        ds = [k for k, d in by.items() if m in d and "grid" in d]
        if ds:
            lines.append(f"| {m} | {np.mean([by[k][m] for k in ds]):.4f} | {np.mean([by[k][m] / by[k]['grid'] for k in ds]):.3f} | "
                         f"{np.mean([nocp[k][m] / nocp[k]['grid'] for k in ds]):.3f} | {len(ds)} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m macromind.place.evaluate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="*", default=[], help="NAME=PATH (repeatable)")
    ap.add_argument("--synthetic", default="8-20")
    ap.add_argument("--n-designs", type=int, default=2)
    ap.add_argument("--design-seed-base", type=int, default=1000)
    ap.add_argument("--design-seeds", type=int, nargs="*", default=None)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--sa-budgets", type=int, nargs="*", default=[20000])
    ap.add_argument("--sa-alt", type=int, default=0, help="also run cluster-pass-aware SA with this budget")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    ckpts = dict(c.split("=", 1) for c in a.ckpt)
    recs = []
    for d in synthetic_designs(a.synthetic, a.n_designs, a.design_seed_base, a.design_seeds):
        print(d.summary())
        recs += evaluate_design(d, ckpts, k=a.k, sa_budgets=a.sa_budgets, seed=a.seed, sa_alt=a.sa_alt)
    print(markdown_table(recs))
    print()
    print(summary_table(recs))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(recs, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
