"""Train the RL macro placer with PPO.

Examples::

    python -m macromind.place.train --smoke --out runs/smoke
    python -m macromind.place.train --synthetic 8-20 --n-designs 6 --steps 200000 --out runs/rl-v1
    python -m macromind.place.train --synthetic 12 --design-seeds 1001 --init runs/rl-v1/policy.pt \\
        --steps 20000 --out runs/rl-v1-ft     # fine-tune on one design

Synthetic design ``i`` has ``n_macros`` drawn from the ``--synthetic`` range,
``10 * n_macros`` clusters and generator seed ``--design-seeds`` (or
``--design-seed-base + i``). Logs go to ``OUT/log.jsonl``, the checkpoint to
``OUT/policy.pt`` (every ``--ckpt-every`` iterations and at the end).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ..core.synthetic import generate_synthetic
from .agent import load_checkpoint, save_checkpoint
from .env import EnvConfig
from .policy import PolicyConfig, PolicyNet
from .ppo import DesignTask, PPOConfig, ppo_update, rollout


def parse_range(s: str) -> tuple[int, int]:
    if "-" in s:
        a, b = s.split("-")
        return int(a), int(b)
    return int(s), int(s)


def synthetic_designs(rng_spec: str, n: int, seed_base: int = 0, seeds: list[int] | None = None):
    lo, hi = parse_range(rng_spec)
    seeds = seeds or [seed_base + i for i in range(n)]
    out = []
    for s in seeds:
        nm = int(np.random.default_rng(s).integers(lo, hi + 1))
        out.append(generate_synthetic(nm, 10 * nm, seed=s))
    return out


def train(designs, out: str | Path, *, steps: int, ppo: PPOConfig | None = None, policy_cfg: PolicyConfig | None = None,
          env_cfg: EnvConfig | None = None, init: str | None = None, seed: int = 0, threads: int = 2,
          ckpt_every: int = 10, time_limit: float | None = None, log=print) -> PolicyNet:
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    ppo = ppo or PPOConfig()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if init:
        policy, env_loaded, _ = load_checkpoint(init)
        env_cfg = env_cfg or env_loaded
    else:
        env_cfg = env_cfg or EnvConfig()
        policy = PolicyNet(policy_cfg or PolicyConfig(grid=env_cfg.grid))
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=ppo.lr)
    gen = torch.Generator().manual_seed(seed)
    t0 = time.perf_counter()
    tasks = [DesignTask(d, env_cfg) for d in designs]
    log(f"{len(tasks)} designs ready in {time.perf_counter() - t0:.1f}s: " + ", ".join(f"{t.design.name}(ref={t.reference:.4f})" for t in tasks))
    best = {t.design.name: float("inf") for t in tasks}
    total = 0
    it = 0
    logf = open(out / "log.jsonl", "a")
    meta = {"designs": [t.design.name for t in tasks], "ppo": ppo.as_dict(), "seed": seed}
    while total < steps and (time_limit is None or time.perf_counter() - t0 < time_limit):
        it += 1
        rolls = []
        rec = {"iter": it}
        for t in tasks:
            ro = rollout(policy, t, ppo.episodes_per_batch, cfg=ppo, generator=gen)
            rolls.append(ro)
            total += len(ro.actions)
            px = [r.proxy for r in ro.proxies if r is not None]
            if px:
                best[t.design.name] = min(best[t.design.name], min(px))
            rec[t.design.name] = {"reward": float(ro.rewards.mean()), "proxy_mean": float(np.mean(px)) if px else None,
                                  "proxy_best": best[t.design.name]}
        stats = ppo_update(policy, opt, rolls, ppo, gen)
        rec.update(stats, steps=total, time=time.perf_counter() - t0,
                   reward=float(np.mean([r.rewards.mean() for r in rolls])))
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        log(f"it {it:4d} steps {total:7d} t {rec['time']:6.0f}s reward {rec['reward']:+.4f} "
            f"ent {stats['entropy']:.3f} kl {stats['kl']:+.4f} vf {stats['vf']:.4f}")
        if it % ckpt_every == 0:
            save_checkpoint(out / "policy.pt", policy, env_cfg, {"meta": meta, "steps": total})
    save_checkpoint(out / "policy.pt", policy, env_cfg, {"meta": meta, "steps": total, "best": best})
    logf.close()
    return policy


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m macromind.place.train", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", default="8-20", help="macro-count range, e.g. 8-20 or 12")
    ap.add_argument("--n-designs", type=int, default=4)
    ap.add_argument("--design-seed-base", type=int, default=0)
    ap.add_argument("--design-seeds", type=int, nargs="*", default=None)
    ap.add_argument("--steps", type=int, default=100_000, help="total env steps (macro placements)")
    ap.add_argument("--time-limit", type=float, default=None, help="seconds")
    ap.add_argument("--out", default="runs/rl")
    ap.add_argument("--init", default=None, help="checkpoint to fine-tune")
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--episodes", type=int, default=16, help="episodes per design per iteration")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--smoke", action="store_true", help="tiny run (< 2 min): 2 small designs, few iterations")
    a = ap.parse_args(argv)
    if a.smoke:
        a.synthetic, a.n_designs, a.steps, a.grid, a.episodes, a.epochs = "5-6", 2, 600, 16, 8, 2
        a.time_limit = a.time_limit or 90
    designs = synthetic_designs(a.synthetic, a.n_designs, a.design_seed_base, a.design_seeds)
    env_cfg = EnvConfig(grid=a.grid)
    ppo = PPOConfig(lr=a.lr, episodes_per_batch=a.episodes, ent_coef=a.ent, epochs=a.epochs)
    t = time.perf_counter()
    train(designs, a.out, steps=a.steps, ppo=ppo, env_cfg=None if a.init else env_cfg,
          policy_cfg=PolicyConfig(grid=a.grid), init=a.init, seed=a.seed, threads=a.threads,
          ckpt_every=a.ckpt_every, time_limit=a.time_limit)
    print(f"done in {time.perf_counter() - t:.1f}s -> {Path(a.out) / 'policy.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
