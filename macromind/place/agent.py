"""Checkpoints and inference: greedy / best-of-k RL placement."""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import torch

from ..core.design import Design, Placement
from .env import EnvConfig
from .policy import PolicyConfig, PolicyNet
from .ppo import DesignTask, rollout

DEFAULT_CKPT = Path("runs/rl-v1/policy.pt")


def save_checkpoint(path: str | Path, policy: PolicyNet, env_cfg: EnvConfig, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"policy_config": policy.cfg.as_dict(), "state_dict": policy.state_dict(),
                "env": {"grid": env_cfg.grid, "cluster_pass": env_cfg.cluster_pass}, **(extra or {})}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path) -> tuple[PolicyNet, EnvConfig, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    pc = dict(ck["policy_config"])
    pc["dilations"] = tuple(pc["dilations"])
    policy = PolicyNet(PolicyConfig(**pc))
    policy.load_state_dict(ck["state_dict"])
    policy.eval()
    env = EnvConfig(grid=ck["env"]["grid"], cluster_pass=ck["env"]["cluster_pass"])
    return policy, env, ck


def run_policy(policy: PolicyNet, design: Design, env_cfg: EnvConfig | None = None, *, k: int = 0, seed: int = 0,
               reference: float | None = None, task: DesignTask | None = None) -> tuple[Placement, dict]:
    """Greedy episode plus ``k`` sampled ones; returns the best placement and stats."""
    t0 = time.perf_counter()
    task = task or DesignTask(design, env_cfg or EnvConfig(grid=policy.cfg.grid), reference)
    gen = torch.Generator().manual_seed(seed)
    policy.eval()
    cands = []
    g = rollout(policy, task, 1, greedy=True)
    cands.append((g.proxies[0], g.placements[0], "greedy"))
    greedy_proxy = g.proxies[0].proxy if g.proxies[0] is not None else float("inf")
    done = 0
    while done < k:
        n = min(32, k - done)
        s = rollout(policy, task, n, generator=gen)
        cands += [(r, p, "sample") for r, p in zip(s.proxies, s.placements)]
        done += n
    valid = [c for c in cands if c[0] is not None]
    if not valid:
        raise RuntimeError("RL policy produced no legal placement")
    best = min(valid, key=lambda c: c[0].proxy)
    p = best[1].copy()
    p.meta.update(method="rl", seed=seed, evaluations=1 + k, greedy_proxy=greedy_proxy, best_proxy=best[0].proxy,
                  best_from=best[2], runtime_s=time.perf_counter() - t0)
    return p, {"result": best[0], "greedy_proxy": greedy_proxy}


_CACHE: dict[str, tuple] = {}


def rl_place(design: Design, seed: int = 0, budget: int | None = None, checkpoint: str | None = None) -> Placement:
    """Registry entry ``rl``. ``budget`` = number of sampled episodes on top of the greedy one."""
    path = checkpoint or os.environ.get("MACROMIND_RL_CKPT") or str(DEFAULT_CKPT)
    if Path(path).exists():
        if path not in _CACHE:
            _CACHE[path] = load_checkpoint(path)
        policy, env_cfg, _ = _CACHE[path]
    else:
        warnings.warn(f"RL checkpoint {path} not found; using an untrained policy (seed {seed})", stacklevel=2)
        torch.manual_seed(seed)
        policy, env_cfg = PolicyNet(), EnvConfig()
        path = None
    p, _ = run_policy(policy, design, env_cfg, k=int(budget or 0), seed=seed)
    p.meta["checkpoint"] = path
    return p
