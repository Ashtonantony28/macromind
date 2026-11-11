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
               reference: float | None = None, task: DesignTask | None = None, include_greedy: bool = True,
               select: str = "proxy") -> tuple[Placement, dict]:
    """Greedy episode (unless ``include_greedy`` is False and ``k > 0``) plus ``k`` sampled ones; returns the best
    placement and stats. ``select="proxy"`` keeps the lowest proxy; ``select="reward"`` keeps the highest training
    reward of ``env_cfg`` (proxy plus any predictor terms), i.e. the policy's own objective."""
    if select not in ("proxy", "reward"):
        raise ValueError(f"select must be 'proxy' or 'reward', not {select!r}")
    t0 = time.perf_counter()
    task = task or DesignTask(design, env_cfg or EnvConfig(grid=policy.cfg.grid), reference)
    gen = torch.Generator().manual_seed(seed)
    policy.eval()
    cands = []  # (proxy result, placement, origin, reward)
    greedy_proxy = None
    if include_greedy or k <= 0:
        g = rollout(policy, task, 1, greedy=True)
        cands.append((g.proxies[0], g.placements[0], "greedy", float(g.rewards[0])))
        greedy_proxy = g.proxies[0].proxy if g.proxies[0] is not None else float("inf")
    done = 0
    while done < k:
        n = min(32, k - done)
        s = rollout(policy, task, n, generator=gen)
        cands += [(r, p, "sample", float(rw)) for r, p, rw in zip(s.proxies, s.placements, s.rewards)]
        done += n
    valid = [c for c in cands if c[0] is not None]
    if not valid:
        raise RuntimeError("RL policy produced no legal placement")
    best = min(valid, key=lambda c: c[0].proxy) if select == "proxy" else max(valid, key=lambda c: c[3])
    p = best[1].copy()
    p.meta.update(method="rl", seed=seed, evaluations=len(cands), greedy_proxy=greedy_proxy, best_proxy=best[0].proxy,
                  best_reward=best[3], best_from=best[2], select=select, runtime_s=time.perf_counter() - t0,
                  distinct_samples=len({(tuple(c[1].x), tuple(c[1].y)) for c in valid}))
    return p, {"result": best[0], "greedy_proxy": greedy_proxy, "reward": best[3]}


def inference_env(env_cfg: EnvConfig, inference: dict | None) -> EnvConfig:
    """The checkpoint's training reward for inference: attach the learned predictor it was fine-tuned with."""
    pr = (inference or {}).get("predictor")
    if not pr:
        return env_cfg
    from dataclasses import replace

    from ..predict.runtime import Predictor

    pred = Predictor.load(pr["bundle"], w_congestion=pr.get("w_congestion", 1.0), w_timing=pr.get("w_timing", 0.0),
                          w_wl=pr.get("w_wl", 0.0), design_key=pr.get("design_key"))
    return replace(env_cfg, predictor=pred, pred_weight=pr.get("pred_weight", 0.0), unc_weight=pr.get("unc_weight", 0.0))


_CACHE: dict[str, tuple] = {}


def rl_place(design: Design, seed: int = 0, budget: int | None = None, checkpoint: str | None = None) -> Placement:
    """Registry entry ``rl``. ``budget`` = number of sampled episodes on top of the greedy one."""
    path = checkpoint or os.environ.get("MACROMIND_RL_CKPT") or str(DEFAULT_CKPT)
    if Path(path).exists():
        if path not in _CACHE:
            _CACHE[path] = load_checkpoint(path)
        policy, env_cfg, ck = _CACHE[path]
        inf = ck.get("inference") or {}
    else:
        warnings.warn(f"RL checkpoint {path} not found; using an untrained policy (seed {seed})", stacklevel=2)
        torch.manual_seed(seed)
        policy, env_cfg = PolicyNet(), EnvConfig()
        path, inf = None, {}
    # checkpoints may carry an "inference" dict (written by macromind.place.finetune): sample-only decoding,
    # best-of-k by the training reward (incl. the learned-predictor terms the policy was fine-tuned with)
    k = int(budget) if budget is not None else int(inf.get("k", 0))
    p, _ = run_policy(policy, design, inference_env(env_cfg, inf), k=k, seed=seed,
                      include_greedy=inf.get("include_greedy", True), select=inf.get("select", "proxy"))
    p.meta["checkpoint"] = path
    return p
