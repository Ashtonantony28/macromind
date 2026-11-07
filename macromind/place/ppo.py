"""PPO with GAE for the macro-placement env (CPU, batched lockstep episodes).

All episodes of one design have the same length (#movable macros), so a
rollout runs ``B`` envs of one design in lockstep and the graph encoding is
computed once per policy call. Rewards are terminal only; advantages use GAE
with ``gamma = 1`` and are normalised per design batch, which makes the
signal "better or worse than this design's batch" and keeps multi-design
training balanced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.distributions import Categorical

from ..core.design import Design
from ..graph.tensors import GraphTensors, build_graph
from .env import EnvConfig, PlacementEnv, reference_proxy
from .policy import PolicyNet, obs_to_tensors


@dataclass
class PPOConfig:
    lr: float = 3e-4
    episodes_per_batch: int = 16
    epochs: int = 4
    minibatch: int = 256
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 1.0
    lam: float = 0.95

    def as_dict(self) -> dict:
        return asdict(self)


class DesignTask:
    """A design with its graph tensors, reward reference and a pool of envs."""

    def __init__(self, design: Design, env_cfg: EnvConfig | None = None, reference: float | None = None):
        self.design = design
        self.env_cfg = env_cfg or EnvConfig()
        self.graph: GraphTensors = build_graph(design)
        self.reference = float(reference) if reference is not None else reference_proxy(design, self.env_cfg)
        self.envs: list[PlacementEnv] = []

    def get_envs(self, n: int) -> list[PlacementEnv]:
        while len(self.envs) < n:
            self.envs.append(PlacementEnv(self.design, self.env_cfg, self.reference))
        return self.envs[:n]


@dataclass
class Rollout:
    task: DesignTask
    obs: dict[str, torch.Tensor]  # each (T*B, ...)
    actions: torch.Tensor
    logp: torch.Tensor
    values: torch.Tensor
    adv: torch.Tensor
    returns: torch.Tensor
    rewards: np.ndarray  # (B,) terminal rewards
    proxies: list  # ProxyResult | None per episode
    placements: list


@torch.no_grad()
def rollout(policy: PolicyNet, task: DesignTask, n: int, *, greedy: bool = False, cfg: PPOConfig | None = None,
            generator: torch.Generator | None = None) -> Rollout:
    cfg = cfg or PPOConfig()
    envs = task.get_envs(n)
    obs = [e.reset() for e in envs]
    T = envs[0].n_steps
    enc = policy.encode(task.graph)
    store: list[dict] = []
    acts, logps, vals = [], [], []
    rewards = np.zeros(n)
    for t in range(T):
        tens = obs_to_tensors(obs)
        logits, v = policy(enc, **tens)
        if greedy:
            a = logits.argmax(-1)
        else:
            probs = torch.softmax(logits, -1)
            a = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        lp = torch.log_softmax(logits, -1).gather(1, a[:, None]).squeeze(-1)
        store.append(tens)
        acts.append(a)
        logps.append(lp)
        vals.append(v)
        new_obs = []
        for k, (e, ak) in enumerate(zip(envs, a.tolist())):
            o, r, done, _ = e.step(ak)
            new_obs.append(o)
            if done:
                rewards[k] = r
        obs = new_obs
    V = torch.stack(vals) if T else torch.zeros(0, n)  # (T, B)
    R = torch.zeros(T, n)
    if T:
        R[-1] = torch.as_tensor(rewards, dtype=torch.float32)
    adv = torch.zeros(T, n)
    last = torch.zeros(n)
    for t in reversed(range(T)):
        nv = V[t + 1] if t + 1 < T else torch.zeros(n)
        delta = R[t] + cfg.gamma * nv - V[t]
        last = delta + cfg.gamma * cfg.lam * last
        adv[t] = last
    ret = adv + V
    obs_cat = {k: torch.cat([s[k] for s in store]) for k in store[0]} if store else {}
    return Rollout(
        task, obs_cat, torch.cat(acts) if acts else torch.zeros(0, dtype=torch.long),
        torch.cat(logps) if logps else torch.zeros(0), V.reshape(-1), adv.reshape(-1), ret.reshape(-1), rewards,
        [e.result for e in envs], [getattr(e, "final_placement", None) for e in envs],
    )


def ppo_loss(policy: PolicyNet, ro: Rollout, idx: torch.Tensor, cfg: PPOConfig, adv: torch.Tensor):
    enc = policy.encode(ro.task.graph)
    obs = {k: v[idx] for k, v in ro.obs.items()}
    logits, v = policy(enc, **obs)
    logp_all = torch.log_softmax(logits, -1)
    lp = logp_all.gather(1, ro.actions[idx][:, None]).squeeze(-1)
    ratio = torch.exp(lp - ro.logp[idx])
    a = adv[idx]
    pg = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
    vf = 0.5 * ((v - ro.returns[idx]) ** 2).mean()
    p = logp_all.exp()
    ent = -(p * torch.where(obs["mask"], logp_all, torch.zeros_like(logp_all))).sum(-1).mean()
    loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent
    with torch.no_grad():
        kl = (ro.logp[idx] - lp).mean().item()
        cf = ((ratio - 1).abs() > cfg.clip).float().mean().item()
    return loss, {"pg": pg.item(), "vf": vf.item(), "entropy": ent.item(), "kl": kl, "clipfrac": cf}


def ppo_update(policy: PolicyNet, opt: torch.optim.Optimizer, rollouts: list[Rollout], cfg: PPOConfig,
               generator: torch.Generator | None = None) -> dict:
    stats: dict[str, list] = {}
    norm_adv = []
    for ro in rollouts:
        a = ro.adv
        norm_adv.append((a - a.mean()) / (a.std() + 1e-8) if len(a) > 1 else a)
    for _ in range(cfg.epochs):
        for ro, adv in zip(rollouts, norm_adv):
            N = len(ro.actions)
            perm = torch.randperm(N, generator=generator)
            for s in range(0, N, cfg.minibatch):
                idx = perm[s : s + cfg.minibatch]
                loss, st = ppo_loss(policy, ro, idx, cfg, adv)
                opt.zero_grad()
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                opt.step()
                st["loss"] = loss.item()
                st["grad_norm"] = float(gn)
                for k, v in st.items():
                    stats.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in stats.items()}
