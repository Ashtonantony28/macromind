"""Per-design RL fine-tuning on a real floorplan (RL v2), with proxy-only or proxy + learned-predictor reward.

    python -m macromind.place.finetune --def 2_1_floorplan.def --lef tech.lef cells.lef ram.lef --halo 10 \\
        --init runs/rl-v1/policy.pt --reward proxy --steps 60000 --out runs/rl-v2/bp_fe_top/proxy
    python -m macromind.place.finetune ... --reward pred --bundle runs/predict/bundle.pt --design-key bp_fe_top \\
        --pred-weight 0.05 --unc-weight 0.05 --out runs/rl-v2/bp_fe_top/pred

The design is loaded exactly as the ORFS hook loads it (``macromind.flow.score.load_reduced``: DEF + LEF, HEM
clustering seed 0), so the fine-tuned policy sees the same clustered design at inference. The budget is a number of
env steps (macro placements), identical for both reward variants; wall and CPU time are recorded in
``OUT/finetune.json``. The checkpoint gets an ``inference`` dict that :func:`macromind.place.agent.rl_place`
honours: ``k`` sampled episodes, no greedy episode, best-of-k by the *training* reward (for ``pred``: proxy +
lambda * predicted badness + kappa * ensemble std), never by an ORFS result.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import torch


def _cpu_s() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def build_env_cfg(base, reward: str, *, bundle: str | None, design_key: str | None, w_congestion: float, w_wl: float,
                  w_timing: float, pred_weight: float, unc_weight: float):
    from dataclasses import replace

    if reward == "proxy":
        return replace(base, predictor=None, pred_weight=0.0, unc_weight=0.0), None
    from .agent import inference_env

    pr = {"bundle": str(Path(bundle).resolve()), "design_key": design_key, "w_congestion": w_congestion, "w_wl": w_wl,
          "w_timing": w_timing, "pred_weight": pred_weight, "unc_weight": unc_weight}
    return inference_env(base, {"predictor": pr}), pr


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m macromind.place.finetune", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--def", dest="def_path", required=True)
    ap.add_argument("--lef", nargs="+", required=True)
    ap.add_argument("--halo", type=float, default=0.0)
    ap.add_argument("--init", default="runs/rl-v1/policy.pt")
    ap.add_argument("--reward", choices=["proxy", "pred"], required=True)
    ap.add_argument("--bundle", default="runs/predict/bundle.pt")
    ap.add_argument("--design-key", default=None, help="predictor dataset name (None = global statistics)")
    ap.add_argument("--w-congestion", type=float, default=1.0)
    ap.add_argument("--w-wl", type=float, default=1.0)
    ap.add_argument("--w-timing", type=float, default=0.0)
    ap.add_argument("--pred-weight", type=float, default=0.05)
    ap.add_argument("--unc-weight", type=float, default=0.05)
    ap.add_argument("--steps", type=int, required=True, help="env steps (macro placements)")
    ap.add_argument("--time-limit", type=float, default=None, help="safety cap in seconds")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--k", type=int, default=64, help="inference samples written into the checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from ..flow.score import load_reduced
    from .agent import load_checkpoint, save_checkpoint
    from .ppo import PPOConfig
    from .train import train

    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0, c0 = time.time(), _cpu_s()
    _, red, stats = load_reduced(a.def_path, a.lef, halo_um=a.halo)
    _, env0, _ = load_checkpoint(a.init)
    env_cfg, pr = build_env_cfg(env0, a.reward, bundle=a.bundle, design_key=a.design_key, w_congestion=a.w_congestion,
                                w_wl=a.w_wl, w_timing=a.w_timing, pred_weight=a.pred_weight, unc_weight=a.unc_weight)
    ppo = PPOConfig(lr=a.lr, episodes_per_batch=a.episodes, ent_coef=a.ent, epochs=a.epochs)
    train([red], out, steps=a.steps, ppo=ppo, env_cfg=env_cfg, init=a.init, seed=a.seed, threads=a.threads,
          time_limit=a.time_limit, ckpt_every=5)
    ck = torch.load(out / "policy.pt", map_location="cpu", weights_only=False)
    inference = {"k": a.k, "include_greedy": False, "select": "reward", "predictor": pr}
    policy, env_saved, _ = load_checkpoint(out / "policy.pt")
    save_checkpoint(out / "policy.pt", policy, env_saved,
                    {k: v for k, v in ck.items() if k not in ("policy_config", "state_dict", "env")} | {"inference": inference})
    log = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
    rec = {"design": red.name, "def": a.def_path, "reward": a.reward, "init": a.init, "steps_budget": a.steps,
           "steps": log[-1]["steps"] if log else 0, "iters": len(log), "episodes_per_iter": a.episodes,
           "wall_s": round(time.time() - t0, 1), "cpu_s": round(_cpu_s() - c0, 1), "threads": a.threads,
           "clustering": stats, "inference": inference, "ppo": ppo.as_dict(),
           "reward_first": log[0]["reward"] if log else None, "reward_last": log[-1]["reward"] if log else None}
    (out / "finetune.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: rec[k] for k in ("design", "reward", "steps", "iters", "wall_s", "cpu_s", "reward_first",
                                          "reward_last")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
