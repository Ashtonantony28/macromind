"""SG10: per-design RL fine-tuning (macromind.place.finetune), checkpoint inference settings, and the SG10 report."""

from __future__ import annotations

import json
import math

import pytest
import torch

from macromind.place import EnvConfig
from macromind.place.agent import inference_env, load_checkpoint, rl_place, save_checkpoint
from macromind.place.policy import PolicyConfig, PolicyNet
from macromind.report import sg10


@pytest.fixture(scope="module")
def tiny_bundle(tmp_path_factory):
    from macromind.predict.models import CongestionNet, Ensemble, TimingNet
    from macromind.predict.train import Normalizer, save_bundle

    torch.manual_seed(0)
    norm = Normalizer(torch.zeros(2), torch.ones(2))
    path = tmp_path_factory.mktemp("bundle") / "bundle.pt"
    save_bundle(path, {"congestion": (Ensemble([CongestionNet(), CongestionNet()]), norm),
                       "timing": (Ensemble([TimingNet(), TimingNet()]), norm)})
    return path


@pytest.fixture(scope="module")
def init_ckpt(tmp_path_factory):
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("init") / "policy.pt"
    save_checkpoint(path, PolicyNet(PolicyConfig(grid=16)), EnvConfig(grid=16))
    return path


def test_build_env_cfg_proxy_and_pred(tiny_bundle):
    from macromind.place.finetune import build_env_cfg

    base = EnvConfig(grid=16, pred_weight=0.3)
    cfg, pr = build_env_cfg(base, "proxy", bundle=None, design_key=None, w_congestion=1, w_wl=1, w_timing=0,
                            pred_weight=0.05, unc_weight=0.1)
    assert pr is None and cfg.predictor is None and cfg.pred_weight == 0.0 and cfg.unc_weight == 0.0
    cfg, pr = build_env_cfg(base, "pred", bundle=str(tiny_bundle), design_key=None, w_congestion=1, w_wl=1,
                            w_timing=0, pred_weight=0.05, unc_weight=0.1)
    assert cfg.predictor is not None and cfg.pred_weight == 0.05 and cfg.unc_weight == 0.1
    assert pr["w_timing"] == 0 and pr["bundle"] == str(tiny_bundle.resolve())
    assert cfg.predictor.w == {"congestion": 1, "timing": 0, "wl": 1}


def test_inference_env_passthrough_and_attach(tiny_bundle):
    base = EnvConfig(grid=16)
    assert inference_env(base, None) is base and inference_env(base, {"k": 4}) is base
    cfg = inference_env(base, {"predictor": {"bundle": str(tiny_bundle), "pred_weight": 0.2, "unc_weight": 0.0}})
    assert cfg.predictor is not None and cfg.pred_weight == 0.2 and base.predictor is None


@pytest.mark.parametrize("reward", ["proxy", "pred"])
def test_finetune_cli_end_to_end(fixtures, tmp_path, init_ckpt, tiny_bundle, reward):
    from macromind.place.finetune import main

    out = tmp_path / reward
    rc = main(["--def", str(fixtures / "tiny.def"), "--lef", str(fixtures / "tiny.lef"), "--init", str(init_ckpt),
               "--reward", reward, "--bundle", str(tiny_bundle), "--steps", "8", "--episodes", "2", "--epochs", "1",
               "--k", "3", "--threads", "1", "--out", str(out)])
    assert rc == 0
    rec = json.loads((out / "finetune.json").read_text())
    assert rec["reward"] == reward and rec["steps"] >= 8 and rec["iters"] >= 1 and rec["cpu_s"] > 0
    inf = rec["inference"]
    assert inf["k"] == 3 and inf["include_greedy"] is False and inf["select"] == "reward"
    assert (inf["predictor"] is None) == (reward == "proxy")
    _, _, ck = load_checkpoint(out / "policy.pt")
    assert ck["inference"] == inf  # the checkpoint carries its own decoding / selection rule

    # rl_place honours it: k sampled episodes, no greedy, best by training reward
    from macromind.io.defio import read_def

    d = read_def(fixtures / "tiny.def", fixtures / "tiny.lef")
    p = rl_place(d, seed=0, checkpoint=str(out / "policy.pt"))
    assert p.meta["evaluations"] == 3 and p.meta["select"] == "reward" and p.meta["greedy_proxy"] is None
    assert math.isfinite(p.meta["best_reward"])


def test_welch_and_rank_helpers():
    d, hw, p = sg10.welch([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert d == 0 and hw > 0 and p == pytest.approx(1.0)
    d, hw, p = sg10.welch([10.0, 10.1, 9.9], [1.0, 1.1, 0.9])
    assert d == pytest.approx(9.0) and p < 1e-3 and 0 < hw < 1
    assert sg10.welch([1.0], [1.0, 2.0]) is None
    assert sg10.welch([1.0, 1.0], [2.0, 2.0])[1:] == (0.0, None)
    assert sg10.welch([0.1 + 0.2, 0.3], [0.5, 0.5])[2] is None  # float round-off is not spread
    assert sg10.kendall([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert sg10.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert sg10.kendall([1, 2], [1, 2]) is None


def _rec(tag, seed, wl, hot, proxy, placer="macromind-rl", status="ok", run_dir=None):
    return {"job": {"design": "nangate45/bp_fe_top", "placer": placer, "seed": seed, "stop_after": "grt", "tag": tag},
            "status": status, "run_dir": run_dir or f"/x/{tag or placer}-{seed}", "make_wall_s": 600.0,
            "ppa": {"grt_wl_um": wl, "wns_ns": -0.01 * seed, "tns_ns": -0.1 * seed, "power_w": 0.18},
            "grt_congestion": {"hot_frac": hot},
            "placement": {"proxy": proxy, "wirelength": proxy / 2, "congestion": proxy / 3,
                          "macros": [{"name": "m0", "x": float(seed), "y": float(len(tag or placer)), "orient": "R0"}]},
            "placer": {"total_s": 5.0}}


def test_sg10_tables():
    groups = {"rtlmp": [_rec(None, 0, 1.0e6, 0.02, 1.5, placer="rtlmp")],
              "rl-v2-proxy": [_rec("rl-v2-proxy", s, 1.1e6 + 1e4 * s, 0.03, 1.4 + 0.01 * s) for s in range(3)],
              "rl-v2-pred": [_rec("rl-v2-pred", s, 1.0e6 + 1e4 * s, 0.02, 1.45) for s in range(3)]}
    main = "\n".join(sg10.main_table(groups, 1.0e6))
    assert "| rtlmp | 1/1 | 0 | 1 |" in main and "| rl-v2-pred (best-of-64) | 3/3 | 0,1,2 | 3 |" in main
    abl = "\n".join(sg10.ablation(groups))
    assert "rl-v2-pred vs rl-v2-proxy | 3 (3), 3 (3) |" in abl and "p=" in abl
    recs = [r for rs in groups.values() for r in rs]
    corr = "\n".join(sg10.correlations(recs, {}))
    assert "Over 7 distinct GRT placements" in corr and "(n=7)" in corr
    per = "\n".join(sg10.per_run_table(groups, {}))
    assert per.count("\n") == 1 + 7


def test_sg10_identical_placements_are_not_independent():
    same = [_rec("rl-v2-pred", s, 1.0e6, 0.02, 1.3) for s in range(3)]
    for r in same:
        r["placement"]["macros"][0]["x"] = 0.0  # mode-collapsed policy: every seed gives the same placement
    other = [_rec("rl-v2-proxy", s, 1.1e6, 0.01, 1.2) for s in range(3)]
    for r in other:
        r["placement"]["macros"][0]["x"] = 0.0
    assert sg10.n_distinct(same) == 1
    abl = "\n".join(sg10.ablation({"rl-v2-pred": same, "rl-v2-proxy": other}))
    assert "3 (1), 3 (1)" in abl and "untestable" in abl and "p=0.00" not in abl
    assert "Over 2 distinct GRT placements" in "\n".join(sg10.correlations(same + other, {}))
