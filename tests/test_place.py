"""SG6: RL environment, policy, PPO and the rl placer."""

import numpy as np
import pytest
import torch

from macromind.baselines import get_placer, place, random_place
from macromind.core import generate_synthetic
from macromind.legalize import check_legality
from macromind.place import (
    DesignTask, EnvConfig, PlacementEnv, PolicyConfig, PolicyNet, PPOConfig, evaluate_placement, ppo_update, rollout,
)
from macromind.place.policy import MASK_VALUE, obs_to_tensors

torch.set_num_threads(2)


@pytest.fixture(scope="module")
def design():
    return generate_synthetic(10, 100, seed=21)


@pytest.fixture(scope="module")
def env(design):
    return PlacementEnv(design, EnvConfig(grid=16))


def _random_episode(env, rng):
    obs = env.reset()
    n, masks_ok, r, info = 0, True, 0.0, {}
    done = env.done
    while not done:
        valid = np.flatnonzero(obs["mask"])
        a = int(rng.choice(valid))
        if env.valid_mask().any():
            x, y = env.decode(a)
            i = env.current
            halo = env.design.halo
            for k in np.flatnonzero(env.placed):
                if (x < env.x[k] + env.w[k] + halo and x + env.w[i] + halo > env.x[k]
                        and y < env.y[k] + env.h[k] + halo and y + env.h[i] + halo > env.y[k]):
                    masks_ok = False
        obs, r, done, info = env.step(a)
        n += 1
    return n, masks_ok, r, info


def test_env_masks_never_allow_overlap(env):
    rng = np.random.default_rng(0)
    for _ in range(5):
        n, ok, r, info = _random_episode(env, rng)
        assert ok
        assert check_legality(env.design, env.placement()) == [] or env.fallbacks
        assert n == env.n_steps == 10
        assert np.isfinite(r) and "proxy" in info


def test_env_order_and_shapes(env, design):
    obs = env.reset()
    a = design.arrays()
    areas = (a.node_w * a.node_h)[: a.n_macros][env.order]
    assert np.all(np.diff(areas) <= 0)  # largest first
    assert obs["image"].shape == (6, 16, 16) and obs["image"].dtype == np.float32
    assert obs["mask"].shape == (256,) and obs["mask"].all()  # nothing placed yet
    assert np.isfinite(obs["image"]).all()


def test_env_masked_action_rejected(env):
    obs = env.reset()
    obs, *_ = env.step(int(np.flatnonzero(obs["mask"])[len(obs["mask"]) // 2]))
    bad = np.flatnonzero(~obs["mask"])
    assert len(bad) > 0  # the first (largest) macro blocks some cells
    with pytest.raises(ValueError):
        env.step(int(bad[0]))


def test_env_deterministic(design):
    e1 = PlacementEnv(design, EnvConfig(grid=16))
    e2 = PlacementEnv(design, EnvConfig(grid=16), reference=e1.reference)
    r = []
    for e in (e1, e2):
        _, _, rr, _ = _random_episode(e, np.random.default_rng(5))
        r.append((rr, e.placement().x.tolist()))
    assert r[0] == r[1]


def test_reward_matches_protocol(env, design):
    _random_episode(env, np.random.default_rng(1))
    res, _ = evaluate_placement(design, env.final_placement)
    assert res.proxy == pytest.approx(env.result.proxy)


def test_fixed_macros_and_fallback():
    d = generate_synthetic(6, 60, seed=2, n_fixed_macros=2)
    e = PlacementEnv(d, EnvConfig(grid=8))
    assert e.n_steps == 4 and e.placed.sum() == 2
    _random_episode(e, np.random.default_rng(0))
    assert check_legality(d, e.final_placement) == []
    assert e.final_placement.x[0] == d.macros[0].x


@pytest.fixture(scope="module")
def task(design):
    return DesignTask(design, EnvConfig(grid=16))


def test_policy_shapes_and_masking(task):
    torch.manual_seed(0)
    pol = PolicyNet(PolicyConfig(grid=16, hidden=32, conv_ch=16))
    envs = task.get_envs(3)
    obs = [e.reset() for e in envs]
    for e, o in zip(envs, obs):  # place one macro so masks are non-trivial
        e.step(int(np.flatnonzero(o["mask"])[100]))
    obs = [e.observe() for e in envs]
    t = obs_to_tensors(obs)
    logits, v = pol(pol.encode(task.graph), **t)
    assert logits.shape == (3, 256) and v.shape == (3,)
    assert torch.isfinite(logits).all() and torch.isfinite(v).all()
    assert (logits[~t["mask"]] == MASK_VALUE).all()
    p = torch.softmax(logits, -1)
    assert float(p[~t["mask"]].max()) == 0.0
    g = torch.Generator().manual_seed(0)
    s = torch.multinomial(p, 2000, replacement=True, generator=g)
    assert bool(t["mask"].gather(1, s).all())


def test_rollout_and_sampled_actions_valid(task):
    torch.manual_seed(0)
    pol = PolicyNet(PolicyConfig(grid=16, hidden=32, conv_ch=16))
    ro = rollout(pol, task, 4, generator=torch.Generator().manual_seed(0))
    T = task.get_envs(1)[0].n_steps
    assert ro.actions.shape == (4 * T,)
    assert bool(ro.obs["mask"].gather(1, ro.actions[:, None]).all())
    assert np.isfinite(ro.rewards).all() and all(r is not None for r in ro.proxies)
    assert torch.isfinite(ro.adv).all()
    # deterministic with seed
    torch.manual_seed(0)
    pol2 = PolicyNet(PolicyConfig(grid=16, hidden=32, conv_ch=16))
    ro2 = rollout(pol2, task, 4, generator=torch.Generator().manual_seed(0))
    assert torch.equal(ro.actions, ro2.actions)


def test_ppo_update_reduces_loss(task):
    from macromind.place.ppo import ppo_loss

    torch.manual_seed(1)
    pol = PolicyNet(PolicyConfig(grid=16, hidden=32, conv_ch=16))
    cfg = PPOConfig(epochs=1, minibatch=10_000, lr=1e-3)
    ro = rollout(pol, task, 8, generator=torch.Generator().manual_seed(1))
    idx = torch.arange(len(ro.actions))
    adv = (ro.adv - ro.adv.mean()) / (ro.adv.std() + 1e-8)
    before, _ = ppo_loss(pol, ro, idx, cfg, adv)
    opt = torch.optim.Adam(pol.parameters(), lr=cfg.lr)
    for _ in range(10):
        st = ppo_update(pol, opt, [ro], cfg)
    after, _ = ppo_loss(pol, ro, idx, cfg, adv)
    assert after.item() < before.item()
    assert np.isfinite(st["loss"]) and st["grad_norm"] >= 0


def test_rl_placer_registered(design, tmp_path, monkeypatch):
    from macromind.place import save_checkpoint

    torch.manual_seed(0)
    pol = PolicyNet(PolicyConfig(grid=16, hidden=32, conv_ch=16))
    ck = tmp_path / "p.pt"
    save_checkpoint(ck, pol, EnvConfig(grid=16))
    monkeypatch.setenv("MACROMIND_RL_CKPT", str(ck))
    p = get_placer("rl")(design, seed=0, budget=4)
    assert check_legality(design, p) == []
    assert p.meta["method"] == "rl" and p.meta["evaluations"] == 5 and p.meta["checkpoint"] == str(ck)
    q = place(design, "rl", seed=0, budget=4, cluster_pass=True)
    assert np.array_equal(p.x, q.x) and "cluster_design" in q.meta


def test_train_smoke_cli(tmp_path):
    from macromind.place.train import main

    assert main(["--synthetic", "4", "--n-designs", "1", "--steps", "40", "--grid", "8", "--episodes", "4",
                 "--epochs", "1", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "policy.pt").exists() and (tmp_path / "log.jsonl").read_text().strip()


@pytest.mark.slow
def test_training_beats_random(tmp_path):
    from macromind.place import run_policy
    from macromind.place.train import train

    d = generate_synthetic(8, 80, seed=31)
    pol = train([d], tmp_path, steps=8 * 32 * 40, ppo=PPOConfig(episodes_per_batch=32), env_cfg=EnvConfig(grid=16),
                policy_cfg=PolicyConfig(grid=16), seed=0, time_limit=240, log=lambda *_: None)
    _, info = run_policy(pol, d, EnvConfig(grid=16))
    rnd = [evaluate_placement(d, random_place(d, seed=s))[0].proxy for s in range(10)]
    assert info["greedy_proxy"] < np.mean(rnd), (info["greedy_proxy"], rnd)


def test_evaluate_design_table(tmp_path):
    from macromind.place import save_checkpoint
    from macromind.place.evaluate import evaluate_design, markdown_table, summary_table, wiremask_place

    d = generate_synthetic(5, 50, seed=41)
    assert check_legality(d, wiremask_place(d)) == []
    torch.manual_seed(0)
    ck = tmp_path / "p.pt"
    save_checkpoint(ck, PolicyNet(PolicyConfig(grid=8, hidden=16, conv_ch=16)), EnvConfig(grid=8))
    recs = evaluate_design(d, {"t": str(ck)}, k=2, sa_budgets=(50,), log=lambda *_: None)
    methods = [r["method"] for r in recs]
    assert methods == ["random@2", "grid", "wiremask", "sa@50", "rl:t", "rl:t@2"]
    assert all(r["legal"] and np.isfinite(r["proxy"]) and np.isfinite(r["proxy_nocp"]) for r in recs)
    assert "rl:t@2" in markdown_table(recs) and "proxy / grid" in summary_table(recs)


def test_run_policy_sample_only_and_reward_selection(design):
    from macromind.place.agent import run_policy

    pol = PolicyNet(PolicyConfig(grid=16))
    cfg = EnvConfig(grid=16)
    p0, _ = run_policy(pol, design, cfg, k=6, seed=0, include_greedy=False, select="reward")
    p1, _ = run_policy(pol, design, cfg, k=6, seed=1, include_greedy=False, select="reward")
    assert p0.meta["evaluations"] == 6 and p0.meta["best_from"] == "sample" and p0.meta["greedy_proxy"] is None
    assert not (np.array_equal(p0.x, p1.x) and np.array_equal(p0.y, p1.y))  # seeds give different samples
    assert not check_legality(design, p0, check_grid=False)
    # proxy-only reward: best reward == lowest proxy
    pp, _ = run_policy(pol, design, cfg, k=6, seed=0, include_greedy=False, select="proxy")
    assert pp.meta["best_proxy"] == pytest.approx(p0.meta["best_proxy"])
    with pytest.raises(ValueError):
        run_policy(pol, design, cfg, k=1, select="nope")
