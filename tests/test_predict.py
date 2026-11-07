"""SG8: label parsers, dataset assembly, predictor models, ensembles, RL reward hook, datagen plumbing."""

import json
import math

import numpy as np
import pytest
import torch

from macromind.baselines import grid_place, random_place
from macromind.core import generate_synthetic
from macromind.legalize import check_legality
from macromind.predict.datagen import gcell_arrays, make_plan, perturb, spec_id
from macromind.predict.dataset import Sample
from macromind.predict.features import IMG_CHANNELS, NODE_FEATURES, GraphCache, featurize
from macromind.predict.labels import congestion_maps, congestion_summary, endpoint_instance, parse_endpoint_slack
from macromind.predict.models import CongestionNet, Ensemble, TimingNet, gather_img, scatter_grid
from macromind.predict.train import (
    Normalizer, TrainConfig, kendall_tau, predict, ranking_table, scalar_targets, spearman, train_ensemble, train_model,
)

torch.set_num_threads(2)


# ----------------------------------------------------------------------------- label parsers (real reports)
def test_parse_endpoint_slack_real_report(fixtures):
    text = (fixtures / "predict" / "endpoint_slack_tinyRocket.rpt").read_text()
    eps = parse_endpoint_slack(text)
    data_lines = [ln for ln in text.splitlines() if ln.rstrip().endswith(("(MET)", "(VIOLATED)"))]
    assert len(eps) == len(data_lines) > 10
    worst = min(eps.values())
    assert worst < 0  # tinyRocket does not close timing at GRT
    # every slack equals required - actual (to the report's 4 digits)
    for ln in data_lines[:20]:
        parts = ln.split()
        req, act, slack = float(parts[-4]), float(parts[-3]), float(parts[-2])
        assert abs((req - act) - slack) < 2e-4
        assert eps[parts[0]] <= slack + 1e-12


def test_parse_endpoint_slack_groups_and_wrapping():
    text = """max_delay/setup group clk

                                      Required    Actual
Endpoint                                 Delay     Delay     Slack
------------------------------------------------------------------
a/b/very_long_register_name_that_overflows_the_column[3]$_DFF_P_/D (DFF_X1)    0.4949    0.5267   -0.0318 (VIOLATED)
out[1] (output)                         0.3680    0.2493    0.1187 (MET)

max_delay/setup group other

                                      Required    Actual
Endpoint                                 Delay     Delay     Slack
------------------------------------------------------------------
out[1] (output)                         0.3680    0.4000   -0.0320 (VIOLATED)
"""
    eps = parse_endpoint_slack(text)
    assert eps == {"a/b/very_long_register_name_that_overflows_the_column[3]$_DFF_P_/D": -0.0318, "out[1]": -0.032}
    assert endpoint_instance("a/b/reg[3]$_DFF_P_/D") == ("a/b/reg[3]$_DFF_P_", False)
    assert endpoint_instance("out[1]") == ("out[1]", True)


def test_gcell_maps_from_real_odb_dump(fixtures):
    d = json.loads((fixtures / "predict" / "gcells_bp_fe_crop.json").read_text())
    g = gcell_arrays(d)
    ny, nx = len(d["grid_y"]), len(d["grid_x"])
    assert g["cap_h"].shape == g["use_v"].shape == (ny, nx)
    # metal3 is the only horizontal layer in the crop
    assert np.array_equal(g["cap_h"], np.asarray(d["layers"]["metal3"]["capacity"], np.float32))
    assert np.array_equal(g["use_v"], np.asarray(d["layers"]["metal2"]["usage"], np.float32) +
                          np.asarray(d["layers"]["metal4"]["usage"], np.float32))
    m = congestion_maps(g, 4, 4)
    assert m["util_h"].shape == m["use_v"].shape == (4, 4)
    assert abs(float(m["use_h"].sum()) - float(g["use_h"].sum()) / (float(g["cap_h"].sum()) / 16)) < 1e-3
    # pooled utilisation = total usage / total capacity
    tot = float(g["use_h"].sum() / g["cap_h"].sum())
    assert abs(float(np.average(m["util_h"])) - tot) < 0.15
    assert 0 <= m["util_v"].min() and m["util_v"].max() <= 1.0 + 1e-6
    s = congestion_summary(g)
    assert 0 <= s["hot_frac_h"] <= 1 and s["util_top_h"] >= s["util_mean_h"] and s["overflow"] == 0


def test_congestion_summary_overflow():
    cap = np.full((4, 4), 10.0, np.float32)
    use = cap.copy() * 0.5
    use[0, 0] = 13
    g = {"cap_h": cap, "cap_v": cap, "use_h": use, "use_v": use * 0.1, "grid_x": np.arange(4) * 10,
         "grid_y": np.arange(4) * 10, "die": np.array([0, 0, 40, 40])}
    s = congestion_summary(g)
    assert s["overflow_h"] == 3 and s["overflow_v"] == 0 and abs(s["hot_frac_h"] - 1 / 16) < 1e-9


# ----------------------------------------------------------------------------- synthetic dataset
@pytest.fixture(scope="module")
def design():
    return generate_synthetic(6, 60, seed=5)


def _fake_sample(design, seed, G=16, graph=None):
    """A Sample with labels derived from the features (so a model can fit them)."""
    p = random_place(design, seed=seed)
    f = featurize(design, p, G=G, graph=graph)
    cong = torch.stack([f.img[2], f.img[3], f.img[2], f.img[3]]).clamp(0, 2) * 0.5  # "true" util/usage = scaled RUDY
    n = f.node_x.shape[0]
    ns = torch.full((n,), float("nan"))
    ns[f.n_macros:f.n_macros + 10] = f.pos[f.n_macros:f.n_macros + 10, 0] - 0.5
    sc = {"grt_wl_um": f.hpwl_um * (1.1 + 0.1 * f.proxy["density"]), "hpwl_um": f.hpwl_um, "hot_frac": float(cong.mean()),
          "wns_ns": -0.1 * f.proxy["wirelength"], "tns_ns": -1.0 * f.proxy["wirelength"], "clock_period_ns": 1.0,
          "proxy": f.proxy["proxy"], "proxy_wirelength": f.proxy["wirelength"], "proxy_density": f.proxy["density"],
          "proxy_congestion": f.proxy["congestion"]}
    return Sample("synth", f"s{seed}", "test", "", f, cong, sc, ns, {})


@pytest.fixture(scope="module")
def samples(design):
    g = GraphCache(design)
    return [_fake_sample(design, k, graph=g) for k in range(4)]


def test_features_shapes(design, samples):
    f = samples[0].feats
    G = 16
    assert f.img.shape == (len(IMG_CHANNELS), G, G)
    assert f.node_x.shape == (design.arrays().n_nodes, len(NODE_FEATURES))
    assert f.pos.min() >= 0 and f.pos.max() <= 1
    assert torch.isfinite(f.img).all() and torch.isfinite(f.node_x).all()
    assert f.img[0].max() <= 1.0 + 1e-5  # macro occupancy is a fraction
    # mirroring twice is the identity
    ff = f.flipped(True, True).flipped(True, True)
    assert torch.allclose(ff.img, f.img) and torch.allclose(ff.pos, f.pos)
    # mirroring moves macro occupancy accordingly
    assert torch.allclose(f.flipped(True, False).img[0], f.img[0].flip(-1))


def test_gather_scatter():
    img = torch.arange(16.0).reshape(1, 4, 4)
    pos = torch.tensor([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]])
    assert gather_img(img, pos)[:, 0].tolist() == [0.0, 3.0, 12.0]
    h = torch.tensor([[1.0], [3.0], [5.0]])
    g = scatter_grid(h, torch.tensor([[0.1, 0.1], [0.12, 0.1], [0.9, 0.9]]), 4)
    assert g.shape == (1, 4, 4) and g[0, 0, 0] == 2.0 and g[0, 3, 3] == 5.0 and g.sum() == 7.0


def test_model_shapes(samples):
    f = samples[0].feats
    m, s = CongestionNet()(f)
    assert m.shape == (4, 16, 16) and s.shape == (2,)
    node, g = TimingNet()(f)
    assert node.shape == (f.node_x.shape[0],) and g.shape == (2,)


def test_scalar_targets_and_normalizer(samples):
    y = torch.stack([scalar_targets(s, "congestion") for s in samples])
    assert torch.allclose(y[:, 0], torch.tensor([math.log(s.scalars["grt_wl_um"] / s.scalars["hpwl_um"]) for s in samples]))
    yt = torch.stack([scalar_targets(s, "timing") for s in samples])
    assert (yt[:, 1] <= 0).all()
    n = Normalizer.fit(y)
    assert torch.allclose(n.inv(n(y)), y, atol=1e-6)
    assert torch.allclose(n(y).mean(0), torch.zeros(2), atol=1e-5)


def test_tiny_overfit(samples):
    """Both models drive the training loss down on 2 samples."""
    tr = samples[:2]
    for kind in ("congestion", "timing"):
        cfg = TrainConfig(epochs=60, lr=3e-3, augment=False, model={"dropout": 0.0})
        m, norm, info = train_model(kind, tr, cfg)
        curve = info["loss_curve"]
        assert info["final_loss"] < 0.2 * curve[0], (kind, curve)
    m, norm, _ = train_model("congestion", tr, TrainConfig(epochs=80, lr=3e-3, augment=False, model={"dropout": 0.0}))
    with torch.no_grad():
        pm, _ = m(tr[0].feats)
    assert float(((pm - tr[0].cong) ** 2).mean()) < float(((tr[0].cong - tr[0].cong.mean()) ** 2).mean())


def test_ensemble_uncertainty(samples):
    s = samples[0]
    torch.manual_seed(0)
    a = TimingNet()
    b = TimingNet()
    b.load_state_dict(a.state_dict())
    (_, sd_same), (_, gsd_same) = Ensemble([a, b])(s.feats)
    assert float(sd_same.abs().max()) == 0 and float(gsd_same.abs().max()) == 0
    ens, norm, infos = train_ensemble("congestion", samples[:3], seeds=3, cfg=TrainConfig(epochs=3))
    assert len(ens.members) == 3 and len(infos) == 3
    (m, m_sd), (sc, sc_sd) = ens(s.feats)
    assert m_sd.shape == m.shape and float(m_sd.mean()) > 0 and (sc_sd > 0).all()
    p = predict(ens, norm, "congestion", s)
    assert p["grt_wl_um"] > 0 and len(p["scalars_std"]) == 2


def test_kendall_tau_and_ranking_table(samples):
    assert kendall_tau([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert abs(kendall_tau([1, 2, 3], [1, 3, 2]) - 1 / 3) < 1e-12
    assert math.isnan(kendall_tau([1, 1, 1], [1, 2, 3]))
    assert abs(spearman([1, 2, 3], [3, 2, 1]) + 1) < 1e-12
    preds = {"congestion": [{"grt_wl_um": s.scalars["grt_wl_um"], "hot_frac": -s.scalars["hot_frac"], "scalars_std": [0.1, 0.1]}
                            for s in samples]}
    t = ranking_table(samples, preds)["synth"]
    assert t["n"] == 4 and t["grt_wl"]["tau_pred"] == 1.0 and t["hot_frac"]["tau_pred"] == -1.0


# ----------------------------------------------------------------------------- RL reward hook
def test_rl_reward_hook(design, samples):
    from macromind.place import EnvConfig, PlacementEnv
    from macromind.predict.runtime import Predictor
    from macromind.predict.train import Normalizer

    torch.manual_seed(0)
    norm = Normalizer(torch.zeros(2), torch.ones(2))
    ens = {"congestion": (Ensemble([CongestionNet(), CongestionNet()]), norm),
           "timing": (Ensemble([TimingNet(), TimingNet()]), norm)}
    pred = Predictor(ens, G=16)
    p = grid_place(design)
    out = pred.predict(design, p)
    assert out["congestion_map"].shape == (4, 16, 16) and out["hot_frac_std"] > 0 and out["infer_s"] < 5
    mu, sigma = pred.penalty(design, p)
    assert math.isfinite(mu) and sigma > 0

    def episode(cfg):
        env = PlacementEnv(design, cfg)
        obs = env.reset()
        rng = np.random.default_rng(0)
        r, info, done = 0.0, {}, False
        while not done:
            obs, r, done, info = env.step(int(rng.choice(np.flatnonzero(obs["mask"]))))
        return env, r, info

    env0, r0, i0 = episode(EnvConfig(grid=16))
    env1, r1, i1 = episode(EnvConfig(grid=16, predictor=pred, pred_weight=0.0, unc_weight=0.0))
    assert r0 == r1 and "pred_penalty" not in i1  # zero weights: identical to the proxy reward
    env2, r2, i2 = episode(EnvConfig(grid=16, predictor=pred, pred_weight=0.1, unc_weight=0.0))
    env3, r3, i3 = episode(EnvConfig(grid=16, predictor=pred, pred_weight=0.1, unc_weight=1.0))
    assert "pred_penalty" in i2 and i3["pred_std"] > 0
    assert abs(i2["cost"] - (i2["proxy"].proxy + 0.1 * i2["pred_penalty"])) < 1e-9
    assert abs(i3["cost"] - i2["cost"] - 1.0 * i3["pred_std"]) < 1e-6  # same placement, extra uncertainty term
    assert env2.reference != env0.reference  # reference cost includes the predictor term too


# ----------------------------------------------------------------------------- datagen plumbing
def test_make_plan_deterministic_and_diverse():
    a, b = make_plan("bp_fe_top"), make_plan("bp_fe_top")
    assert a == b
    ids = [s["id"] for s in a]
    assert len(set(ids)) == len(ids) >= 40
    kinds = {s["kind"] for s in a}
    assert {"replay_rtlmp", "sa", "random", "grid", "perturb"} <= kinds
    assert [s["kind"] for s in a[:4]] == ["replay_rtlmp", "sa", "random", "grid"]  # interleaved
    assert len(make_plan("tinyRocket")) >= 20
    assert spec_id({"kind": "sa", "seed": 1}) != spec_id({"kind": "sa", "seed": 2})


def test_perturb_is_legal_and_moves(design):
    base = grid_place(design)
    p = perturb(design, base, seed=3, sigma=0.1, swaps=2, p_flip=1.0)
    assert not check_legality(design, p)
    assert (p.x != base.x).any() or (p.y != base.y).any()
    assert (p.orient != base.orient).any()


def test_datagen_job_identity(tmp_path, monkeypatch):
    from macromind.flow.orfs import flow_home
    from macromind.predict import datagen as dg

    if not (flow_home() / "designs/nangate45/tinyRocket/config.mk").exists():
        pytest.skip("ORFS not installed")
    monkeypatch.setenv("MACROMIND_HARNESS_DIR", str(tmp_path))
    ctx = dg.DesignContext("tinyRocket", dg.DesignSpec.parse("nangate45/tinyRocket"), tmp_path / "x.def", [], 0.0)
    t1, t2 = tmp_path / "a.tcl", tmp_path / "b.tcl"
    t1.write_text("place_macro -macro_name m -location {1 2} -orientation R0\n")
    t2.write_text("place_macro -macro_name m -location {3 2} -orientation R0\n")
    j1 = dg.make_job(ctx, {"id": "sa-1234"}, t1)
    j2 = dg.make_job(ctx, {"id": "sa-1234"}, t2)
    assert j1.key() != j2.key()  # the placement is part of the cache key
    assert j1.run_dir().parent == tmp_path / "jobs" / "nangate45_tinyRocket" and j1.run_dir().name.startswith("sa-1234-")
    assert j1.extra["MACRO_PLACEMENT_TCL"] == str(t1) and j1.extra["PRE_GLOBAL_ROUTE"].endswith("grt_labels.tcl")
    assert j1.stop_after == "grt" and j1.placer == "rtlmp"
    rd = j1.run_dir()
    (rd / "labels").mkdir(parents=True)
    assert not dg.sample_done(rd)
    (rd / "ppa.json").write_text("{}")
    (rd / "labels" / "done.json").write_text("{}")
    assert dg.sample_done(rd)


def test_ridge_baseline_and_report(samples, tmp_path):
    from macromind.predict.report import ranking_md
    from macromind.predict.train import map_metrics, ridge_predict

    r = ridge_predict(samples[:3], samples[3:])
    assert len(r) == 1 and set(r[0]) == {"grt_wl_um", "hot_frac", "wns_ns", "tns_ns"} and r[0]["grt_wl_um"] > 0
    preds = [{"map": s.cong.clone()} for s in samples]
    m = map_metrics(samples, preds, [samples[0].cong] * len(samples))["synth"]
    assert m["mse"] == 0 and abs(m["pixel_corr"] - 1) < 1e-6
    res = {"ranking": {"synth": {"n": 4, "grt_wl": {"tau_pred": 0.5, "tau_proxy": 0.1, "tau_ridge": None, "tau_hpwl": 0.0,
                                                     "rho_pred": 0.6, "rho_proxy": 0.2, "rho_unc_err": float("nan")}}}}
    md = ranking_md(res)
    assert "| synth | 4 | GRT wirelength, scalar head | +0.50 | +0.10 | - |" in md and "n/a" in md


def test_per_design_normalizer():
    y = torch.tensor([[1.0], [2.0], [3.0], [101.0], [102.0], [103.0]])
    n = Normalizer.fit(y, ["a"] * 3 + ["b"] * 3)
    assert torch.allclose(n(y[0], "a"), n(y[3], "b"))  # same within-design rank -> same target
    assert torch.allclose(n.inv(n(y[4], "b"), "b"), y[4])
    st = Normalizer.from_state(json.loads(json.dumps(n.state())))
    assert torch.allclose(st(y[1], "a"), n(y[1], "a")) and torch.allclose(st(y[1], "zz"), n(y[1]))
