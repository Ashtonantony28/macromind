"""SG7: ORFS hook config, placement emit/parse, metric parsing, cache keys, report; eda integration runs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from macromind.flow import emit
from macromind.flow.orfs import HOOK_TCL, REPO_ROOT, DesignSpec, PlacerSpec, wrapper_config
from macromind.harness import metrics
from macromind.harness.runner import Job
from macromind.io import read_def
from macromind import report

ORFS_FIX = Path(__file__).parent / "fixtures" / "orfs_tinyRocket"


# ----------------------------------------------------------------------------- config generation
def test_specs_parse():
    assert DesignSpec.parse("nangate45/bp_fe_top") == DesignSpec("nangate45", "bp_fe_top")
    assert DesignSpec.parse("gcd") == DesignSpec("nangate45", "gcd")
    assert DesignSpec("asap7", "riscv32i").tag == "asap7_riscv32i"
    with pytest.raises(ValueError):
        DesignSpec.parse("a/b/c")
    assert PlacerSpec.parse("rtlmp").kind == "rtlmp"
    assert PlacerSpec.parse("macromind-sa") == PlacerSpec("macromind", "sa")
    assert PlacerSpec.parse("grid").name == "macromind-grid"
    with pytest.raises(ValueError):
        PlacerSpec.parse("dreamplace")


def test_wrapper_config_rtlmp_and_macromind(tmp_path):
    d = DesignSpec("nangate45", "bp_fe_top")
    rt = wrapper_config(d, PlacerSpec.parse("rtlmp"), 0, tmp_path, home=Path("/flow"))
    assert "include /flow/designs/nangate45/bp_fe_top/config.mk" in rt
    assert "export MACRO_PLACER = rtlmp" in rt
    assert "MACRO_PLACEMENT_TCL" not in rt
    mm = wrapper_config(d, PlacerSpec.parse("macromind-sa"), 3, tmp_path / "mm", python="/venv/python", budget=500,
                        extra={"PLACE_DENSITY": "0.6", "X": "$y"}, home=Path("/flow"))
    assert f"export MACRO_PLACEMENT_TCL = {HOOK_TCL}" in mm
    assert "export MACROMIND_METHOD = sa" in mm and "export MACROMIND_SEED = 3" in mm
    assert "export MACROMIND_PYTHON = /venv/python" in mm and "export MACROMIND_BUDGET = 500" in mm
    assert f"export MACROMIND_OUT_DIR = {(tmp_path / 'mm').resolve()}" in mm
    assert "export PLACE_DENSITY = 0.6" in mm and "export X = $$y" in mm  # make-escaped
    assert "MACROMIND_RL_CKPT" not in mm
    rl = wrapper_config(d, PlacerSpec.parse("rl"), 0, tmp_path, rl_ckpt="/ck/policy.pt", home=Path("/flow"))
    assert "export MACROMIND_RL_CKPT = /ck/policy.pt" in rl


def test_hook_tcl_uses_supported_mechanism():
    txt = HOOK_TCL.read_text()
    assert "write_def" in txt and "macromind.flow.place" in txt and "source $mm_tcl" in txt
    flow = REPO_ROOT / "tools/ORFS/flow/scripts/macro_place_util.tcl"
    if flow.exists():  # the ORFS version we target sources MACRO_PLACEMENT_TCL instead of running RTLMP
        assert "source $::env(MACRO_PLACEMENT_TCL)" in flow.read_text()


# ----------------------------------------------------------------------------- emit round trip
@pytest.fixture
def tiny(fixtures):
    return read_def(fixtures / "tiny.def", fixtures / "tiny.lef")


def test_tcl_roundtrip(tiny):
    assert tiny.macros
    p = tiny.placement()
    rng = np.random.default_rng(0)
    p.x[:] = rng.integers(0, 100000, len(p)) // 10 * 10
    p.y[:] = rng.integers(0, 100000, len(p)) // 10 * 10
    p.orient[:] = rng.choice([0, 2, 4, 6], len(p))
    for m in tiny.macros:
        m.fixed = False
    txt = emit.placement_to_tcl(tiny, p)
    assert txt.count("place_macro") == len(tiny.macros)
    parsed = emit.parse_place_macro_tcl(txt)
    q = emit.apply_um_placement(tiny, parsed)
    assert np.array_equal(p.x, q.x) and np.array_equal(p.y, q.y) and np.array_equal(p.orient, q.orient)


def test_parse_orfs_written_tcl_and_braced_names():
    txt = ("place_macro -macro_name dcache/data.mem -location {32.47 25.025} -orientation R180\n"
           "place_macro -macro_name {a\\[0\\]/mem} -location {1 2} -orientation MX\n# comment\n")
    got = emit.parse_place_macro_tcl(txt)
    assert got["dcache/data.mem"] == (32.47, 25.025, 2)
    assert got["a\\[0\\]/mem"] == (1.0, 2.0, 6)


def test_def_emission_roundtrip(tiny, fixtures, tmp_path):
    for m in tiny.macros:
        m.fixed = False
    p = tiny.placement()
    p.x[:] = 40000 + 1000 * np.arange(len(p))
    p.y[:] = 50000
    out = tmp_path / "placed.def"
    assert emit.write_placement(out, tiny, p, src_def=fixtures / "tiny.def") == "def"
    back = read_def(out, fixtures / "tiny.lef")
    assert [m.x for m in back.macros] == list(p.x) and all(m.fixed for m in back.macros)
    assert emit.write_placement(tmp_path / "p.tcl", tiny, p) == "tcl"
    with pytest.raises(ValueError):
        emit.write_placement(tmp_path / "x.def", tiny, p)


def test_flow_place_cli_on_fixture(fixtures, tmp_path):
    from macromind.flow.place import main

    out = tmp_path / "placed.tcl"
    js = tmp_path / "p.json"
    rc = main(["--def", str(fixtures / "tiny.def"), "--lef", str(fixtures / "tiny.lef"), "--method", "grid",
               "--out", str(out), "--json", str(js), "--clusters", "8"])
    assert rc == 0
    rec = json.loads(js.read_text())
    assert rec["proxy"]["legal"] and np.isfinite(rec["proxy"]["proxy"])
    placed = emit.parse_place_macro_tcl(out.read_text())
    movable = [m.name for m in read_def(fixtures / "tiny.def", fixtures / "tiny.lef").macros if not m.fixed]
    assert sorted(placed) == sorted(movable)  # fixed macros are left to ORFS


# ----------------------------------------------------------------------------- metric parsing (real ORFS files)
def test_merge_and_extract_ppa():
    m = metrics.merge_stage_jsons(ORFS_FIX)
    ppa, src = metrics.extract_ppa(m)
    assert ppa["routed_wl_um"] == 637956 and src["routed_wl_um"] == "detailedroute__route__wirelength"
    assert ppa["drc_errors"] == 0
    assert ppa["wns_ns"] == pytest.approx(-0.097767, abs=1e-6) and src["wns_ns"].startswith("finish__")
    assert ppa["tns_ns"] == pytest.approx(-26.6243, abs=1e-3)
    assert ppa["power_w"] == pytest.approx(0.119474, rel=1e-4)
    assert ppa["grt_wl_um"] == 816305 and ppa["macros"] == 2


def test_extract_ppa_falls_back_to_grt(tmp_path):
    for f in ("2_1_floorplan.json", "5_1_grt.json"):
        shutil.copy(ORFS_FIX / f, tmp_path / f)
    ppa, src = metrics.extract_ppa(metrics.merge_stage_jsons(tmp_path))
    assert ppa["routed_wl_um"] is None and ppa["drc_errors"] is None
    assert src["wns_ns"] == "globalroute__timing__setup__ws" and ppa["power_w"] is not None


def test_grt_overflow_and_runtimes():
    ov = metrics.grt_overflow((ORFS_FIX / "5_1_grt.log").read_text())
    assert ov == {"resource": 1451826, "demand": 300057, "usage_pct": 20.67, "max_h": 0, "max_v": 0, "total": 0,
                  "tables": ov["tables"]}
    synthetic = "Total          100        50           50.00%             3 /  4 /  7\n"
    assert metrics.grt_overflow(synthetic)["total"] == 7
    assert metrics.grt_overflow("nothing") is None
    rt = metrics.stage_runtimes(ORFS_FIX)
    assert rt["5_1_grt"]["s"] == pytest.approx(258.15) and rt["5_1_grt"]["peak_mb"] > 700
    assert metrics.parse_hms("1:02:03.5") == pytest.approx(3723.5)
    rec = metrics.collect(ORFS_FIX)
    assert rec["ppa"]["grt_overflow"] == 0 and rec["flow_runtime_s"] > 0


# ----------------------------------------------------------------------------- cache keys
def test_cache_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("MACROMIND_HARNESS_DIR", str(tmp_path))
    a = Job("nangate45/gcd", "rtlmp", 0)
    assert a.key() == Job("gcd", "default", 0).key()  # normalised spec
    assert a.key() != Job("nangate45/gcd", "rtlmp", 1).key()
    assert a.key() != Job("nangate45/gcd", "rtlmp", 0, stop_after="grt").key()
    assert a.key() != Job("nangate45/gcd", "rtlmp", 0, extra={"PLACE_DENSITY": "0.5"}).key()
    sa = Job("nangate45/gcd", "sa", 0)
    assert sa.placer == "macromind-sa" and "code" in sa.key_material() and "code" not in a.key_material()
    assert sa.key() != Job("nangate45/gcd", "grid", 0).key()
    assert sa.run_dir().parent == tmp_path / "jobs" / "nangate45_gcd"
    assert sa.run_dir().name.startswith("macromind-sa-s0-finish-")
    with pytest.raises(ValueError):
        Job("nangate45/gcd", "rtlmp", 0, stop_after="tapeout")


def test_cache_hit_does_not_rerun(monkeypatch, tmp_path):
    from macromind.harness import runner

    monkeypatch.setenv("MACROMIND_HARNESS_DIR", str(tmp_path))
    j = Job("nangate45/gcd", "rtlmp", 0)
    j.run_dir().mkdir(parents=True)
    (j.run_dir() / "ppa.json").write_text(json.dumps({"status": "ok", "job": {"design": j.design}}))
    monkeypatch.setattr(runner, "ensure_base", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    rec = runner.run_job(j)
    assert rec["cached"] and rec["status"] == "ok"


# ----------------------------------------------------------------------------- report
def test_report_mean_ci():
    m, h, n = report.mean_ci([1.0, 2.0, 3.0])
    assert m == 2.0 and n == 3 and h == pytest.approx(4.303 * 1 / 3 ** 0.5, rel=1e-3)
    assert report.mean_ci([5.0])[1] is None
    assert report.fmt_ci([]) == "n/a"


def test_build_report():
    def rec(placer, seed, wl, status="ok"):
        return {"job": {"design": "nangate45/x", "placer": placer, "seed": seed, "stop_after": "finish"},
                "status": status, "ppa": {"routed_wl_um": wl, "wns_ns": -0.1, "power_w": 0.1, "drc_errors": 0},
                "placement": {"proxy": wl / 1e6}, "make_wall_s": 600, "error": "boom" if status != "ok" else None}

    md = report.build_report([rec("rtlmp", 0, 1e6), rec("rtlmp", 1, 1.2e6), rec("macromind-sa", 0, 2e6),
                              rec("macromind-sa", 1, 0, "failed")])
    assert "## nangate45/x (stop after finish)" in md
    assert "| rtlmp | 2/2 | 0,1 |" in md and "| macromind-sa | 1/2 | 0 |" in md
    assert "1,100,000 ± " in md and "failed" in md and "Spearman" in md


# ----------------------------------------------------------------------------- EDA integration
def _have_orfs() -> bool:
    return (REPO_ROOT / "tools/bin/openroad").exists() and (REPO_ROOT / "tools/ORFS/flow/Makefile").exists()


eda = pytest.mark.skipif(not _have_orfs(), reason="ORFS toolchain not installed (scripts/setup_toolchain.sh)")


@pytest.mark.eda
@eda
def test_eda_gcd_rtlmp_to_grt():
    """gcd (no macros) through global route with the default placer: metrics + runtimes parsed."""
    from macromind.harness.runner import run_job

    rec = run_job(Job("nangate45/gcd", "rtlmp", 0, stop_after="grt", num_cores=2))
    assert rec["status"] == "ok", rec.get("error_tail") or rec.get("error")
    p = rec["ppa"]
    assert p["grt_wl_um"] > 0 and p["wns_ns"] is not None and p["power_w"] > 0
    assert p["grt_overflow"] is not None and "5_1_grt" in rec["stage_runtime"]
    assert rec["placement"]["unplaced_macros"] == 0


@pytest.mark.eda
@eda
def test_eda_hook_tinyrocket_floorplan():
    """The MACRO_PLACEMENT_TCL hook places tinyRocket's 2 macros with macromind-grid (stop after floorplan)."""
    from macromind.harness.runner import run_job

    rec = run_job(Job("nangate45/tinyRocket", "macromind-grid", 0, stop_after="floorplan", num_cores=2))
    assert rec["status"] == "ok", rec.get("error_tail") or rec.get("error")
    pl = rec["placement"]
    assert pl["unplaced_macros"] == 0 and len(pl["macros"]) == 2
    tcl = emit.parse_place_macro_tcl((Path(rec["run_dir"]) / "macromind" / "macro_placement.tcl").read_text())
    for m in pl["macros"]:  # what ORFS ended with == what MacroMind asked for (place_macro snaps < 0.2 um)
        x, y, _ = tcl[m["name"]]
        assert abs(m["x"] - x) < 0.2 and abs(m["y"] - y) < 0.2
    assert rec["placer"]["proxy"]["legal"]


def test_report_dedupes_superseded_runs():
    old = {"job": {"design": "d", "placer": "p", "seed": 0, "stop_after": "grt"}, "status": "ok", "finished": "2026-01-01",
           "ppa": {"grt_wl_um": 1.0}}
    new = {**old, "finished": "2026-02-01", "ppa": {"grt_wl_um": 2.0}}
    other = {**old, "job": {**old["job"], "seed": 1}}
    out = report.dedupe([old, new, other])
    assert len(out) == 2 and new in out


# ----------------------------------------------------------------------------- SG10 additions
def test_job_tag_key_and_report_label(monkeypatch, tmp_path):
    monkeypatch.setenv("MACROMIND_HARNESS_DIR", str(tmp_path))
    plain = Job("nangate45/gcd", "rl", 0, stop_after="grt")
    tagged = Job("nangate45/gcd", "rl", 0, stop_after="grt", tag="rl-v2-proxy")
    assert "tag" not in plain.key_material() and tagged.key_material()["tag"] == "rl-v2-proxy"
    assert plain.key() != tagged.key() and tagged.run_dir().name.startswith("rl-v2-proxy-s0-grt-")

    def rec(tag, wl):
        return {"job": {"design": "nangate45/x", "placer": "macromind-rl", "seed": 0, "stop_after": "grt", "tag": tag},
                "status": "ok", "ppa": {"grt_wl_um": wl}, "placer": {"total_s": 3.0}}

    md = report.build_report([rec(None, 1e6), rec("rl-v2-proxy", 2e6)])  # same seed, different variants: both kept
    assert "| macromind-rl | 1/1 |" in md and "| rl-v2-proxy | 1/1 |" in md


def test_local_design_adapter():
    from macromind.flow.orfs import LOCAL_DESIGNS, design_digest

    d = DesignSpec.parse("nangate45/tinygpu_c4")
    assert d.config_path() == LOCAL_DESIGNS / "nangate45/tinygpu_c4/config.mk"
    assert "TG_CONFIG = c4" in d.config_path().read_text()
    assert len(design_digest(d)) == 16  # hashes the adapter and designs/tinygpu
