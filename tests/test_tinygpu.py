"""SG9: tiny-gpu-extended macro variant (designs/tinygpu).

Fast tests check the derived RTL and the macro table; `eda` tests elaborate the
RTL with yosys-slang and count fakeram45 macros per configuration, and run the
RTL simulation (upstream gpu vs macro variant vs upstream ISA model).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TG = ROOT / "designs" / "tinygpu"
UPSTREAM = TG / "upstream" / "src"
FLOW = ROOT / "tools" / "ORFS" / "flow"
YOSYS = ROOT / "tools" / "ORFS" / "tools" / "install" / "yosys" / "bin" / "yosys"

needs_upstream = pytest.mark.skipif(not UPSTREAM.is_dir(), reason="upstream not cloned: scripts/run_tinygpu.sh fetch")

# config -> (NUM_CORES, DCACHE_BANKS); expected macros = cores x (SMEM + I-cache) + banks
CONFIGS = {"c4": (4, 4), "c8": (8, 4), "c16": (16, 8)}


def expected_macros(cores: int, banks: int) -> dict[str, int]:
    return {"fakeram45_64x32": cores, "fakeram45_64x21": cores, "fakeram45_256x34": banks}


def test_pin_file():
    pin = (TG / "UPSTREAM_PIN").read_text().strip()
    assert re.fullmatch(r"[0-9a-f]{40}", pin)
    if (TG / "upstream" / ".git").is_dir():
        head = subprocess.run(["git", "-C", str(TG / "upstream"), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        assert head == pin


@needs_upstream
def test_derived_rtl_up_to_date():
    r = subprocess.run([sys.executable, str(TG / "derive_rtl.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_configs_match_test_table():
    txt = (TG / "configs.mk").read_text()
    for cfg, (cores, banks) in CONFIGS.items():
        block = txt.split(f"($(TG_CONFIG),{cfg})", 1)[1].split("else", 1)[0]
        assert f"TG_NUM_CORES={cores}" in block and f"TG_DCACHE_BANKS={banks}" in block
        if banks * 256 != 1024:
            assert f"TG_DCACHE_LINES={banks * 256}" in block


def test_mm_sram_macros_exist_in_platform():
    names = set(re.findall(r"fakeram45_(\d+x\d+) u_ram", (TG / "rtl" / "mm_sram.sv").read_text()))
    assert {"64x32", "64x21", "256x34"} <= names
    lefdir = FLOW / "platforms" / "nangate45" / "lef"
    if lefdir.is_dir():
        have = {p.stem.removeprefix("fakeram45_") for p in lefdir.glob("fakeram45_*.lef")}
        assert names <= have, names - have


def _elaborate_macros(cfg: str) -> dict[str, int]:
    cores, banks = CONFIGS[cfg]
    lib = FLOW / "platforms" / "nangate45" / "lib"
    libs = " ".join(str(lib / f"{m}.lib") for m in expected_macros(1, 1))
    files = " ".join(str(TG / f) for f in (TG / "files.f").read_text().split())
    script = (f"read_liberty -lib {libs}; read_slang {files} --top tinygpu_top "
              f"-D TG_MACROS=1 -D TG_NUM_CORES={cores} -D TG_DCACHE_BANKS={banks} -D TG_DCACHE_LINES={256 * banks}; "
              "hierarchy -top tinygpu_top; flatten; select -list t:fakeram45_*")
    env = dict(os.environ, LD_LIBRARY_PATH=str(ROOT / "tools" / "ORFS" / "tools" / "lib"))
    r = subprocess.run([str(YOSYS), "-q", "-m", "slang", "-p", script], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    counts: dict[str, int] = {}
    # flatten keeps cell types; count instances by type with a second listing pass
    script2 = script.replace("select -list t:fakeram45_*", "stat")
    r2 = subprocess.run([str(YOSYS), "-m", "slang", "-p", script2], capture_output=True, text=True, env=env)
    for name, n in re.findall(r"^\s+(\d+)\s+(?:[\d.]+\s+)?(fakeram45_\w+)$", r2.stdout, re.M):
        counts[n] = int(name)
    if not counts:
        for name, n in re.findall(r"^\s+(fakeram45_\w+)\s+(\d+)$", r2.stdout, re.M):
            counts[name] = int(n)
    return counts


@pytest.mark.eda
@needs_upstream
@pytest.mark.skipif(not YOSYS.exists(), reason="toolchain not installed")
@pytest.mark.parametrize("cfg", sorted(CONFIGS))
def test_macro_count_per_config(cfg):
    cores, banks = CONFIGS[cfg]
    counts = _elaborate_macros(cfg)
    assert counts == expected_macros(cores, banks)
    assert 8 <= sum(counts.values()) <= 40


@pytest.mark.eda
@needs_upstream
@pytest.mark.skipif(not shutil.which("iverilog") or not YOSYS.exists(), reason="needs iverilog + yosys")
def test_rtl_sim_matches_isa_model(tmp_path):
    env = dict(os.environ, YOSYS_EXE=str(YOSYS), LD_LIBRARY_PATH=str(ROOT / "tools" / "ORFS" / "tools" / "lib"))
    r = subprocess.run([sys.executable, str(TG / "sim" / "run_sim.py"), "--cores", "2", "--out", str(tmp_path)],
                       capture_output=True, text=True, env=env, timeout=1800)
    assert r.returncode == 0 and "RESULT: PASS" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    assert r.stdout.count("MATCH") == 6


@pytest.mark.eda
def test_orfs_synth_macro_count_if_run():
    """If scripts/run_tinygpu.sh has produced a synthesized netlist, it holds the expected macros."""
    hits = 0
    for cfg, (cores, banks) in CONFIGS.items():
        net = ROOT / "runs" / "orfs" / "tinygpu" / "results" / "nangate45" / f"tinygpu_{cfg}" / "base" / "1_synth.v"
        if not net.exists():
            continue
        hits += 1
        txt = net.read_text()
        counts = {m: len(re.findall(rf"^\s*{m}\s", txt, re.M)) for m in expected_macros(1, 1)}
        assert counts == expected_macros(cores, banks)
    if not hits:
        pytest.skip("no ORFS tinygpu synthesis results; run scripts/run_tinygpu.sh c8 synth")
