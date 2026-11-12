"""SpikeMesh neurocore macro variant (designs/spikemesh).

Fast tests check the pin, the macro arithmetic in configs.mk and that the OpenRAM
macro views exist; `eda` tests run upstream's core cocotb tests on the variant and
count macros in an ORFS netlist if one has been produced.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SM = ROOT / "designs" / "spikemesh"
FLOW = ROOT / "tools" / "ORFS" / "flow"
RAM = "sky130_sram_1rw1r_64x256_8"
ROWS, BITS = 256, 64

needs_upstream = pytest.mark.skipif(not (SM / "upstream" / "rtl").is_dir(),
                                    reason="upstream not cloned: scripts/run_spikemesh.sh fetch")


def config_params(cfg: str) -> dict[str, int]:
    txt = (SM / "configs.mk").read_text()
    block = txt.split(f"($(SM_CONFIG),{cfg})", 1)[1].split("else", 1)[0]
    return {k: int(v) for k, v in re.findall(r"-D SM_(\w+)=(\d+)", block)}


def expected_macros(p: dict[str, int]) -> int:
    # weights: N*A bytes in 64-bit rows; fan-out dest + axon tables: FD rows each.
    wmem = p["N"] * p["A"] * 8 // BITS // ROWS
    return wmem + 2 * (p["FD"] // ROWS)


def test_pin_file():
    pin = (SM / "UPSTREAM_PIN").read_text().strip()
    assert re.fullmatch(r"[0-9a-f]{40}", pin)
    if (SM / "upstream" / ".git").is_dir():
        head = subprocess.run(["git", "-C", str(SM / "upstream"), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        assert head == pin


def test_n128_is_ten_macros():
    assert expected_macros(config_params("n128")) == 10


def test_wrapper_only_maps_true_fits():
    src = (SM / "rtl" / "sram_1r1w.sv").read_text()
    assert RAM in src and "FORCE_FLOPS" in src


def test_macro_views_exist_in_platform():
    d = FLOW / "platforms" / "sky130ram" / RAM
    if not d.is_dir():
        pytest.skip("toolchain not installed")
    for ext in (".lef", ".gds", "_TT_1p8V_25C.lib"):
        assert (d / f"{RAM}{ext}").exists()


@pytest.mark.eda
@pytest.mark.slow
@needs_upstream
@pytest.mark.skipif(not shutil.which("iverilog"), reason="needs iverilog")
def test_core_sim_n128():
    r = subprocess.run([sys.executable, str(SM / "sim" / "run_sim.py"), "n128"],
                       capture_output=True, text=True, timeout=3600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "FAIL" not in re.sub(r"FAIL=0", "", r.stdout)


@pytest.mark.eda
def test_orfs_synth_macro_count_if_run():
    hits = 0
    for base in (ROOT / "runs" / "orfs" / "spikemesh", FLOW):
        net = base / "results" / "sky130hd" / "spikemesh_n128" / "base" / "1_synth.v"
        if net.exists():
            hits += 1
            assert len(re.findall(rf"^\s*{RAM}\s", net.read_text(), re.M)) == 10
    if not hits:
        pytest.skip("no ORFS spikemesh synthesis results; run scripts/run_spikemesh.sh n128 synth")
