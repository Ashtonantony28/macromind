#!/usr/bin/env python3
"""Run SpikeMesh's own neurocore cocotb bench (upstream tests/core/core_tb.py, golden
model included) on the MacroMind macro variant: upstream neurocore.sv + learn_engine.sv
with sram_1r1w replaced by designs/spikemesh/rtl/sram_1r1w.sv and the sky130 macro
model. Every configuration here maps at least one memory onto macros.

    python3 designs/spikemesh/sim/run_sim.py            # needs upstream/ (run_spikemesh.sh fetch)
"""
import os
import sys
from pathlib import Path

from cocotb_tools.runner import get_results, get_runner

HERE = Path(__file__).resolve().parent
D = HERE.parent
UP = D / "upstream"
SRC = [D / "rtl" / "sram_1r1w.sv", HERE / "sky130_sram_1rw1r_64x256_8.v",
       UP / "rtl/core/learn_engine.sv", UP / "rtl/core/neurocore.sv"]

# name: (neurocore params, [(testcase, env)])   -- macros: WDEPTH = A * N / 8 rows of 64 bits
CONFIGS = {
    # the hardened config: 8 weight macros + 2 fan-out macros
    "n128": (dict(N=128, A=128, FD=256, LEARN=0), [
        ("test_random", dict(SEED=61, MODE="mixed", STEPS=4, BP=0.3)),
        ("test_random", dict(SEED=62, MODE="sat", STEPS=3, BP=0.5)),
        ("test_cfg_readback", dict(SEED=63, BP=0.5)),
        ("test_cycles", dict(SEED=1)),
    ]),
    # learning build: eligibility traces and STDP read-modify-write through the macros
    "n128_l2": (dict(N=128, A=128, FD=256, LEARN=2), [
        ("test_learn", dict(SEED=71, MODE="rstdp", STEPS=4, BP=0.3)),
    ]),
    # upstream's full-size core: 32 weight macros + 8 fan-out macros
    "big": (dict(N=256, A=256, FD=1024, LEARN=0), [
        ("test_random", dict(SEED=51, MODE="sparse", STEPS=3, BP=0.3)),
    ]),
}


def run(name):
    for pth in (str(UP), str(UP / "tests/core")):       # the cocotb runner passes sys.path on
        if pth not in sys.path:
            sys.path.insert(0, pth)
    p, cases = CONFIGS[name]
    bdir = D / "sim" / "sim_build" / name
    runner = get_runner("icarus")
    runner.build(sources=[str(s) for s in SRC], hdl_toplevel="neurocore", build_dir=str(bdir), always=True,
                 parameters=dict(X=2, Y=1, N_NEURONS=p["N"], N_AXONS=p["A"], W_BITS=8,
                                 FANOUT_DEPTH=p["FD"], LANES=8, LEARN=p["LEARN"], ADAPT_TH=1),
                 timescale=("1ns", "1ps"))
    fails = 0
    for tc, env in cases:
        e = {"N": p["N"], "A": p["A"], "LANES": 8, "W": 8, "FD": p["FD"], "X": 2, "Y": 1,
             "LEARN": p["LEARN"], "ADAPT": 1, **env}
        e = {k: str(v) for k, v in e.items()}
        xml = bdir / f"results_{tc}_{env.get('SEED', 0)}.xml"
        runner.test(hdl_toplevel="neurocore", test_module="core_tb", testcase=tc, build_dir=str(bdir),
                    test_dir=str(bdir), extra_env=e, results_xml=str(xml))
        n, f = get_results(xml)
        print(f"[{name}] {tc} {env}: {'PASS' if n and not f else 'FAIL'}", flush=True)
        fails += (f or not n)
    return fails


if __name__ == "__main__":
    assert UP.is_dir(), "upstream/ missing: run scripts/run_spikemesh.sh fetch"
    names = sys.argv[1:] or list(CONFIGS)
    sys.exit(1 if sum(run(n) for n in names) else 0)
