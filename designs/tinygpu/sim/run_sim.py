#!/usr/bin/env python3
"""Behavioural check of the tiny-gpu macro variant against upstream and the ISA model.

For each kernel, builds a 32-bit / 4096-word memory image with the upstream xf
toolchain, runs upstream's pure-Python ISA simulator (xf/isasim.py) as the golden
model, and runs tinygpu_top in Icarus Verilog (RTL ingested with yosys-slang,
written back as Verilog by Yosys) for MACROS=0 (upstream gpu, flop memories) and
MACROS=1 (gpu_mm, SRAM macros with behavioural fakeram45 models). Final data
memories must match the golden model word for word.

Usage: python3 designs/tinygpu/sim/run_sim.py [--cores N] [--kernels a,b] [--out DIR]
Needs: yosys (scripts/env.sh), iverilog/vvp, numpy.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TG = HERE.parent
ROOT = TG.parent.parent
UP = TG / "upstream"


def files() -> list[str]:
    return [str(TG / line.strip()) for line in (TG / "files.f").read_text().splitlines() if line.strip()]


def build(out: Path, macros: int, cores: int, defines: dict[str, int]) -> Path:
    """Yosys-slang -> Verilog netlist (RTL level) -> Icarus vvp."""
    tag = f"m{macros}_c{cores}_w{defines.get('TG_WARPS', 1)}"
    v = out / f"tinygpu_{tag}.v"
    vvp = out / f"tinygpu_{tag}.vvp"
    d = {"TG_MACROS": macros, "TG_NUM_CORES": cores, **defines}
    dargs = " ".join(f"-D {k}={val}" for k, val in d.items())
    models = str(HERE / "fakeram45_models.v")
    script = (f"read_slang {' '.join(files())} {models if macros else ''} --top tinygpu_top {dargs}; "
              f"hierarchy -top tinygpu_top; proc; opt_clean; write_verilog -noattr {v}")
    yosys = os.environ.get("YOSYS_EXE", "yosys")
    subprocess.run([yosys, "-q", "-m", "slang", "-p", script], check=True)
    ivd = [f"-DTG_DATA_CHANNELS={d.get('TG_DATA_CHANNELS', 4)}",
           f"-DTG_DATA_ADDR_BITS={d.get('TG_DATA_ADDR_BITS', 12)}",
           f"-DTG_WORD_BITS={d.get('TG_WORD_BITS', 32)}"]
    srcs = [str(v), str(HERE / "tb_tinygpu.v")] + ([models] if macros else [])
    subprocess.run(["iverilog", "-g2012", "-s", "tb", "-o", str(vvp), *ivd, *srcs], check=True)
    return vvp


def kernels(seed: int = 1):
    """Yield (name, program words, image words, threads, image, output segment)."""
    sys.path.insert(0, str(UP))
    import numpy as np
    from xf import kernels as K
    from xf.image import Image
    rng = np.random.default_rng(seed)
    for kind, (m, k, n) in (("tiled", (8, 8, 16)), ("naive", (4, 8, 8))):
        img = Image(word_bits=32, addr_bits=12)
        img.alloc("a", [m, k], "i8", data=rng.integers(-128, 128, (m, k)))
        img.alloc("b", [k, n], "i8", data=rng.integers(-128, 128, (k, n)))
        img.alloc("c", [m, n], "i32")
        prog, threads = K.matmul(img, kind, "a", "b", "c")
        yield f"matmul_{kind}", list(prog), list(img.words), threads, img, "c"
    img = Image(word_bits=32, addr_bits=12)
    from xf import elementwise as E
    if hasattr(E, "transpose"):
        img.alloc("x", [6, 10], "i32", data=rng.integers(-1000, 1000, (6, 10)))
        img.alloc("y", [10, 6], "i32")
        prog, threads = E.transpose(img, "x", "y")
        yield "transpose", list(prog), list(img.words), threads, img, "y"


def golden(prog, data, threads):
    sys.path.insert(0, str(UP))
    from xf import isasim
    return isasim.run(prog, data, threads, word_bits=32, addr_bits=12).memory


def run_vvp(vvp: Path, out: Path, name: str, prog, data, threads) -> tuple[list[int], int]:
    ph, dh, oh = out / f"{name}.prog.hex", out / f"{name}.data.hex", out / f"{name}.{vvp.stem}.out.hex"
    ph.write_text("".join(f"{w:04x}\n" for w in prog))
    dh.write_text("".join(f"{w & 0xffffffff:08x}\n" for w in data))
    r = subprocess.run(["vvp", "-n", str(vvp), f"+prog={ph}", f"+data={dh}", f"+out={oh}", f"+threads={threads}", "+maxcycles=400000"],
                       check=True, capture_output=True, text=True)
    m = re.search(r"TB_DONE done=(\d) cycles=(\d+)", r.stdout)
    if not m or m.group(1) != "1":
        raise RuntimeError(f"{name}: no done\n{r.stdout[-2000:]}")
    mem = [int(x, 16) if re.fullmatch(r"[0-9a-f]+", x) else -1 for x in oh.read_text().split()]
    return mem, int(m.group(2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "runs" / "tinygpu" / "sim"))
    ap.add_argument("--macros", default="0,1")
    ap.add_argument("--warps", type=int, default=4, help="WARPS_PER_CORE of the MACROS=1 build")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # MACROS=0 (upstream gpu) runs without I-cache: upstream fetcher never clears its
    # valid bits (the telemetry_reset wipe is shadowed by the per-core reset), so in a
    # 2-state-faithful netlist sim the first lookup reads X. Its L1-D is shrunk to 16
    # lines because a 1024-line flop cache simulates ~100x slower.
    # MACROS=1 is checked both with the configs' 4 warps x 1 lane (one shared ALU per
    # core) and with upstream's default 1 warp x 4 lanes (exercises the SMEM arbiter).
    defs = {0: {"TG_ICACHE_LINES": 0, "TG_DCACHE_LINES": 16, "TG_WARPS": 1},
            1: {"TG_ICACHE_LINES": 64, "TG_DCACHE_LINES": 1024, "TG_DCACHE_BANKS": 4, "TG_WARPS": a.warps}}
    vvps = {mac: build(out, mac, a.cores, defs[mac]) for mac in (int(x) for x in a.macros.split(","))}
    fails = 0
    for name, prog, data, threads, img, seg in kernels():
        gold = golden(prog, data, threads)
        for mac, vvp in vvps.items():
            mem, cycles = run_vvp(vvp, out, name, prog, data, threads)
            bad = [i for i, (x, y) in enumerate(zip(mem, gold)) if x != y]
            ok = not bad
            fails += not ok
            print(f"{name:16s} MACROS={mac} warps={defs[mac]['TG_WARPS']} cores={a.cores} threads={threads:3d} cycles={cycles:7d} "
                  f"{'MATCH' if ok else f'MISMATCH at {len(bad)} words, first {bad[:5]}'}")
    print("RESULT:", "PASS" if not fails else f"FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
