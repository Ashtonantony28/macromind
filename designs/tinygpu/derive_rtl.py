#!/usr/bin/env python3
"""Derive the MacroMind copies of tiny-gpu-extended RTL from the pinned upstream.

Upstream (designs/tinygpu/upstream/, pinned in UPSTREAM_PIN) is never edited.
Every derived file is produced by exact, asserted text substitutions so the
difference to upstream is small, reviewable and reproducible:

  rtl/controller.sv  synthesis-front-end fix only (behaviour-identical)
  rtl/lsu_mm.sv      shared-memory WAITING state waits for smem_ready
  rtl/core_mm.sv     core wired to shared_memory_mm / fetcher_mm / lsu_mm
  rtl/gpu_mm.sv      gpu wired to core_mm / l1d_cache_mm (+ DCACHE_BANKS)

Hand-written (not derived): rtl/mm_sram.sv, rtl/shared_memory_mm.sv,
rtl/fetcher_mm.sv, rtl/l1d_cache_mm.sv, rtl/tinygpu_top.sv.

Usage: python3 designs/tinygpu/derive_rtl.py [--check]
  --check  exit 1 if any derived file differs from what would be generated.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
UP = HERE / "upstream" / "src"
OUT = HERE / "rtl"


def banner(name: str, what: str) -> str:
    return (
        f"// DERIVED by designs/tinygpu/derive_rtl.py from upstream src/{name}\n"
        f"// (Didactic_GPU_Simulator, commit in designs/tinygpu/UPSTREAM_PIN). Do not edit by hand.\n"
        f"// MacroMind change: {what}\n"
    )


class Src:
    def __init__(self, name: str):
        self.name = name
        self.text = (UP / name).read_text()

    def rep(self, old: str, new: str, count: int | None = 1) -> None:
        n = self.text.count(old)
        if n == 0 or (count is not None and n != count):
            raise SystemExit(f"{self.name}: expected {count} x {old!r}, found {n}")
        self.text = self.text.replace(old, new)

    def head(self, what: str) -> None:
        self.rep("`timescale 1ns/1ns\n", "`timescale 1ns/1ns\n" + banner(self.name, what))


def controller() -> str:
    s = Src("controller.sv")
    s.head(
        "synthesis front-end fix only, behaviour-identical.\n"
        "//   Upstream mixes blocking (=) and non-blocking (<=) writes to the serving-consumer\n"
        "//   mask in one always block, which yosys-slang rejects (assert in procedural.cc).\n"
        "//   Blocking claims now go to a blocking shadow `csc`, non-blocking releases to\n"
        "//   `csc_clr`, and the register is written once with `csc & ~csc_clr`. NBAs land\n"
        "//   after all blocking updates of a time step, so this equals the upstream semantics."
    )
    s.rep("    reg [31:0] requests, coalesced;\n",
          "    reg [31:0] requests, coalesced;\n    reg [NUM_CONSUMERS-1:0] csc, csc_clr;\n")
    s.rep("            requests  = mem_requests;\n",
          "            csc       = channel_serving_consumer;\n"
          "            csc_clr   = {NUM_CONSUMERS{1'b0}};\n"
          "            requests  = mem_requests;\n")
    s.rep("channel_serving_consumer[j] = 1'b1;", "csc[j] = 1'b1;", None)
    s.rep("channel_serving_consumer[k] = 1'b1;", "csc[k] = 1'b1;", None)
    s.rep("!channel_serving_consumer[j]", "!csc[j]", None)
    s.rep("!channel_serving_consumer[k]", "!csc[k]", None)
    s.rep("channel_serving_consumer[j] <= 1'b0;", "csc_clr[j] = 1'b1;")
    s.rep("channel_serving_consumer[current_consumer[i]] <= 1'b0;", "csc_clr[current_consumer[i]] = 1'b1;")
    s.rep("            mem_requests           <= requests;\n",
          "            channel_serving_consumer <= csc & ~csc_clr;\n"
          "            mem_requests           <= requests;\n")
    return s.text


def lsu_mm() -> str:
    s = Src("lsu.sv")
    s.head(
        "module renamed lsu_mm; new input smem_ready. The shared-memory\n"
        "//   read/write WAITING states now wait for smem_ready (the SRAM-macro shared memory\n"
        "//   serves one lane per cycle) instead of completing unconditionally after one cycle."
    )
    s.rep("module lsu #(", "module lsu_mm #(")
    s.rep("    input  wire [WORD_BITS-1:0] smem_read_data,\n",
          "    input  wire [WORD_BITS-1:0] smem_read_data,\n    input  wire        smem_ready,\n")
    s.rep("                    WAITING: begin\n                        smem_read_enable <= 0;\n",
          "                    WAITING: if (smem_ready) begin\n                        smem_read_enable <= 0;\n")
    s.rep("                    WAITING: begin\n                        smem_write_enable <= 0;\n",
          "                    WAITING: if (smem_ready) begin\n                        smem_write_enable <= 0;\n")
    return s.text


def core_mm() -> str:
    s = Src("core.sv")
    s.head(
        "module renamed core_mm; instantiates fetcher_mm (SRAM I-cache),\n"
        "//   shared_memory_mm (SRAM shared memory, one lane per cycle) and lsu_mm (waits for\n"
        "//   smem_ready). Everything else is upstream."
    )
    s.rep("module core #(", "module core_mm #(")
    s.rep("    wire [WORD_BITS-1:0] smem_write_data [THREADS_PER_BLOCK-1:0];\n",
          "    wire [WORD_BITS-1:0] smem_write_data [THREADS_PER_BLOCK-1:0];\n"
          "    wire [THREADS_PER_BLOCK-1:0] smem_ready_v;\n")
    s.rep("    fetcher #(", "    fetcher_mm #(")
    s.rep("    shared_memory #(", "    shared_memory_mm #(")
    s.rep("        .lane_write_data(smem_write_data)\n",
          "        .lane_write_data(smem_write_data),\n        .lane_ready(smem_ready_v)\n")
    s.rep("            lsu #(", "            lsu_mm #(")
    s.rep("                .smem_read_data(smem_read_data[i]),\n",
          "                .smem_read_data(smem_read_data[i]),\n                .smem_ready(smem_ready_v[i]),\n")
    return s.text


def gpu_mm() -> str:
    s = Src("gpu.sv")
    s.head(
        "module renamed gpu_mm; instantiates core_mm and l1d_cache_mm\n"
        "//   (banked SRAM-macro L1-D). New parameter DCACHE_BANKS. Everything else is upstream."
    )
    s.rep("module gpu #(", "module gpu_mm #(")
    s.rep("    parameter DCACHE_LINES              = 0,\n",
          "    parameter DCACHE_LINES              = 0,\n    parameter DCACHE_BANKS              = 1,\n")
    s.rep("            l1d_cache #(", "            l1d_cache_mm #(")
    s.rep("                .LINES(DCACHE_LINES),\n",
          "                .LINES(DCACHE_LINES),\n                .BANKS(DCACHE_BANKS),\n")
    s.rep("            core #(", "            core_mm #(")
    return s.text


TARGETS = {
    "controller.sv": controller,
    "lsu_mm.sv": lsu_mm,
    "core_mm.sv": core_mm,
    "gpu_mm.sv": gpu_mm,
}


def main() -> int:
    check = "--check" in sys.argv[1:]
    if not UP.is_dir():
        print(f"upstream missing: {UP} (run scripts/run_tinygpu.sh fetch)", file=sys.stderr)
        return 2
    bad = 0
    OUT.mkdir(exist_ok=True)
    for name, fn in TARGETS.items():
        text = fn()
        path = OUT / name
        if check:
            if not path.exists() or path.read_text() != text:
                print(f"stale: {path}")
                bad += 1
        else:
            path.write_text(text)
            print(f"wrote {path.relative_to(HERE)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
