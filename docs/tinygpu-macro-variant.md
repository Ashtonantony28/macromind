# tiny-gpu-extended macro variant (SG9)

A macro-heavy, synthesisable variant of the user's GPU
([Ashtonantony28/Didactic_GPU_Simulator](https://github.com/Ashtonantony28/Didactic_GPU_Simulator),
titled *tiny-gpu-extended*), hardened with OpenROAD-flow-scripts on NanGate45 with
fakeram45 SRAM macros. It is a macro-placement benchmark for MacroMind. The user's
repo is not modified: upstream is cloned (gitignored) into `designs/tinygpu/upstream/`
at the commit in `designs/tinygpu/UPSTREAM_PIN`
(`096c6ef586ab08bd1f641f33cf4b2808d4e41444`, 2026-11-03), and everything else lives here.

```bash
scripts/run_tinygpu.sh fetch          # clone upstream at the pin, check derived RTL is current
scripts/run_tinygpu.sh sim            # RTL sim: upstream gpu vs macro variant vs upstream ISA model
scripts/run_tinygpu.sh c4 synth       # Yosys only (yosys-slang front end)
scripts/run_tinygpu.sh c4 grt         # + floorplan, RTLMP macro placement, place, CTS, global route
scripts/run_tinygpu.sh c4             # full flow (finish)
pytest tests/test_tinygpu.py          # fast checks;  pytest -m eda tests/test_tinygpu.py  for elaboration + sim
make -C $FLOW_HOME DESIGN_CONFIG=$PWD/designs/tinygpu/config.mk TG_CONFIG=c8 floorplan   # plain ORFS
```

Runs go to `runs/orfs/tinygpu/` (`WORK_HOME`), `NUM_CORES` defaults to 2;
`runs/orfs/tinygpu/tinygpu_<cfg>.metrics.json` holds the summary written by
`designs/tinygpu/report.py`.

## 1. Getting the RTL through Yosys

* **Front end: yosys-slang**, which ships in the ORFS image's Yosys 0.55 (`slang.so`
  plugin, ORFS `SYNTH_HDL_FRONTEND=slang`). sv2v was not needed (no binary is
  installed and GitHub release downloads are blocked); Yosys's native
  `read_verilog -sv` cannot parse upstream's unpacked-array ports.
* Upstream needed two front-end workarounds, both outside `upstream/`:
  1. `src/controller.sv` mixes blocking and non-blocking writes to one variable in one
     `always` block; yosys-slang aborts (`procedural.cc:191` assert).
     `rtl/controller.sv` is derived by `derive_rtl.py` with a behaviour-identical
     rewrite (blocking shadow + clear mask, one NBA at the end).
  2. `src/simt_stack.sv` uses `$fatal` inside a clocked block, which yosys-slang does
     not implement. It is only instantiated when `SIMT_STACK=1` (default 0), so it is
     simply left out of `files.f`.
* `rtl/tinygpu_top.sv` wraps the GPU with packed ports (unpacked-array top ports are
  awkward for ORFS) and leaves the bench telemetry outputs unconnected (their
  counters are optimised away).

**Baseline** (upstream default config: 8-bit words, 2 cores x 4 threads, no caches,
flop shared memory; `TG_CONFIG=ff2`, 5 ns clock, ORFS synthesis, NanGate45):
**61,547 cells, 88,592 um^2** (16 % sequential), 0 macros, 9 min synthesis.
About 80 % of that is the per-lane ALUs: upstream's ALU has a single-cycle
combinational divider and multiplier.

## 2. The macro variant

All macro configs use the transformer (xf) datapath: 32-bit words, 4096-word data
memory, 4 data channels, 4 threads per core, `ICACHE_LINES=64`, 64-word shared memory
per core, and an L1-D of `TG_DCACHE_BANKS` x 256 lines. They also use upstream's
multi-warp mode `WARPS_PER_CORE=4` (4 warps of 1 lane, ADR-012/014): the 4 threads of
a core time-share one ALU. This is an upstream feature, not a change, and it keeps the
number of 32-bit dividers equal to `NUM_CORES`. With one ALU per thread (upstream's
default 1 warp x 4 lanes) the 8-core ORFS synthesis was killed by the OOM killer in
techmap at 11.5 GB after 5.5 min, so it does not fit this 15 GB container.

| Memory | Upstream | Macro variant | Macro | Count |
|---|---|---|---|---|
| Shared memory (per core) | 64 x WORD flops, 4 comb. read + 4 write ports | `rtl/shared_memory_mm.sv`: one 1RW SRAM, lanes served one per cycle, 64-bit "written" vector emulates reset-to-zero | `fakeram45_64x32` | NUM_CORES |
| I-cache (per core) | valid/tag/data flop arrays | `rtl/fetcher_mm.sv`: {tag(2), instr(16)} in SRAM, valid bits in flops | `fakeram45_64x21` | NUM_CORES |
| L1-D (per GPU) | 1 direct-mapped flop cache, one lookup port per lane | `rtl/l1d_cache_mm.sv`: address-interleaved banks, {tag, data} in SRAM, per-bank round-robin arbiter, valid bits in flops | `fakeram45_256x34` | TG_DCACHE_BANKS |
| Register files | flops | unchanged (flops) | - | - |

`rtl/mm_sram.sv` maps a (depth, width) to the smallest fakeram45 macro with that depth
(falls back to flops if none fits); `sim/fakeram45_models.v` are behavioural models of
the macros for simulation only.

| Config | NUM_CORES | L1-D banks | 64x32 | 64x21 | 256x34 | **Macros** | Macro area (um^2) |
|---|---|---|---|---|---|---|---|
| `c4` | 4 | 4 (1024 lines) | 4 | 4 | 4 | **12** | 34,664 |
| `c8` | 8 | 4 (1024 lines) | 8 | 8 | 4 | **20** | 43,424 |
| `c16` | 16 | 8 (2048 lines) | 16 | 16 | 8 | **40** | 86,848 |

### What changed vs upstream (files)

| File | Kind | Change |
|---|---|---|
| `rtl/controller.sv` | derived | front-end fix only (above); behaviour-identical |
| `rtl/lsu_mm.sv` | derived | shared-memory `WAITING` waits for a new `smem_ready` input |
| `rtl/core_mm.sv` | derived | instantiates `fetcher_mm`, `shared_memory_mm`, `lsu_mm` |
| `rtl/gpu_mm.sv` | derived | instantiates `core_mm`, `l1d_cache_mm`; new `DCACHE_BANKS` |
| `rtl/fetcher_mm.sv` | copy + rewrite | I-cache branch rewritten around `mm_sram`; no-cache branch unchanged |
| `rtl/shared_memory_mm.sv`, `rtl/l1d_cache_mm.sv`, `rtl/mm_sram.sv`, `rtl/tinygpu_top.sv` | new | as above |

"Derived" files are generated by `designs/tinygpu/derive_rtl.py` from upstream with
asserted text substitutions (`--check` verifies they are current; a test runs it).

### Behavioural differences (timing, not results)

* **Synchronous SRAM reads.** Upstream shared memory reads combinationally; the LSU
  already took REQUESTING -> WAITING -> DONE, so the variant only extends WAITING until
  the arbiter acknowledges (1 cycle per lane, lanes in ascending order). Same-address
  writes in one `STS` still end with the highest lane's value. The I-cache adds one
  lookup cycle per fetch; the L1-D adds a bank-arbitration + SRAM cycle per lookup and
  a bank slot for each fill.
* **Reset of SRAM contents.** Upstream clears the shared memory on every per-block
  reset; the variant keeps a valid-bit vector so unwritten words still read 0.
* **L1-D writes** are write-through + write-invalidate (upstream: update-on-hit).
  Memory contents are identical; hit/miss telemetry differs. As upstream, a fill racing
  a different lane's write to the same line is not ordered (a data race in the kernel).
* **I-cache valid bits** are cleared on the global (telemetry) reset, which is what the
  upstream comment intends. In upstream the wipe sits behind `if (reset)` for the per-core
  reset, which `dispatch` holds high during a global reset, so upstream's valid bits are
  only initialised through X-semantics of the first reset cycle (an Icarus event sim
  happens to take the wipe branch; a Yosys-written netlist sim does not and goes X).
* **Cycle counts** change (more cycles per memory op; 4 warps share an ALU); kernel
  results do not.

### Would upstream's cocotb tests pass?

Not run here (no cocotb; upstream's Makefile also needs sv2v). Result-checking tests
(matmul, transpose, xf kernels) should pass, since outputs are unchanged; tests that
assert exact cycle counts, I-cache/L1-D hit counts (`test_l1d_indepth`, `test_medic`,
bench JSON comparisons) or the one-cycle SMEM latency would fail. Instead,
`sim/run_sim.py` checks results directly: it builds memory images with upstream's
`xf` toolchain, runs upstream's ISA model (`xf/isasim.py`) as golden, and simulates
`tinygpu_top` in Icarus (RTL written out by yosys-slang + Yosys) for MACROS=0 (upstream
gpu) and MACROS=1. All final memories match word for word:

| Kernel | threads | upstream gpu, 1 warp | macro variant, 4 warps | macro variant, 1 warp (SMEM arbiter) |
|---|---|---|---|---|
| matmul_tiled 8x8x16 (LDS/STS/BAR) | 32 | match, 15,218 cyc (4 cores) | match, 30,031 cyc (4 cores) | match, 16,599 cyc (2 cores); 9,714 (8 cores) |
| matmul_naive 4x8x8 | 32 | match, 4,884 | match, 9,343 | match, 4,511 (2 cores) |
| transpose 6x10 | 60 | match, 2,167 | match, 4,840 | match, 2,330 (2 cores) |

## 3. ORFS results

All runs: ORFS `v3.0-3468-gc673f615`, NanGate45, `NUM_CORES=2` (make/OpenROAD threads),
shared 4-core machine (other agents' ORFS jobs ran concurrently, so runtimes are
inflated), yosys-slang front end, flat synthesis, ABC speed script.
Clock: 15 ns for the macro configs (the single-cycle 32-bit divider; a first c4
floorplan at 5 ns had WNS -10.25 ns), 5 ns for the 8-bit baseline.
`CORE_UTILIZATION` 35 %, `PLACE_DENSITY` 0.60, `MACRO_PLACE_HALO` 8 um, default
macro placer (hier-RTLMP, `2_2_floorplan_macro`).

### Synthesis (ORFS `1_synth.v`, `reports/.../synth_stat.txt`)

| Config | Cells (incl. macros) | Area um^2 (incl. macros) | Sequential share | Macros in netlist | Synth time | Yosys peak RSS |
|---|---|---|---|---|---|---|
| `ff2` (upstream default, 5 ns) | 61,547 | 88,592 | 16 % | 0 | 8.8 min | 1.9 GB |
| `c4` | 133,987 | 261,680 | 29 % | 12 (4+4+4) | 5.7 min | 1.4 GB |
| `c8` | 260,120 | 481,412 | 31 % | 20 (8+8+4) | 12.1 min | 2.7 GB |
| `c16` | 542,699 | 989,076 | 30 % | 40 (16+16+8) | 26.8 min | 5.3 GB |

The macro counts are what `tests/test_tinygpu.py` asserts (elaborated with yosys-slang
for every config, and in the ORFS netlists when present).

### Physical design, `c4` (12 macros)

| Metric | Value |
|---|---|
| Die / core | 872.7 x 872.7 um (761,553 um^2), core util 38.1 % after GRT (std cells 35.1 %) |
| Std cells after GRT | 143,542 (249,809 um^2); macros 12 (34,664 um^2) |
| Setup WNS / TNS after GRT (15 ns) | -0.006 ns / -0.006 ns |
| GRT wirelength | 3,684,708 um; 0 antenna violations, 0 diodes |
| Power (GRT estimate) | 5.43 W |
| Stage runtimes | synth 341 s, floorplan 514 s (timing repair), RTLMP macro placement 444 s, global place 742 s, resize 89 s, detailed place 119 s, CTS 173 s, GRT 845 s; **~56 min** synth to GRT |
| Detailed route (TritonRoute, 2 threads) | **0 DRC** after 6 iterations (25,049 -> 0), wirelength **2,850,630 um**, 1,018,893 vias, 0 antenna violations; 2,573 s (43 min) |
| Total, synth to detailed route | ~99 min wall (on a shared machine) |

Post-route STA (`finish`) was not run; the timing numbers are the GRT estimates.

RTLMP placement of c4 (lower-left corner, um): the macros are pushed to the die
boundary, mostly the left edge and the top edge, with the per-core SMEM/I-cache pairs
not always kept together (core 1's I-cache ends at the bottom right, far from its
SMEM on the left) - a placement a learned placer could plausibly improve on.

| Macro | x, y | Macro | x, y |
|---|---|---|---|
| core0 I$ 64x21 | 12, 234 | core0 SMEM 64x32 | 12, 388 |
| core1 I$ 64x21 | 736, 12 | core1 SMEM 64x32 | 12, 311 |
| core2 I$ 64x21 | 12, 689 | core2 SMEM 64x32 | 48, 388 |
| core3 I$ 64x21 | 547, 708 | core3 SMEM 64x32 | 547, 631 |
| L1-D bank0 256x34 | 433, 713 | L1-D bank1 256x34 | 433, 794 |
| L1-D bank2 256x34 | 224, 12 | L1-D bank3 256x34 | 433, 631 |

`c8` and `c16` were synthesised but not placed: from c4's 56 min to GRT at 134 k
cells, c8 (260 k) would take roughly 2 h and c16 (543 k) 4-5 h at 2 threads.


## 4. Caveats and what is incomplete

* **Not the user's GPU verbatim.** Results-equivalent on the kernels tested, but cycle
  timing differs (synchronous SRAMs, arbitration, 4 warps sharing one ALU), the L1-D is
  banked and write-invalidate, and telemetry outputs are dropped. Upstream's cocotb
  suite was not run.
* **Register files stay in flops** (4 x 13 x 32 bits per core, ~30 % of cell area is
  sequential). Mapping them to 2R1W SRAM would need duplicated macros or a 2-cycle
  read; not done.
* **Std-cell heavy.** Macros are 13 % of c4's cell area (9 % of c16's): the 32-bit
  combinational divider/multiplier per core, the per-thread register files and the
  data-memory controller (whose arbiter scans all NUM_CORES x 4 consumers) dominate.
  This makes the benchmark timing- and wirelength-relevant but macro-light by area.
* **Only c4 was placed and routed** (no `finish`/post-route STA). c8/c16 are synthesised (macro counts verified);
  floorplan/RTLMP/GRT for them is scripted (`scripts/run_tinygpu.sh c8 grt`) but needs
  ~2-5 h at 2 threads.
* **Timing**: the 15 ns clock is set from the observed critical path, not a target; IO
  delays are a flat 20 % of the period. No IO placement constraints (ORFS random IO).
* **sim coverage**: three kernels (tiled matmul with LDS/STS/BAR, naive matmul,
  transpose) at 2, 4 and 8 cores; not MeDiC, COALESCE, PIPELINED or SIMT_STACK. The
  upstream gpu reference is simulated without I-cache (see the valid-bit note) and with
  a 16-line L1-D for speed.
* The upstream clone is gitignored; `scripts/run_tinygpu.sh fetch` re-clones it at the
  pinned commit and checks the derived RTL is current.

