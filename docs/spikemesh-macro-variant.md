# SpikeMesh neurocore macro variant (sky130hd)

A second real-RTL macro-placement benchmark for MacroMind, next to the tiny-gpu
variant ([tinygpu-macro-variant.md](tinygpu-macro-variant.md)). It hardens one
neurocore of the user's neuromorphic chip
([Ashtonantony28/spikemesh](https://github.com/Ashtonantony28/spikemesh)) with
OpenROAD-flow-scripts on **sky130hd**, with its memories in **OpenRAM SRAM macros**.
The SpikeMesh repo is not modified: upstream is cloned (gitignored) into
`designs/spikemesh/upstream/` at the commit in `designs/spikemesh/UPSTREAM_PIN`
(`5517a40576293cb05113b17d60fb80a8d5fd103e`), and everything else lives here.

```bash
scripts/run_spikemesh.sh fetch              # clone upstream at the pin
scripts/run_spikemesh.sh sim                # upstream core cocotb tests on the variant (n128, n128_l2, big)
scripts/run_spikemesh.sh n128 synth         # Yosys only (yosys-slang front end)
scripts/run_spikemesh.sh n128 grt           # + floorplan, RTLMP macro placement, place, CTS, global route
pytest tests/test_spikemesh.py              # fast checks;  pytest -m eda tests/test_spikemesh.py  for sim + netlist
python -m macromind.harness sweep --design sky130hd/spikemesh_n128 --placer rtlmp sa grid --seeds 0 --stop-after grt
```

## 1. Why sky130 and these macros

The neurocore keeps every memory behind upstream's `sram_1r1w` (one write port, one
read port). Simulation showed it really does read and write in the same cycle
(spike accumulate and weight fetch overlap), so a 1RW macro such as NanGate45's
`fakeram45` would need a functional change to the core. The ORFS image ships
OpenRAM **dual-port** sky130 macros instead; `sky130_sram_1rw1r_64x256_8`
(64-bit word, 256 deep, byte write mask, 1040.76 x 403.035 um) fits the
neurocore's widest memories exactly. Port 0 is used for writes, port 1 for reads.

## 2. The variant

| File | What it is |
|---|---|
| `rtl/sram_1r1w.sv` | Drop-in replacement for upstream `rtl/core/sram_1r1w.sv` (same name, parameters, ports, behaviour). Maps a memory onto `DEPTH/256` macro banks when `DEPTH % 256 == 0`, `WIDTH <= 64` and its lanes are bytes (or it has one lane); otherwise it is the upstream flop array. `FORCE_FLOPS=1` turns mapping off. A hold register keeps `rdata` stable while `re` is low (the macro only drives `dout` in the cycle after a read). |
| `rtl/spikemesh_core_top.sv` | Wraps upstream `neurocore` with sizes from defines (`SM_N`, `SM_A`, `SM_FD`, `SM_LEARN`). |
| `sim/sky130_sram_1rw1r_64x256_8.v` | Quiet behavioural model of the macro for simulation: `dout` goes X at the clock edge and the read data arrives after the falling edge, like the OpenRAM model, so a design that sampled it early would fail. |
| `sim/run_sim.py` | Runs **upstream's own** `tests/core/core_tb.py` (with its golden model) on upstream `neurocore.sv` + `learn_engine.sv` + the replacement wrapper + the macro model. |
| `config.mk`, `configs.mk`, `constraint.sdc`, `files.f` | ORFS design (sky130hd, slang front end, 25 ns clock). |
| `macromind/harness/designs/sky130hd/spikemesh_n128/config.mk` | Harness adapter (`--design sky130hd/spikemesh_n128`). |

Config `n128` (128 neurons, 128 axons, 256-entry fan-out table, no learning) maps
**10 macros**: 8 for the 128 x 128 int8 weight memory (2048 rows of 64 bits) and
one each for the fan-out destination and axon tables. Neuron state (16-bit x 128)
stays in flops. Die 3000 x 3000 um.

### Simulation (upstream tests, macro variant)

| Config | Macros | Tests | Result |
|---|---|---|---|
| `n128` | 10 | `test_random` x2 (mixed, saturating), `test_cfg_readback`, `test_cycles` | 4/4 pass |
| `n128_l2` | learning on | `test_learn` (R-STDP) | pass |
| `big` (N = A = 256, FD = 1024) | 40 | `test_random` (sparse) | pass |

Three mutants of the wrapper show the macro path is really exercised: reading the
bank selected by the *current* read address instead of the registered one, writing
all bytes regardless of the lane mask, and ignoring the write bank select. Each one
fails 3 of the 4 `n128` tests (`test_cycles` only checks timing, so it passes).
Dropping the hold register does **not** fail any test: the neurocore always consumes
read data in the cycle after `re`. It is kept anyway, because upstream's
`sram_1r1w` contract says read data is held.

## 3. ORFS results, `n128`

Stock ORFS (`scripts/run_spikemesh.sh n128 grt`, RTLMP): synthesis 1 min,
floorplan + RTLMP about 5 min, through global route about 15 min with 4 cores.
The flow emits `MPL-0002` warnings (some macro pins are off the routing track
grid, a known property of the OpenRAM sky130 LEFs), but global routing finishes
with **0 overflow**.

Harness comparison (`--stop-after grt`, 2 cores per job, seed 0):

| Placer | Seed | Timing-driven GPL | Proxy cost | GRT wirelength (um) | vs RTLMP | WNS (ns) | Power (W) | GRT overflow |
|---|---|---|---|---|---|---|---|---|
| RTLMP (ORFS default) | 0 | on | 0.4186 | **3,769,580** | - | 4.62 | 0.1660 | 0 |
| MacroMind grid | 0 | on | 0.4440 | 4,099,883 | +8.8 % | 4.75 | 0.1657 | 0 |
| MacroMind SA | 1 | on | 0.4547 | 4,522,853 | +20.0 % | 4.97 | 0.1709 | 0 |
| MacroMind SA | 0 | on | 0.4336 | OpenROAD crash | - | - | - | - |
| RTLMP | 0 | off | 0.4186 | **3,849,261** | - | 3.81 | 0.1682 | 0 |
| MacroMind SA | 0 | off | 0.4336 | 4,354,824 | +13.1 % | -0.04 | 0.1708 | 0 |

**RTLMP wins on this design.** Every MacroMind placement routes without overflow,
but with 8 to 20 % more global-route wirelength. MacroMind's proxy cost ranks
RTLMP first too, so the gap is in the search, not in a cost model that disagrees
with the router: 10 large macros on a 3 mm die leave few good arrangements
(the weight banks want to sit together beside the datapath), and RTLMP's
dataflow-aware clustering finds one that SA and grid do not.

SA seed 0 makes OpenROAD 3_3 global placement crash with signal 11 inside
timing-driven repair (`rsz::Rebuffer::recoverArea`, from `gpl::TimingBase`), on the
first run and again on resume, so it is deterministic for that macro placement. It is
an OpenROAD bug, not an illegal placement: the same placement goes through with
`--set GPL_TIMING_DRIVEN=0`, and RTLMP was rerun the same way for a like-for-like pair.
Those two rows are tagged `notd` in `runs/harness`.


## 4. Caveats

* Global route only: detailed route, DRC and signoff timing have not been run.
  The 25 ns clock is loose on purpose (OpenRAM sky130 macros are slow); timing
  closes with several ns of slack under every placer, so this design separates
  placers on wirelength and congestion, not timing.
* Only one neurocore. SpikeMesh's mesh router and multi-core top are not included.
* `n128_l2` has an ORFS config but has not been run through the flow.
