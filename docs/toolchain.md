# EDA toolchain (SG1)

Status as of 2026-11-06, in the cloud container (Ubuntu 24.04.5, 4 cores, 15 GB RAM, no GPU).

## TL;DR

```bash
scripts/setup_toolchain.sh          # idempotent; 1-2 min first run, <1 s after
source scripts/env.sh               # PATH, FLOW_HOME, OPENROAD_EXE, YOSYS_EXE ...
scripts/smoke_orfs.sh gcd           # full ORFS flow, prints metrics; 72 s @4 cores, 122 s @2
NUM_CORES=2 scripts/smoke_orfs.sh tinyRocket   # 2 fakeram45 macros, full flow; ~20 min @2 cores
openroad -python -exit scripts/odb_probe.py <file.odb>   # odb Python API
```

Everything is installed under `tools/` (gitignored); runs go to `runs/` (gitignored).

## What is installed

| Component | Version | Where |
|---|---|---|
| ORFS flow (Makefile, scripts, platforms, designs) | image tag `openroad/orfs:v3.0-3468-gc673f615` (ORFS commit `c673f615`, image built 2025-07-20) | `tools/ORFS/flow` (= `$FLOW_HOME`) |
| OpenROAD | the build shipped in that image (`-version` prints `HEAD-HASH-NOTFOUND`; features `+GPU +GUI +Python`), odb schema 0.109 | `tools/ORFS/tools/install/OpenROAD/bin/openroad`, wrapper `tools/bin/openroad` |
| Yosys | 0.55 | `tools/ORFS/tools/install/yosys/bin/yosys` |
| kepler-formal | as shipped | `tools/ORFS/tools/install/kepler-formal` |
| Bundled libs | `libortools.so.9` (OR-Tools 9), `libpython3.10.so.1.0` + CPython 3.10 stdlib | `tools/ORFS/tools/lib`, `tools/ORFS/tools/py310` |
| Host apt packages | `tcl8.6 tcl-tclreadline libqt5{core,gui,widgets,charts}5 klayout (0.28.16) time make` | system |

The binaries were built on Ubuntu 22.04 (glibc 2.35), so they run natively on the
24.04 host once the libraries above are present. No docker is needed at run time.
Platforms available: nangate45, asap7, sky130hd/hs, gf180, ihp-sg13g2 (+ sky130io/ram).

### How it was obtained, and what failed

| Attempt | Result |
|---|---|
| Docker Hub `openroad/orfs` | 429 rate limit (as reported by the lead). |
| GitHub release assets (`Precision-Innovations/OpenROAD`, `The-OpenROAD-Project/OpenROAD` .debs) | **Blocked**: the session proxy serves anonymous `git clone/fetch` of public repos, but github.com web/API/release downloads return 403 "GitHub access to this repository is not enabled for this session". `add_repo` read access confirms only git reads are served. |
| conda `litex-hub` channel | Reachable (`conda.anaconda.org` works) but newest openroad there is `2.0_12381` from **Feb 2024** and yosys 0.38; `micro.mamba.pm` and `conda-forge.org` are blocked (403), so installing needs a manual micromamba bootstrap. Kept as a fallback, not used. |
| ghcr.io, quay.io, public.ecr.aws, prefix.dev | ghcr reachable but no ORFS image known there; quay/prefix blocked; ECR needs auth. |
| **`mirror.gcr.io/openroad/orfs`** (Google's Docker Hub pull-through cache) | **Works**, anonymously. Only cached tags are available: `latest` (manifest 404), `26Q4-229-g64939f371`, `26Q4-171-…`, `26Q4-71-…`, `26Q3-651-…`, `v3.0-3468-gc673f615`, `v3.0-2703-g4ef9e9aa`. |
| Newest image `26Q4-229-g64939f371` (Ubuntu 26.04, Yosys 0.68, GCC 15) | gcd routed fine *inside docker* (71 s), **but its OpenROAD is built without Python** (`-python` flag absent, no `PyInit_*` symbols), and its glibc/libstdc++ are too new to run on the 24.04 host. Rejected because the placer needs `openroad -python` + `odb`. |
| **`v3.0-3468-gc673f615`** (Ubuntu 22.04, Python-enabled) | **Chosen.** Binaries + flow extracted to `tools/ORFS`, run natively. |
| Build from source | Not attempted; not needed. |

`setup_toolchain.sh` uses `docker pull` + `docker cp` when a daemon is available
(it starts `dockerd` if root), otherwise `scripts/_oci_extract.py` pulls the image
layers straight from the registry HTTP API and extracts only the needed paths (tested:
1.5 GB in ~77 s, byte-identical to the docker extraction). The docker image is deleted
afterwards unless `KEEP_IMAGE=1`. Override `ORFS_TAG=` / `ORFS_REGISTRY=` to change.

Gotcha fixed in the wrapper: `openroad -python` needs `PYTHONHOME=tools/ORFS/tools/py310`,
but if that variable leaks into the ORFS make run, the host `python3` children that
ORFS spawns die with "No module named 'encodings'". `tools/bin/openroad` therefore sets
`PYTHONHOME` only when `-python` is on the command line.

## openroad -python and odb (for the placer)

Exact invocation (works, tested on gcd and tinyRocket ODBs):

```bash
source scripts/env.sh
openroad -python -exit scripts/odb_probe.py runs/orfs/smoke/results/nangate45/gcd/base/6_final.odb
#   (optional) add -no_splash; extra args after the script land in sys.argv
```

Inside the script both APIs work:

```python
import odb
db = odb.dbDatabase.create(); odb.read_db(db, path); blk = db.getChip().getBlock()
from openroad import Design, Tech
d = Design(Tech()); d.readDb(path); d.evalTclString("...")   # Tcl from Python
```

Caveats:
- The embedded interpreter is **CPython 3.10** with only the stdlib (no numpy, no torch).
  The 3.13 project venv cannot `import odb`. Plan: the macromind process (venv, torch)
  calls `openroad -python -exit <adapter>.py` as a subprocess and exchanges JSON/NPZ via
  files, or uses Tcl + DEF. Do not try to load venv packages into OpenROAD's 3.10.
- ODB files are schema-versioned: an ODB written by a newer OpenROAD (e.g. the 26Q4 image,
  schema 0.141) cannot be read by this one (schema 0.109): "incompatible database schema
  revision". Always produce ODBs with the same binary.
- swig prints harmless "memory leak of type 'odb::Rect *'" warnings at exit.

## Smoke results

### nangate45/gcd (full flow) — PASS

`scripts/smoke_orfs.sh gcd`:

```
== ORFS nangate45/gcd target=finish variant=base cores=4
== make exit code: 0, wall time: 72 s
== metrics
  stdcell instances              665
  macros                         0
  core util                      0.756019
  routed wirelength (um)         4381
  vias                           3723
  DRC errors (detailed route)    0
  antenna violating nets         0
  setup WNS (ns)                 -0.0510056
  setup TNS (ns)                 -0.886043
  hold WNS (ns)                  0.0540008
  total power (W)                0.0034374
  die area (um^2)                1323.5
== stage runtimes (s)
  1_1_yosys_canonicalize            0.2
  1_2_yosys                         0.8
  2_1_floorplan                     0.5
  2_2_floorplan_macro               0.4
  2_3_floorplan_tapcell             0.4
  2_4_floorplan_pdn                 0.4
  3_1_place_gp_skip_io              0.5
  3_2_place_iop                     0.4
  3_3_place_gp                      1.7
  3_4_place_resized                 0.6
  3_5_place_dp                      0.7
  4_1_cts                           4.8
  5_1_grt                          17.7
  5_2_route                        25.0
  5_3_fillcell                      0.4
  6_1_fill                          0.4
  6_1_merge                         1.4
  6_report                          2.1
  sum of stages                    58.3   (make wall 72 s)
== RESULT: PASS (routed, 0 DRC)
```

That run used 4 cores. Re-run from a clean shell (`env -i`, then `source scripts/env.sh`)
with `NUM_CORES=2` (another agent was training): identical QoR (WL 4381, 0 DRC, WNS -0.051),
make wall 122 s; stages 1_2_yosys 15.9, 4_1_cts 3.3, 5_1_grt 16.2, 5_2_route 38.0 s.
QoR is bit-for-bit repeatable across thread counts for gcd.

Peak memory 1.1 GB (detailed route). The small negative WNS is ORFS's own gcd
constraint (the 26Q4 image shows -0.16 ns too); not a toolchain problem.

### nangate45/tinyRocket (2 fakeram45_64x32 macros, full flow) — PASS

`NUM_CORES=2 scripts/smoke_orfs.sh tinyRocket` (clean shell; a concurrent torch job held the other 2 cores):

```
== ORFS nangate45/tinyRocket target=finish variant=base cores=2
== make exit code: 0, wall time: 1212 s
== metrics
  stdcell instances              29662
  macros                         2
  core util                      0.300129
  routed wirelength (um)         637956
  vias                           205869
  DRC errors (detailed route)    0
  antenna violating nets         0
  setup WNS (ns)                 -0.097767
  setup TNS (ns)                 -26.6243
  hold WNS (ns)                  0.0556725
  total power (W)                0.119474
  die area (um^2)                212205
  GRT antenna diodes             0
== stage runtimes (s)
  1_1_yosys_canonicalize            1.4
  1_2_yosys                        69.8
  2_1_floorplan                    96.5
  2_2_floorplan_macro              30.8
  2_3_floorplan_tapcell             0.5
  2_4_floorplan_pdn                 1.0
  3_1_place_gp_skip_io              5.3
  3_2_place_iop                     0.5
  3_3_place_gp                     98.4
  3_4_place_resized                22.4
  3_5_place_dp                     23.6
  4_1_cts                          82.3
  5_1_grt                         258.1
  5_2_route                       444.7
  5_3_fillcell                      1.6
  6_1_fill                          0.7
  6_1_merge                         3.4
  6_report                         57.6
  sum of stages                  1198.6   (make wall 1212 s)
== RESULT: PASS (routed, 0 DRC)
```

Peak memory 1.6 GB (detailed route). odb probe of the final ODB:

```
$ openroad -python -exit scripts/odb_probe.py runs/orfs/smoke/results/nangate45/tinyRocket/base/6_final.odb
block=RocketTile dbu/um=2000
die=(0.0,0.0)-(424.92,499.4) um core=(10.07,9.8)-(414.77,488.6) um
insts=79093 nets=32962 macros=2 bterms=269
  dcache/data.data_arrays_0.data_arrays_0_ext.mem master=fakeram45_64x32 loc=(32.47,25.025) size=(20.14x61.6) orient=R180 status=LOCKED
  frontend/icache.data_arrays_0.data_arrays_0_0_ext.mem master=fakeram45_64x32 loc=(256.85,411.915) size=(20.14x61.6) orient=MY status=LOCKED
```

tinyRocket is the recommended fast end-to-end macro test case (~20 min, 2 macros; too few
macros to be interesting for RL, but it exercises the whole macro path).

### nangate45/ariane133 (floorplan + RTLMP macro placement) — PASS

`NUM_CORES=2` (synthesis ran earlier with 4 cores, partly contended):

| Stage | Wall | Peak mem |
|---|---|---|
| 1_2_yosys (SYNTH_HIERARCHICAL) | 696 s | — |
| 2_1_floorplan | 53 s | 0.9 GB |
| 2_2_floorplan_macro (hier-RTLMP) | 563 s | 1.35 GB |
| 2_3 tapcell + 2_4 pdn | 11 s | — |

The netlist has **132** `fakeram45_256x16` macros (the canonical RTL has 133; synthesis
removes one), 167,797 instances, 196,660 nets; die 1500 x 1500 um.

Continued through placement (`scripts/smoke_orfs.sh ariane133 place`, CLEAN=0, NUM_CORES=2):

| Stage | Wall | Peak mem |
|---|---|---|
| 3_1_place_gp_skip_io | 65 s | 0.7 GB |
| 3_3_place_gp (timing-driven) | 1080 s | 2.2 GB |
| 3_4_place_resized | 114 s | 1.0 GB |
| 3_5_place_dp | 145 s | 2.1 GB |

So synth to detailed placement is ~45 min at 2 threads (12 min synth + 10.5 min floorplan/RTLMP
+ 23 min placement); 178,366 std cells after placement. CTS, global and detailed route were
not run here (estimated another 2-4 h from the tinyRocket scaling, where GRT+DRT were 60% of
the flow at 1/14 of ariane's routed WL); memory stayed far below 15 GB so it is feasible but slow.


## ORFS designs with macros (nangate45 / asap7)

Macro counts were measured by running ORFS's RTL canonicalisation (`do-yosys-canonicalize`)
per design, then `yosys: hierarchy -top <top>; flatten; stat` and counting macro cell types.
"stdcells" is the post-placeopt standard-cell count from each design's ORFS
`rules-base.json`, a good proxy for runtime. Runtime estimates are for this container at
NUM_CORES=2..4, scaled from the measured gcd / tinyRocket / ariane133 runs (detailed route
scales roughly linearly-to-superlinearly with routed WL; RTLMP with macro count).

| Design | Macros (measured) | Macro types | Std cells | Feasible here (4 cores / 15 GB)? |
|---|---|---|---|---|
| nangate45/tinyRocket | 2 (ODB verified) | 2x 64x32 | 33k | **Yes, run: full flow 20 min @2 cores**, 1.6 GB |
| nangate45/bp_fe_top | 11 | 9x 512x64, 64x7, 64x96 | 40k | Yes. Est. 30-60 min full flow, ~3 GB. **Best small RL target** |
| nangate45/bp_be_top | 10 | 8x 512x64, 64x15, 64x96 | 63k | Yes. Est. 45-90 min full flow, ~4 GB |
| nangate45/bp_multi_top | 26 | 17x 512x64, 256x96, 2x 32x64, 64x15, 64x7, 4x 64x96 | 144k | Marginal. Est. 2-3 h full flow, ~6-8 GB. Floorplan/GRT-only labels OK |
| nangate45/swerv_wrapper | 28 | 8x 2048x39, 16x 256x34, 4x 64x21 | 113k | Marginal. Est. 2-3 h full flow, ~6 GB |
| nangate45/black_parrot | 24 | 17x 512x64, 256x95, 64x15, 64x7, 4x 64x96 | 281k | Floorplan/place yes; full flow est. 4-6 h, ~10 GB: one-off only |
| nangate45/ariane133 | **132** in netlist (133 in RTL) (ODB verified) | 256x16 | 192k | **Run through detailed place: 45 min @2 threads, peak 2.2 GB** (RTLMP 9.4 min). Full flow est. 3-5 h: occasional evaluation only, not data generation |
| nangate45/ariane136 | 136 | 256x16 | 197k | Same as ariane133 |
| nangate45/mempool_group | 0 at RTL level (its SRAMs are inferred memories mapped during synthesis, count not verified) | 256x32, 128x256, 128x32, 64x64 LEFs | 163k | Marginal (est. 3-4 h) |
| nangate45/bp_quad | 220 | 128x 512x64, 32x 32x32, 32x 64x62, 16x 64x124, 8x 128x116, 4x 256x48 | ~600k cells at RTL; die 3600x3600 um | **No**: est. >12 h and >15 GB for routing. Script only |
| asap7/riscv32i | 4 | 4x fakeram7_256x32 | 12k | Yes. Est. 10-20 min. Good ASAP7 smoke test |
| asap7/riscv32i-mock-sram | 4 | 4x mock 256x32 SRAM (built as a sub-block) | 12k | Yes (est. 15-30 min; builds the mock SRAM first) |
| asap7/cva6 | 8 | 2x each of 128x64, 64x25, 64x256, 64x28 | 136k | Marginal. Est. 2-4 h |
| asap7/swerv_wrapper | 28 | 8x 2048x39, 16x 256x34, 4x 64x21 | 168k | Marginal. Est. 3-4 h |
| asap7/mock-array, asap7/aes-block | hierarchical `BLOCKS` (sub-blocks hardened then placed as macros; mock-array ships `macro-placement.tcl`) | n/a | 12k / 11k top | Yes; useful for testing hierarchical macro flows |

Without macros (std-cell only): nangate45 aes, dynamic_node, gcd, ibex, jpeg, swerv; asap7 aes*,
ethmac*, gcd*, ibex, jpeg*, minimal, mock-alu, mock-cpu, uart.

**Recommendation for MacroMind on this container:** iterate on tinyRocket (2), bp_fe_top (11),
bp_be_top (10) and asap7/riscv32i (4) end to end; use ariane133 at floorplan / global-place
(and GRT) level for proxy-cost studies and run its full route only as a rare evaluation;
leave bp_quad, black_parrot full flows for a bigger machine. RTLMP on ariane133 alone is 9.4
min at 2 threads, so the "hier-RTLMP baseline" on ariane costs ~10 min per seed here.


Only tinyRocket, ariane133 (to detailed place) and gcd were actually run; the other rows'
runtimes are estimates. bp_fe_top is the next design to time.

## Idempotency of setup_toolchain.sh (tested)

Into a scratch copy of the repo:
- fresh install via docker path: 58 s (image already cached by docker) ; second run: 0.4 s, prints
  "already installed ... (use --force to redo)" and only re-checks versions;
- `--force` with docker unavailable (registry-API fallback, `_oci_extract.py`): 1 min 52 s;
- both produced trees identical to `tools/ORFS` (`diff -rq` clean apart from two dangling symlinks
  in the python3.10 stdlib that are dangling in the image too).
- On the real repo it reports "already installed" (stamp `tools/ORFS/.installed-<tag>`).

Disk: `tools/` is 1.4 GB; a full ariane133 run directory is a few GB under `runs/`.

## Using the flow from MacroMind code

- `source scripts/env.sh` exports `MACROMIND_ROOT ORFS_ROOT FLOW_HOME OPENROAD_EXE OPENSTA_EXE
  YOSYS_EXE NUM_CORES` and puts `tools/bin` (openroad wrapper) and yosys on PATH.
- Run ORFS: `make -C $FLOW_HOME DESIGN_CONFIG=designs/nangate45/<d>/config.mk WORK_HOME=<dir> NUM_CORES=2 <target>`
  (targets: synth, floorplan, place, cts, grt, route, finish). Per-stage metrics land in
  `<WORK_HOME>/logs/<platform>/<design>/base/*.json`; ODBs in `results/.../base/*.odb`.
- `CLEAN=0` with `scripts/smoke_orfs.sh` reuses earlier stages (e.g. reuse synth, re-run floorplan).
