# EDA toolchain (SG1)

Status as of 2026-11-06, in the cloud container (Ubuntu 24.04.5, 4 cores, 15 GB RAM, no GPU).

## TL;DR

```bash
scripts/setup_toolchain.sh          # idempotent; ~3 min on first run
source scripts/env.sh               # PATH, FLOW_HOME, OPENROAD_EXE, YOSYS_EXE ...
scripts/smoke_orfs.sh gcd           # full ORFS flow, prints metrics; ~75 s
scripts/smoke_orfs.sh tinyRocket    # 2 fakeram45 macros, full flow
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

Peak memory 1.1 GB (detailed route). The small negative WNS is ORFS's own gcd
constraint (the 26Q4 image shows -0.16 ns too); not a toolchain problem.

TINYROCKET_PLACEHOLDER

## ORFS designs with macros (nangate45 / asap7)

MACRO_TABLE_PLACEHOLDER
