# SG7: ORFS macro-placement hook and evaluation harness

## Mechanism: ORFS `MACRO_PLACEMENT_TCL` (no ORFS patch)

In the pinned ORFS (`v3.0-3468-gc673f615`), stage `2_2_floorplan_macro` runs
`scripts/macro_place_util.tcl`, which does:

```tcl
if { [env_var_exists_and_non_empty MACRO_PLACEMENT_TCL] } {
  log_cmd source $::env(MACRO_PLACEMENT_TCL)           ;# <- our hook
} elseif { [env_var_exists_and_non_empty MACRO_PLACEMENT] } {
  read_macro_placement $::env(MACRO_PLACEMENT)
} else {
  rtl_macro_placer ...                                  ;# hier-RTLMP (default)
}
source placement_blockages.tcl ; block_channels ...
```

and then writes `2_2_floorplan_macro.odb` plus `2_2_floorplan_macro.tcl`
(`write_macro_placement`). The Makefile also lists `$(MACRO_PLACEMENT_TCL)` as a
dependency of the 2_2 stage. So setting `MACRO_PLACEMENT_TCL` is the supported
way to replace RTLMP while keeping everything else in the stage (channel
blockages, metrics, the ODB hand-off) and all later stages unchanged.

We chose it over the alternatives because:

- **`MACRO_PLACEMENT` (a `read_macro_placement` text file)** needs the
  placement to exist *before* the stage runs, so it needs a two-pass flow
  (run to 2_1, export, place, rerun). The TCL hook places inside the stage from
  the exact ODB in memory, so one `make` call does the whole flow.
- **Patching `macro_place.tcl`** would fork ORFS.
- **DEF round trip into the ODB** (`read_def -incremental`) works but is
  touchier than `place_macro`.

The hook (`macromind/flow/orfs_hook.tcl`):

1. `write_def` of the floorplan (macros unplaced) into the job's `macromind/` dir;
2. `exec`s `python -m macromind.flow.place --def ... --lef <TECH_LEF SC_LEF ADDITIONAL_LEFS>
   --method <m> --seed <s> --halo max(MACRO_PLACE_HALO) --out macro_placement.tcl --json placer.json`
   with the venv's python (torch is there; OpenROAD's embedded Python 3.10 is
   stdlib-only), `PYTHONHOME` unset;
3. `source`s the result: one `place_macro -macro_name <inst> -location {x y}
   -orientation <R0|MX|..>` per macro. That is the same format ORFS's
   `write_macro_placement` emits, so a MacroMind placement and an RTLMP one are
   interchangeable files. `place_macro` snaps to the manufacturing grid / pin-track
   alignment (sub-0.2 um moves) and sets the instance status to `LOCKED` (i.e.
   firm: global placement, resizing and detailed placement do not move it).

`python -m macromind.flow.place` reads DEF+LEF into a `Design`, clusters the
std cells (HEM, seed 0, ~n_cells/40 clusters), runs the named placer on the
clustered design through the shared legaliser (macro spacing = the ORFS
`MACRO_PLACE_HALO` value), scores it with the proxy and writes TCL (or a DEF
with FIXED macros when `--out` ends in `.def`).

**Config switch.** The harness generates a wrapper `config.mk` per job:

```make
include $(FLOW_HOME)/designs/nangate45/bp_fe_top/config.mk
export MACRO_PLACER = macromind-sa          # or rtlmp
export MACRO_PLACEMENT_TCL = .../macromind/flow/orfs_hook.tcl
export MACROMIND_METHOD = sa
export MACROMIND_SEED = 0
export MACROMIND_PYTHON = .../.venv/bin/python
export MACROMIND_ROOT = ...
export MACROMIND_OUT_DIR = <job dir>/macromind
```

For `MACRO_PLACER = rtlmp` only the `include` and that line are written, i.e.
the stock ORFS flow.

## Harness

```bash
. .venv/bin/activate
python -m macromind.harness run   --design nangate45/bp_fe_top --placer macromind-sa --seed 0 [--stop-after grt|route|finish]
python -m macromind.harness sweep --design nangate45/bp_fe_top --placer rtlmp sa grid random rl --seeds 0 1 \
                                  --stop-after grt --max-concurrency 2
python -m macromind.harness key   --design nangate45/gcd --placer rtlmp          # cache key + run dir
python -m macromind.harness base  --design nangate45/bp_be_top                   # pre-build synth + 2_1
python -m macromind.harness import-base --design nangate45/tinyRocket --work-home runs/orfs/smoke
python -m macromind.report [--out docs/results/table.md]
```

Options: `--budget` (SA evaluations / RL samples), `--num-cores` (ORFS
`NUM_CORES`, default 2), `--timeout` (s, per job), `--set VAR=VAL` (extra ORFS
variables, part of the cache key), `--rl-ckpt`, `--force`.

- **Base sharing.** Synthesis + `2_1_floorplan` do not depend on the placer, so
  each design config builds them once into `runs/harness/base/<p>_<d>-<key>/`
  (under a file lock). A job copies those stage files (mtimes preserved, so
  `make` does not rebuild them) into its own WORK_HOME and runs from 2_2 on.
- **Isolation.** Every job has its own WORK_HOME
  `runs/harness/jobs/<p>_<d>/<placer>-s<seed>-<stop>-<key>/`.
- **Cache key** = sha256 of: design name + hash of the design dir's config/sdc/
  tcl/lef/lib files, ORFS version tag, placer, seed, budget, stop stage, extra
  variables, and for MacroMind placers the hash of the placement code (all
  `macromind/**/*.py` except `harness/`, `report/`, `predict/`, plus the hook) and
  of the RL checkpoint. RTLMP jobs do not depend on MacroMind code. A run dir with
  a `ppa.json` is reused; failures are cached too (rerun with `--force`).
- **`ppa.json`** has `status` (ok/failed/timeout/error), `ppa` (routed WL, GRT
  WL, WNS, TNS, hold WNS, power, DRC count, GRT overflow, antenna nets, util,
  ...), `sources` (which ORFS metric key each value came from: finish >
  detailed route > global route > CTS), `grt` (overflow table), `stage_runtime`
  (per-stage seconds + peak MB from the stage logs), `make_wall_s`,
  `placement` (the macros as they are in the post-2_2 ODB, exported with
  `write_def`, plus the MacroMind proxy computed the same way for *every* placer:
  cluster std cells, quadratic cluster pass, TILOS proxy), `placer` (the
  placer's own JSON for MacroMind methods), and all merged ORFS stage metrics
  (`orfs_metrics`).
- **Sweep** fans out design x placer x seed on a thread pool
  (`--max-concurrency`); bases are built first, serially.

Gotchas found while building it:

- ORFS helper scripts call `python3` and need PyYAML, so the venv must not be on
  `PATH` for `make`; `tool_env()` strips it.
- ORFS's `make print-VAR` goes through a single fixed temp file
  (`/tmp/print_tmp$`), so concurrent prints race; they are serialised with a lock.
- `DESIGN_NICKNAME` (e.g. `bp_fe`) names the results dir, not `DESIGN_NAME`;
  the harness asks make for `RESULTS_DIR`/`LOG_DIR` instead of guessing.

## Tests

`pytest -q` runs the unit tests (config generation, emit/parse round trips via
TCL and DEF, metric parsing on real tinyRocket ORFS reports in
`tests/fixtures/orfs_tinyRocket/`, GRT overflow, cache keys and cache hits,
report CI maths). `pytest -m eda tests/test_harness.py` (or `MACROMIND_EDA=1`)
runs real flows: nangate45/gcd with RTLMP to global route, and tinyRocket with
`macromind-grid` through floorplan (checks that the ODB holds the placement the
hook asked for).
