# First ORFS runs through the MacroMind harness (SG7)

Date: 2026-11-07. Container: 4 cores, 15 GB, no GPU. ORFS `v3.0-3468-gc673f615`
(OpenROAD odb 0.109, Yosys 0.55). Every job ran with `NUM_CORES=2`, while the
tiny-gpu agent ran Yosys/ORFS on the same machine, so **wall times are
contended and only roughly comparable** (load average 5 to 13 during the runs;
the tiny-gpu Yosys was OOM-killed once while two of these jobs were routing,
and the container restarted once, killing three of the jobs below, which were rerun).

Mechanism and CLI: see [docs/harness.md](../harness.md). In short, MacroMind
placers run inside ORFS stage `2_2_floorplan_macro` through ORFS's own
`MACRO_PLACEMENT_TCL` hook and emit `place_macro` commands (macros `LOCKED`);
`rtlmp` is the stock flow. Each number below comes from a `ppa.json` written by
`python -m macromind.harness`; the tables are `python -m macromind.report`.

"proxy" is the MacroMind proxy (TILOS-style wirelength + density + congestion)
of the macro placement **as found in the post-2_2 ODB**, computed the same way
for every placer (std cells clustered into ~760 HEM clusters, quadratic cluster
pass, proxy). Lower is better.

## tinyRocket (nangate45, 2 fakeram45_64x32 macros, full flow to `finish`)

| placer | routed WL (um) | GRT WL (um) | WNS (ns) | TNS (ns) | setup viol. endpoints | power (mW) | DRC | GRT overflow | proxy | make wall (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| rtlmp | 637,956 | 816,305 | -0.098 | -26.6 | 440 | 119.5 | 0 | 0 | 1.285 | 35.1 |
| macromind-sa (seed 0) | 621,935 | 801,437 | -0.119 | -43.6 | 544 | 119.4 | 0 | 0 | 1.272 | 31.2 |

One seed each. Both route cleanly (0 DRC, 0 antenna nets). SA gives 2.5% less
routed wirelength and the same power but worse timing (WNS -0.119 vs -0.098 ns,
TNS -43.6 vs -26.6 ns). The proxy ranks SA slightly ahead, which agrees with
wirelength and disagrees with timing; the proxy has no timing term. The rtlmp run
reproduces the SG1 smoke run bit for bit (routed WL 637,956, WNS -0.097767), so
the base-sharing (copied synth + floorplan) does not perturb the flow.

Stage times (s): rtlmp 2_2 RTLMP 68.7, GRT 474, DRT 1000; SA 2_2 (incl. placer
subprocess) 14.4, GRT 386, DRT 1074. The SA job was first recorded as failed
by a harness bug (stale `klayout.lyt` copied from the base broke the final GDS
merge after routing had finished), fixed, and finished in place with
`python -m macromind.harness resume-dir`; its wall time includes both attempts.

## bp_fe_top (nangate45, 11 macros: 9x fakeram45_512x64, 64x7, 64x96), stop after global route

Full detailed routing for 10 placements did not fit the session, so the sweep
stops after `5_1_grt` (global-route WL, overflow, post-GRT timing with
estimated parasitics, power). Detailed-route runs for rtlmp and SA seed 0 are
listed in the next section.

BP_FE_GRT_TABLE

Per-run detail (GRT):

BP_FE_GRT_RUNS

Observations (bp_fe, GRT, 1 to 2 seeds; with n = 2 the 95% CIs are very wide
(t = 12.7), so none of the differences below is statistically established):

- **rtlmp has the lowest GRT wirelength** (1.672 M um); SA is next (1.748 M and
  1.717 M, mean 2.6% above rtlmp), then random/grid/RL (1.75 M to 1.94 M).
- **Timing is essentially closed for everyone except random**: WNS within
  ±7 ps for rtlmp, SA, grid and RL; random seed 0 is at -40 ps / TNS -0.15 ns.
  Power is 182 to 188 mW for all.
- **No placer produced GRT overflow** on this 800 x 600 um die (31% std-cell
  utilisation), so congestion does not separate the placers here.
- **RL zero-shot is poor, as expected**: the policy was trained on synthetic
  designs only; on bp_fe its greedy placement has proxy 1.550 (grid 1.534,
  random 1.462 to 1.579, SA 1.399 to 1.510, rtlmp 1.450) and the second-highest
  GRT WL (1.915 M um, +14.5% vs rtlmp). With `--budget` unset the policy runs one
  greedy episode, which is deterministic, so RL seeds 0 and 1 give the same
  placement (seed 1 is a repeat, not an independent sample).
- **Proxy vs GRT WL**: Spearman BP_FE_RHO over the BP_FE_N bp_fe runs (all placers). This is
  the first real-flow evidence that the proxy carries signal for wirelength;
  it has no timing term and says nothing about the tinyRocket timing reversal.
- **MacroMind placers are cheap; RTLMP is not**: stage 2_2 took 6 to 23 s for
  MacroMind (DEF dump + subprocess: DEF read 1.3 s, clustering 2.3 s, SA 11 to 15 s
  for 20k evaluations, RL 2.9 s) vs 256 s for hier-RTLMP.

## bp_fe_top full flow (detailed route)

BP_FE_FINISH

## What is incomplete

- Only 1 to 2 seeds per placer (the plan asks for >= 5): CIs are not meaningful yet.
- bp_fe detailed routing only for rtlmp and SA seed 0; random/grid/RL stop at GRT.
- rtlmp was run with seed 0 only: RTLMP is deterministic for a fixed config.
- RL used greedy decoding (`--budget` unset); best-of-k sampling and fine-tuning
  on bp_fe were not tried.
- Wall-clock times were measured under contention with another agent's jobs.
- MacroMind's spacing rule uses `MACRO_PLACE_HALO` as the macro-to-macro
  separation; macros may still sit on the core edge, and channel blockages are
  ORFS's (`block_channels`).
