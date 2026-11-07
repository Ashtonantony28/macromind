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

| placer | ok/runs | seeds | GRT WL (um) | WNS (ns) | TNS (ns) | power (mW) | GRT overflow | proxy | make wall (min) |
|---|---|---|---|---|---|---|---|---|---|
| rtlmp | 1/1 | 0 | 1,672,341 | -0.0018 | -0.0019 | 183.9 | 0 | 1.450 | 20.8 |
| macromind-sa | 2/2 | 0,1 | 1,732,266 ± 196,279 | 0.0001 ± 0.0138 | -0.0008 ± 0.0105 | 183.2 ± 9.2 | 0 | 1.455 ± 0.701 | 12.5 ± 16.0 |
| macromind-random | 2/2 | 0,1 | 1,847,395 ± 1,180,190 | -0.0292 ± 0.142 | -0.113 ± 0.465 | 186.9 ± 19.7 | 0 | 1.520 ± 0.742 | 12.6 ± 2.6 |
| macromind-grid | 2/2 | 0,1 | 1,855,604 ± 557,072 | -0.0073 ± 0.0083 | -0.0105 ± 0.0482 | 186.2 ± 3.7 | 0 | 1.558 ± 0.302 | 12.9 ± 3.1 |
| macromind-rl (zero-shot, greedy) | 2/2 | 0,1 | 1,915,267 (both seeds identical) | -0.0011 | -0.0011 | 183.8 | 0 | 1.550 | 12.9 |

Mean ± 95% CI (Student t, n = 2). WNS/TNS here are post-GRT with estimated parasitics.

Per-run detail (GRT):

| placer | seed | GRT WL (um) | WNS (ns) | TNS (ns) | setup-viol. endpoints | power (mW) | proxy | 2_2 (s) | GRT (s) | make wall (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| rtlmp | 0 | 1,672,341 | -0.0018 | -0.002 | 2 | 183.9 | 1.450 | 256.4 | 512.0 | 20.8 |
| macromind-sa | 0 | 1,747,714 | +0.0012 | 0.000 | 0 | 182.5 | 1.510 | 23.3 | 251.8 | 13.8 |
| macromind-sa | 1 | 1,716,819 | -0.0010 | -0.002 | 2 | 183.9 | 1.399 | 17.7 | 281.5 | 11.2 |
| macromind-random | 0 | 1,940,278 | -0.0403 | -0.150 | 13 | 188.4 | 1.579 | 6.4 | 335.9 | 12.4 |
| macromind-random | 1 | 1,754,512 | -0.0180 | -0.077 | 8 | 185.3 | 1.462 | 6.2 | 318.2 | 12.8 |
| macromind-grid | 0 | 1,899,447 | -0.0067 | -0.007 | 1 | 186.5 | 1.534 | 7.0 | 313.8 | 12.7 |
| macromind-grid | 1 | 1,811,762 | -0.0080 | -0.014 | 5 | 185.9 | 1.582 | 7.9 | 325.0 | 13.2 |
| macromind-rl | 0 | 1,915,267 | -0.0011 | -0.001 | 1 | 183.8 | 1.550 | 10.3 | 288.2 | 12.2 |
| macromind-rl | 1 | 1,915,267 | -0.0011 | -0.001 | 1 | 183.8 | 1.550 | 63.5* | 345.0 | 13.6 |

\* first job after a container restart: cold torch import inside the placer subprocess.
rtlmp's make wall includes RTLMP (256 s) and ran under the heaviest contention.

Observations (bp_fe, GRT, 1 to 2 seeds; with n = 2 the 95% CIs are very wide
(t = 12.7), so none of the differences below is statistically established):

- **rtlmp has the lowest GRT wirelength** (1.672 M um); SA is next (1.748 M and
  1.717 M, mean 3.6% above rtlmp), then random/grid/RL (1.75 M to 1.94 M).
- **Timing is essentially closed for everyone except random**: WNS within
  ±7 ps for rtlmp, SA, grid and RL; random seed 0 is at -40 ps / TNS -0.15 ns.
  Power is 182 to 188 mW for all.
- **No placer produced GRT overflow** on this 800 x 600 um die (31% std-cell
  utilisation), so congestion does not separate the placers here.
- **RL zero-shot is poor, as expected** (checkpoint `runs/rl-v1/policy.pt`): the policy was trained on synthetic
  designs only; on bp_fe its greedy placement has proxy 1.550 (grid 1.534,
  random 1.462 to 1.579, SA 1.399 to 1.510, rtlmp 1.450) and the second-highest
  GRT WL (1.915 M um, +14.5% vs rtlmp). With `--budget` unset the policy runs one
  greedy episode, which is deterministic, so RL seeds 0 and 1 give the same
  placement (seed 1 is a repeat, not an independent sample).
- **Proxy vs GRT WL**: Spearman rho = +0.80 over the 9 bp_fe GRT runs (8 distinct placements, RL seeds coincide; all placers; `python -m macromind.report`). This is
  the first real-flow evidence that the proxy carries signal for wirelength;
  it has no timing term and says nothing about the tinyRocket timing reversal.
- **MacroMind placers are cheap; RTLMP is not**: stage 2_2 took 6 to 23 s for
  MacroMind (DEF dump + subprocess: DEF read 1.3 s, clustering 2.3 s, SA 11 to 15 s
  for 20k evaluations, RL 2.9 s) vs 256 s for hier-RTLMP.

## bp_fe_top full flow (detailed route)

`python -m macromind.harness sweep --design nangate45/bp_fe_top --placer rtlmp sa --seeds 0 --stop-after finish`

| placer | routed WL (um) | GRT WL (um) | WNS (ns) | TNS (ns) | power (mW) | DRC | antenna nets | GRT overflow | proxy | 2_2 / GRT / DRT (s) | make wall (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rtlmp | 1,444,654 | 1,672,341 | -0.085 | -3.46 | 186.0 | 0 | 0 | 0 | 1.450 | 102 / 329 / 523 | 24.2 |
| macromind-sa (seed 0) | 1,521,788 | 1,747,714 | -0.058 | -3.37 | 184.4 | 0 | 0 | 0 | 1.510 | 17 / 167 / 495 | 18.3 |

Both route DRC-clean. RTLMP's placement routes with 5.3% less wire; SA's has
better WNS (-58 vs -85 ps), about the same TNS (-3.37 vs -3.46 ns) and 0.9%
lower power, i.e. the opposite trade-off from tinyRocket (there SA won WL and
lost timing). The GRT numbers of these two runs are identical to the GRT-only
runs of the same placement (1,672,341 and 1,747,714 um), so the flow is
deterministic for a given placement and the GRT-stop sweep is directly
comparable. Note that post-route timing (-58 / -85 ps) is much worse than the
post-GRT estimate (about 0), so GRT-stage WNS should not be used as a timing
label for this design.

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
