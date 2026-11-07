# Predictors v1: GRT-labelled placements, congestion and timing predictors (SG8)

Date: 2026-11-07. Container: 4 cores, 15 GB, no GPU (torch CPU). ORFS `v3.0-3468-gc673f615`.
Raw evaluation records: [`predictors-v1-cv.json`](predictors-v1-cv.json) (includes every
held-out prediction), [`predictors-v1-lodo.json`](predictors-v1-lodo.json).

**Summary.** 75 GRT-labelled placements (47 bp_fe_top, 28 tinyRocket). On bp_fe_top, with
held-out placements, the learned models rank placements better than the MacroMind proxy for
GRT wirelength (Kendall tau 0.52 vs 0.32) and for congestion (0.58 vs 0.23). On tinyRocket
(2 macros, GRT WL spread of only ±3%) nothing ranks well, the proxy included. Neither the
models nor the proxy rank WNS/TNS: post-repair GRT timing is close to noise for these
placements. Leave-one-design-out with one training design does not transfer. Each predictor
ensemble (3 members) runs in 6 to 20 ms. Featurization plus both ensembles takes 21 to 41 ms,
plus the 54 to 122 ms cluster pass that the RL env already runs.

## 1. Data generation (`macromind/predict/datagen.py`)

```bash
python -m macromind.predict.datagen plan   --design bp_fe_top tinyRocket   # deterministic placement specs
python -m macromind.predict.datagen run    --design tinyRocket bp_fe_top --concurrency 2
python -m macromind.predict.datagen retry-congested
python -m macromind.predict.datagen legacy                                 # label the existing SG7 harness runs
python -m macromind.predict.datagen status
```

* **Placements.** Each design gets a fixed plan, with the kinds interleaved so that a partial
  run is already diverse:
  * the RTLMP placement replayed;
  * SA with budget 300 to 20k, acceptance temperature 0.2/0.5/0.9, density weight
    0.1/0.5/1.5, congestion weight 0/0.5/2, and grid or random init;
  * random legal placements;
  * shelf packing (grid);
  * perturbations of RTLMP and of SA@20k: Gaussian jitter of 2 to 20% of the core,
    0 to 2 swaps, random mirroring; legalised, orientation kept.

  The placements are generated offline on the floorplan DEF exported from the harness base,
  and written as `place_macro` TCL.
* **Flow.** Generation reuses the SG7 harness (shared synth + floorplan base,
  content-hash cache, its own run dir `runs/datagen/jobs/...`) and stops at `5_1_grt`. Three
  ORFS variables are set, and no ORFS file is patched:
  * `MACRO_PLACEMENT_TCL=<placement.tcl>`: ORFS's own stage-2_2 hook replays the placement;
  * `MACRO_PLACEMENT=` (empty);
  * `PRE_GLOBAL_ROUTE=macromind/predict/grt_labels.tcl`: wraps ORFS's `report_metrics` so
    that the final "global route" report first dumps `report_checks -format end`. This gives
    every setup endpoint's slack, after incremental repair, with GRT parasitics, in the same
    state the ORFS GRT metrics describe.
* **Congestion labels.** `gui::dump_heatmap` fails headless (GUI-0072). The global router
  stores per-layer gcell capacity and usage in the `dbGCellGrid` of `5_1_grt.odb`.
  `macromind/predict/odb_gcells.py` (run under `openroad -python`) reads them. The SWIG binding
  returns `uint8_t*`, which the script reads with ctypes. bp_fe has 380 x 285 gcells
  (2.1 um). The H and V layers are summed into H/V utilisation, and also into usage relative
  to the mean bin capacity. Both are pooled to the 32 x 32 model grid over the core. The
  scalar congestion label is `hot_frac`: the fraction of gcells at 90% or more utilisation,
  averaged over H and V. GRT overflow is 0 for every labelled sample, so overflow cannot be
  the label.
* **Resumable.** A sample is done when its run dir has `ppa.json` and `labels/done.json`.
  After label extraction, ODBs, netlists, guides and `mem.json` are pruned (about 1 to 4 MB
  per sample remain). The container restarted once mid-run. Re-running `run` restarted the 2
  interrupted jobs and skipped the finished ones.
* **Budget and runtime.** 73 placements were planned and 73 ORFS GRT jobs ran. Each took 6
  to 11 min with `NUM_CORES=2`. One job at a time ran while the tiny-gpu agent was active,
  then two. That is about 5.5 h wall in total.
  * tinyRocket: 27/27 ok.
  * bp_fe_top: 40/46 ok. Five placements failed in PDN (`PDN-0179`, unrepairable channel:
    perturbed macros left a sliver between the macro and the core edge). One SA placement
    (w_cong = 0) failed GRT with overflow (GRT-0116, WL 2.39 M, WNS -0.40). Re-running it
    with `-allow_congestion` also fails, because ORFS's incremental `global_route` calls do
    not take `GLOBAL_ROUTE_ARGS`. The congested and PDN failures are real outcomes of those
    placements, but they are not in the training set.
* **Legacy runs.** The 13 SG7 harness runs that reached GRT were labelled from their
  `5_1_grt.odb`: maps and scalars, no endpoint slacks. The GRT-stage metrics are used even
  for the runs that went on to `finish`. Samples are de-duplicated by macro placement, which
  leaves 7 for bp_fe and 1 for tinyRocket.

### Dataset

| design | samples | with endpoint slacks | kinds | GRT WL (M um) | hot_frac | WNS (ns) | TNS (ns) | proxy |
|---|---|---|---|---|---|---|---|---|
| bp_fe_top (11 macros, T = 1.8 ns) | 47 (40 datagen + 7 SG7) | 40 (6,000 endpoints each, 1,368 graph nodes) | 14 sa, 16 perturb, 10 random, 5 grid, 1 replay, 1 rl | 1.60 to 2.30 (mean 1.79) | 0.007 to 0.23 | -0.116 to +0.002 | -3.1 to 0 | 1.36 to 1.60 |
| tinyRocket (2 macros) | 28 (27 + 1) | 27 (4,146 endpoints, 576 nodes) | 9 sa, 8 perturb, 8 random, 2 grid, 1 replay | 0.80 to 0.86 (mean 0.82) | 0.0004 to 0.004 | -0.122 to -0.083 | -40.6 to -9.8 | 1.09 to 1.34 |

Every endpoint maps to a clustered-graph node, after removing the DEF `\[` escapes.

**Label noise and sensitivity: what the labels can and cannot say.**

* The flow is deterministic for a given 2_2 ODB: SG7's GRT and finish runs of the same
  placement match exactly, and so do their gcell maps.
* Replaying RTLMP's own `place_macro` file does *not* reproduce RTLMP's GRT result, although
  the macro positions and orientations are identical:
  * bp_fe: 1,659,720 vs 1,672,341 um (-0.8%);
  * tinyRocket: 835,312 vs 816,305 um (+2.3%), WNS -0.086 vs -0.097 ns.

  The 2_2 ODBs differ in something other than the macros, and global placement diverges from
  there (instance counts differ from 3_3 on). So the labels carry roughly 1 to 2% of
  "flow-path" noise in WL. On tinyRocket that is comparable to the whole label spread (±3%).
* bp_fe WNS sits within a few ps of 0 for most placements after `repair_timing`. Only the
  worst placements show -40 to -116 ps.
* Over the 47 bp_fe placements, the proxy's rank correlation with GRT WL is Spearman 0.45
  (tau 0.32). That is lower than the 0.80 SG7 measured on 9 runs, which spanned less
  placement diversity. On tinyRocket the proxy is *anti*-correlated with GRT WL (tau -0.30).

## 2. Models (`macromind/predict/{features,models,train,runtime}.py`)

* **Inputs**, computed from the clustered netlist and the macro placement only, after the
  same quadratic cluster pass the proxy uses. The image is 32 x 32 over the core with these
  channels:
  * macro occupancy;
  * log std-cell cluster density;
  * log RUDY H and V (over capacity);
  * log pin density;
  * log macro-pin density;
  * CoordConv x and y.

  The graph is the clique-expanded clustered netlist (bp_fe: 3,282 nodes, 10.6 k edges,
  row-normalised weights). Node features are kind, size, degrees, position for every node,
  and distance to the nearest macro. The image channels are also gathered at each node.
* **Congestion model.** A 3-layer GraphSAGE (32 dimensions) produces 8-channel node
  embeddings, scattered (mean) onto the grid. These are concatenated with the image and fed
  to a 3-level U-Net (width 16). It outputs 4 maps: H/V utilisation and H/V usage relative to
  the mean bin capacity. A scalar head on the bottleneck, which also takes the proxy's
  wirelength, density and congestion terms, outputs `[log(GRT WL / cluster HPWL), hot_frac]`.
  The model has 129 k parameters.
* **Timing model.** A 4-layer GraphSAGE (48 dimensions). The per-node head predicts the worst
  endpoint slack / T, masked to nodes that contain endpoints. The graph head (mean + max
  pool + proxy terms) predicts `[WNS/T, -log1p(-TNS/T)]`. The model has 25 k parameters.
* **Training.** AdamW 2e-3 with cosine schedule, 100 epochs, batch 1, random x/y mirroring of
  the whole layout. Scalar targets are standardised per design: rankings are within a design,
  and global standardisation made the heads learn design identity. On a design without
  training data the global statistics are used. Ensembles have 3 seeds and report mean and
  std. On 2 threads, one ensemble on about 60 samples takes about 6 to 7 min.
* **Baselines.**
  * The MacroMind proxy itself, as a ranker.
  * Cluster HPWL.
  * Ridge regression on the proxy terms plus image statistics, with the same targets: a
    "cheap learned" baseline.
  * For maps: the input RUDY, and the mean training map of the same design.
  * For node slack: the per-node mean slack over the training placements. This knows which
    nodes are critical but nothing about the placement.

## 3. Results

### Held-out placements (5-fold within each design; the other design is always in training; 3-seed ensembles)

Kendall tau between predicted and true cost over the out-of-fold predictions (higher is
better; every target is oriented as a cost). "rho(unc, abs err)" is the Spearman correlation
between ensemble std and absolute error.

| design | n | target | tau predictor | tau proxy | tau ridge | tau cluster HPWL | rho predictor | rho proxy | rho(unc, abs err) |
|---|---|---|---|---|---|---|---|---|---|
| bp_fe_top | 47 | GRT WL, scalar head | +0.45 | +0.32 | +0.44 | -0.23 | +0.62 | +0.45 | +0.25 |
| bp_fe_top | 47 | **GRT WL, sum of predicted usage map** | **+0.52** | +0.32 | - | -0.23 | +0.72 | +0.45 | +0.30 |
| bp_fe_top | 47 | **congestion (hot_frac), scalar head** | **+0.58** | +0.23 | +0.43 | -0.42 | +0.79 | +0.34 | +0.31 |
| bp_fe_top | 47 | congestion, top-10% of predicted util map | +0.54 | +0.23 | - | -0.42 | +0.75 | +0.34 | -0.36 |
| bp_fe_top | 47 | WNS (GRT) | +0.13 | +0.02 | +0.19 | -0.09 | +0.19 | +0.03 | +0.03 |
| bp_fe_top | 47 | TNS (GRT) | +0.06 | -0.01 | +0.16 | -0.07 | +0.10 | -0.01 | +0.27 |
| tinyRocket | 28 | GRT WL, scalar head | -0.24 | -0.30 | -0.09 | -0.29 | -0.33 | -0.42 | +0.35 |
| tinyRocket | 28 | GRT WL, sum of predicted usage map | -0.06 | -0.30 | - | -0.29 | -0.09 | -0.42 | -0.12 |
| tinyRocket | 28 | congestion (hot_frac), scalar head | +0.20 | -0.15 | -0.06 | -0.28 | +0.27 | -0.24 | +0.01 |
| tinyRocket | 28 | congestion, top-10% of predicted util map | -0.24 | -0.15 | - | -0.28 | -0.34 | -0.24 | -0.04 |
| tinyRocket | 28 | WNS (GRT) | -0.33 | +0.12 | -0.25 | +0.18 | -0.48 | +0.17 | +0.18 |
| tinyRocket | 28 | TNS (GRT) | -0.37 | -0.07 | -0.28 | +0.01 | -0.52 | -0.09 | -0.16 |

The GRT utilisation map (32 x 32, H+V):

| design | MSE | MSE (design-mean map) | pixel corr | pixel corr (design-mean map) | pixel corr (input RUDY) | corr of (pred - mean) vs (true - mean) |
|---|---|---|---|---|---|---|
| bp_fe_top | 0.004 | 0.023 | 0.93 | 0.48 | -0.00 | 0.91 |
| tinyRocket | 0.004 | 0.005 | 0.74 | 0.63 | -0.06 | 0.50 |

Per-node endpoint slack (Spearman over nodes, averaged over held-out samples):

| design | predictor | per-node training mean (placement-blind) |
|---|---|---|
| bp_fe_top | 0.82 | 0.89 |
| tinyRocket | 0.70 | 0.97 |

### Leave one design out (train on one design, test on the other)

| held-out design | GRT WL (usage map) tau | congestion (scalar) tau | congestion (map top-10%) tau | WNS tau | map pixel corr | node-slack rho |
|---|---|---|---|---|---|---|
| bp_fe_top (trained on tinyRocket only) | -0.29 (proxy +0.32) | +0.30 (proxy +0.23) | +0.11 | -0.22 | 0.35 | 0.49 |
| tinyRocket (trained on bp_fe only) | +0.23 (proxy -0.30) | +0.03 (proxy -0.15) | +0.26 | -0.02 | 0.40 | 0.41 |

With one training design there is nothing to generalise from. These numbers are at chance
level and are listed for completeness.

### Runtime (bp_fe_top / tinyRocket, 2 torch threads, idle machine, median of 15)

| step | bp_fe_top | tinyRocket |
|---|---|---|
| cluster pass (already part of the RL env's proxy evaluation) | 122 ms | 54 ms |
| featurize (image + node features) | 6.8 ms | 3.0 ms |
| congestion ensemble (3 members) | 17.9 ms | 10.4 ms |
| timing ensemble (3 members) | 19.3 ms | 6.4 ms |
| `Predictor.predict` total (featurize + both ensembles) | 40.6 ms | 20.5 ms |

Under contention from 2 ORFS jobs, the total was 130 to 160 ms. Training: 5-fold CV with 3
seeds and 100 epochs took 65 min. LODO took 17 min. The final bundle fit on all 75 samples
(`runs/predict/bundle.pt`) took about 20 min. All on 2 threads.

## 4. RL reward hook

* `EnvConfig(predictor=Predictor.load("runs/predict/bundle.pt", design_key=...), pred_weight=lambda, unc_weight=kappa)`
  makes the terminal cost `proxy + lambda * mu + kappa * sigma`.
  * `mu` is the standardised predicted badness: `w_c * z(hot_frac) + w_t * z(-WNS/T)`, plus
    optionally `w_wl * z(log WL/HPWL)`.
  * `sigma` is the ensembles' member spread combined with the same weights.
  * The reward reference (grid packing) uses the same cost.
  * With zero weights the reward is bit-identical to the proxy reward (tested).
* The hook is `terminal_cost()` in `macromind/place/env.py`. This edit changes the harness
  `code_version()` hash, so MacroMind-placer harness runs get new cache keys. Old runs stay
  on disk under their old keys.
* No RL training was run with the hook. The timing term should be given weight 0 for now,
  given the results above.

## 5. Caveats and what is incomplete

* **Small data, two designs.** 75 labels instead of the plan's thousands across 6 to 8
  designs, with one 11-macro design carrying the positive result. The bp_fe numbers are 47
  out-of-fold predictions with a 3-seed ensemble per fold. A tau of 0.52 over 47 items has a
  standard error of roughly ±0.1. The win over the proxy for congestion and WL is likely
  real, but its size is not established.
* **Ridge is a strong baseline.** Ridge on the proxy terms gets tau 0.44 for both WL (scalar
  head) and congestion. The networks' extra gain is +0.08 (WL, map-sum) and +0.15
  (congestion) over ridge. Most of the gain over the raw proxy comes from *re-weighting*
  proxy-like features on real labels.
* **Timing is not predicted.** No ranker, learned or not, ranks GRT WNS/TNS above chance.
  On bp_fe this is because post-repair WNS is within a few ps of 0. On tinyRocket the
  placement-driven spread (WNS -0.08 to -0.12 ns) is comparable to the flow-path noise (the
  replay effect above). Per-node slack prediction is beaten by the placement-blind per-node
  mean. Possible next steps:
  * pre-repair labels (`SKIP_INCREMENTAL_REPAIR=1` or a dump before `repair_timing`);
  * the expensive detailed-route label. SG7 showed post-route WNS differs strongly from the
    GRT estimate on bp_fe.
* **No cross-design generalisation is shown.** LODO with one training design is at chance.
  It needs more designs: bp_be, swerv, and tiny-gpu c8/c16 when their flows are cheap enough.
  tiny-gpu was not added because its config was still being brought up during data
  generation.
* **Missing labels.** Congested and PDN-failing placements are absent from the training set
  (6 of 46 bp_fe placements). A predictor trained only on routable placements has never seen
  a failure, which matters for an RL agent that will produce them. Possible fixes: a
  feasibility classifier, or ORFS's incremental GRT accepting `-allow_congestion`.
* **Uncertainty is weakly informative.** Ensemble std vs absolute error has Spearman 0.25 to
  0.31 on bp_fe WL and congestion, and is near 0 or negative elsewhere.
* No CircuitNet pre-training. Detailed-route labels were not generated.

## Reproduce

```bash
. .venv/bin/activate
python -m macromind.predict.datagen run --design tinyRocket bp_fe_top --concurrency 2   # ~5.5 h
python -m macromind.predict.datagen legacy
python -m macromind.predict.train cv   --folds 5 --seeds 3 --epochs 100 --out runs/predict/cv.json
python -m macromind.predict.train lodo --seeds 3 --epochs 100 --out runs/predict/lodo.json
python -m macromind.predict.train fit  --seeds 3 --epochs 100 --out runs/predict/bundle.pt
python -m macromind.predict.report runs/predict/cv.json runs/predict/lodo.json
pytest -q tests/test_predict.py
```
