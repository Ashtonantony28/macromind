# MacroMind results report v1 (SG10): bp_fe_top, global route

2026-11-11. 4 cores, no GPU. ORFS v3.0-3468-gc673f615, NUM_CORES=2, at most 2 jobs at once, nothing else running. Tables come from `python -m macromind.report.sg10 --out docs/results/report-v1-data.md`.

## Short version

Fine-tuning RL v1 on bp_fe_top for about 30 CPU-min cuts GRT WL by about 7%, to SA level. It is still 3.5–4.3% worse than RTLMP (OpenROAD's default macro placer).

Both fine-tuned policies collapsed to a single placement, so the reward ablation compares one placement with one. The predictor reward gave 0.7% less WL (within flow noise) and better WNS, but 2.4x more hot gcells. No demonstrated benefit.

On unseen placements the predictor ranks real outcomes better than the proxy. The policy trained against it still found a placement whose congestion the predictor got wrong (reward hacking).

## Setup

- **Design:** bp_fe_top, 11 macros, 30,404 cells in 760 HEM clusters, halo 10 um. Stop after 5_1_grt.
- **Metrics:** GRT WL, hot gcells (share at ≥90% utilisation), overflow, post-GRT WNS/TNS (estimated parasitics), power.
- **Placers:** rtlmp (deterministic, 1 run); sa (20k evaluations); random; rl-v1 (`runs/rl-v1/policy.pt` zero-shot, best of 64 samples by training reward); rl-v2-proxy (fine-tuned from v1 with proxy reward); rl-v2-pred (proxy + 0.05 x predictor penalty (congestion 1, WL 1, timing 0) + 0.1 x ensemble std; bundle `runs/predict/bundle.pt`).
- **Seeds:** 0–2 for each stochastic placer.
- **Fine-tune:** 35,200 env steps per arm (200 PPO iterations x 16 episodes), 1 thread, lr 1e-4, entropy 0.01.
- **Inference:** 64 samples, best by the arm's own training reward. ORFS results are never used for selection.

## Main table

Mean ± 95% CI; "distinct" = different placements. Identical placements reproduce bit for bit.

| placer | runs (distinct) | GRT WL | vs rtlmp | hot % | WNS ns | TNS ns | power mW | proxy | placer s |
|---|---|---|---|---|---|---|---|---|---|
| rtlmp | 1 (1) | 1,672,341 | 0 | 2.67 | -0.0018 | -0.0019 | 183.9 | 1.450 | 256 (whole stage) |
| sa | 3 (3) | 1,735,823 ± 41,313 | +3.8 ± 2.5% | 2.84 ± 2.12 | -0.0013 | -0.0031 | 182.9 | 1.456 | 22 |
| random | 3 (3) | 1,845,748 ± 230,843 | +10.4 ± 13.8% | 8.44 ± 1.82 | -0.036 | -0.27 | 187.2 | 1.510 | 3.5 |
| rl-v1 best-of-64 | 3 (3) | 1,867,393 ± 68,418 | +11.7 ± 4.1% | 5.62 ± 1.23 | -0.023 | -0.12 | 185.4 | 1.361 | 15.5 |
| rl-v2-proxy | 3 (1) | 1,744,236 | +4.3% | 0.88 | -0.0266 | -0.108 | 182.6 | 1.297 | 14.9 |
| rl-v2-pred | 3 (1) | 1,731,466 | +3.5% | 2.16 | -0.0028 | -0.0028 | 182.7 | 1.310 | 17.7 |

GRT overflow was 0 for all runs. Earlier SG7 rows (grid, rl-v1 greedy) are in `report-v1-data.md`. Detailed route exists only for SG7 rtlmp and sa seed 0: SA had +5.3% routed WL and better WNS.

Readings:
- RL v2 has the best proxy score of any placer, but not the best real WL: the proxy optimum is not the GRT optimum.
- Against rl-v1, fine-tuning gives -7% WL (n_eff = 1 on the v2 side).
- rl-v2-pred is level with SA.
- rl-v2-proxy has the fewest hot gcells but the worst WNS among non-random placers.

## Reward ablation (pred − proxy)

| metric | proxy | pred | Δ |
|---|---|---|---|
| GRT WL | 1,744,236 | 1,731,466 | -0.73% (in noise) |
| hot % | 0.88 | 2.16 | +1.28 pp |
| WNS ns | -0.0266 | -0.0028 | +0.024 |
| TNS ns | -0.108 | -0.0028 | +0.105 |
| proxy | 1.297 | 1.310 | +0.013 |
| predicted GRT WL | 1,794,867 | 1,645,856 | -8.3% |
| predicted hot % | 3.29 | 2.47 | -0.82 pp |
| fine-tune CPU min | 30.3 | 36.9 | +22% |
| distinct samples of 64 | 5–7 | 1 | — |

- **Entropy collapsed:** 4.2 at the start, 0.20 (proxy) and 0.02 (pred) at the end, so n_eff = 1 per arm and no significance test is possible.
- **Wrong direction on congestion:** the predictor called the pred placement less congested; it is 2.4x more.
- **WL gain overstated:** the predicted gain was more than ten times the real one.
- **Verdict:** no demonstrated benefit at this budget and these weights.

## Proxy and predictor vs real GRT

Kendall tau / Spearman rho; each distinct placement counts once. "Fresh" = placements generated after the predictor bundle was trained (same design).

| ranker | all n=15: WL | hot | WNS | fresh n=11: WL | hot | WNS |
|---|---|---|---|---|---|---|
| proxy | .35/.44 | .31/.45 | .05/.00 | .24/.28 | .45/.61 | .09/.08 |
| proxy WL term only | -.10/-.17 | -.30/-.40 | -.33/-.46 | -.13/-.23 | -.27/-.35 | -.42/-.52 |
| predictor GRT WL | .45/.59 | .26/.32 | .14/.21 | .27/.36 | .27/.31 | .20/.26 |
| predictor hot_frac (RL penalty) | .52/.71 | .60/.80 | .49/.66 | .53/.71 | .60/.80 | .60/.75 |

A good ranking over varied placements did not survive optimisation pressure from the policy.

## Compute

- Fine-tunes: proxy 31.4 min wall / 30.3 CPU-min; pred 35.7 min wall / 36.9 CPU-min. Equal step budgets, unequal CPU.
- ORFS: 14 GRT jobs, 1.9 h of job wall time, 7.3–8.6 min per job.
- Placer time per run: RL 15–18 s (including 2.3 s DEF read + clustering), SA 22 s, RTLMP stage 256 s.
- Reused: SG7 rtlmp, sa s0, grid, greedy; SG8 predictor; SG6 policy.

## Skipped (to save compute/usage)

- tiny-gpu c4 (the RTLMP job and fine-tunes did not finish before a container restart).
- Grid rerun; detailed route for the new placers; rtlmp seeds 1–2 (deterministic).

## Threats to validity

1. The predictor was trained on bp_fe_top, so this evaluation is in-distribution. Out-of-distribution behaviour is untested and likely worse.
2. n_eff = 1 per RL v2 arm, with one fine-tune per reward.
3. Flow noise of 1–2% from small placement perturbations; the ablation WL gap and the SA / v2 / RTLMP gaps are at that level.
4. One design, GRT only, estimated parasitics, loose clock.
5. Reward weights were not tuned (no leakage, but possibly poor weights).
6. The proxy is computed after place_macro snapping (1.297 vs the placer's own 1.287).
7. sa s2 and rl-v1 s0 were resumed after a container restart, so their flow wall times are understated.

## Next steps

1. Prevent entropy collapse (entropy floor, inference temperature, diversity bonus), then 3 fine-tune seeds per reward.
2. Out-of-distribution test on tinygpu_c4 and bp_be_top with a leave-one-design-out predictor bundle.
3. Anti-hacking: stronger ensemble-std penalty, a cap on the predictor term, and online relabelling of the policy's own placements with ORFS (the plan's active-learning loop).
4. Detailed route for rtlmp, sa and both v2 arms.
5. A timing term that actually predicts something.

## Reproduce

```bash
python -m macromind.place.finetune --def runs/rl-v2/bp_fe_top/floorplan.def --lef $(cat runs/rl-v2/bp_fe_top/lefs.txt) --halo 10 --reward proxy|pred [--design-key bp_fe_top --pred-weight 0.05 --unc-weight 0.1] --steps 35200 --threads 1 --out runs/rl-v2/bp_fe_top/<arm>
runs/sg10/sweep2.sh
python -m macromind.harness label-prune
python -m macromind.report.sg10 --out docs/results/report-v1-data.md
```
