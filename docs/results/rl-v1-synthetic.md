# RL placer v1 on synthetic designs (2026-11-06)

First end-to-end experiment of the SG6 RL macro placer (PPO, GNN + CNN policy)
against the SG4 baselines. Everything is on CPU (2 torch threads). The cost
is the TILOS-style **proxy** (wirelength + 0.5 density + 0.5 congestion), not
routed PPA, and the designs are synthetic, so these numbers say nothing yet
about real chips.

Raw records: `rl-v1-heldout.json`, `rl-v1-finetune.json`, `rl-v1-train.json`
(same directory). Logs and checkpoints are under `runs/` (git-ignored).

## Protocol

* **Designs.** `generate_synthetic(n, 10 n, seed)` with `n` drawn from 8..20.
  Pre-training: seeds 0..5 (19, 14, 18, 18, 17 and 16 macros). Held out:
  seeds 1001..1004 (19, 15, 11 and 17 macros). Fine-tuning: seed 1001.
* **Scoring (all methods).** Legalise the macros, run the quadratic cluster
  placement pass (`macromind.graph.place_clusters`), then compute the proxy.
  The column "no cluster pass" leaves clusters at the generator's
  bisection positions instead. The two protocols rank methods very
  differently (see the caveats).
* **Methods.** `random@32`: best of 32 random legal placements, selected
  under the scoring protocol. `grid`: shelf packing. `wiremask`: greedy, no
  learning; each macro goes to the valid cell with the smallest HPWL
  increase. `sa@N`: simulated annealing with N incremental proxy
  evaluations, clusters frozen. `sa-alt@20000`: SA aware of the cluster
  pass; grid start, then twice (cluster pass, then SA with 10k
  evaluations). `rl:X`: greedy policy, 1 evaluation. `rl:X@32`: greedy plus 32
  sampled episodes, best kept (33 full evaluations).
* **Training.** Pre-training: 25 min wall clock, 112 PPO iterations, 182k env
  steps (about 10.7k episodes). The mean sampled reward relative to grid went
  from +0.03 to +0.10. Fine-tuning from the pre-trained checkpoint and
  training from scratch on seed 1001 each got 6 min. Grid G = 32,
  16 episodes per design per iteration (32 when fine-tuning), lr 3e-4,
  entropy 0.01, 4 epochs, clip 0.2.
* Total compute is about 45 min: 25 pre-training, 12 for fine-tuning and
  from-scratch, about 8 evaluating.

## Results: held-out designs (zero-shot)

Mean over 4 held-out designs; lower is better. The ratio to grid is averaged
per design.

| method | mean proxy | proxy / grid | proxy / grid (no cluster pass) | evals | time per design |
|---|---|---|---|---|---|
| random@32 | 1.0191 | 0.993 | 0.870 | 32 | 1.2 s |
| grid | 1.0286 | 1.000 | 1.000 | 1 | <0.1 s |
| wiremask (heuristic) | 1.0099 | 0.982 | 0.706 | 1 | 0.1 s |
| sa@33 | 1.0007 | 0.975 | 0.920 | 33 | <0.1 s |
| sa@20000 | 0.9756 | 0.949 | **0.639** | 20000 | 6 s |
| sa-alt@20000 | 0.9571 | 0.931 | 0.919 | 20000 | 6 s |
| rl:pre (greedy) | 1.0262 | 0.999 | 0.807 | 1 | 0.1 s |
| rl:pre@32 | **0.9133** | **0.889** | 0.839 | 33 | 1.8 s |

Per design, rl:pre@32 scores 0.882, 0.990, 0.933 and 0.848. Under the
scoring protocol it is the best method on all four designs. The greedy
policy alone is about as good as grid packing, so on unseen designs the
gain comes from sampling.

## Results: fine-tuning on held-out design seed 1001 (19 macros)

| method | proxy | wirelength | density | congestion | proxy (no cluster pass) | evals |
|---|---|---|---|---|---|---|
| grid | 0.9989 | 0.1302 | 0.3833 | 1.3540 | 1.4854 | 1 |
| wiremask | 0.9459 | 0.0876 | 0.4098 | 1.3067 | 0.9805 | 1 |
| sa@20000 | 0.9263 | 0.1032 | 0.4109 | 1.2352 | **0.8878** | 20000 |
| sa-alt@20000 | 0.9222 | 0.1040 | 0.3928 | 1.2436 | 1.3864 | 20000 |
| rl:pre (zero-shot greedy) | 0.9969 | 0.1072 | 0.4118 | 1.3677 | 1.0570 | 1 |
| rl:pre@32 | 0.8818 | 0.1063 | 0.3955 | 1.1555 | 1.1208 | 33 |
| rl:scratch (6 min) | 0.8660 | 0.0902 | 0.3957 | 1.1559 | 1.0044 | 1 |
| rl:scratch@32 | 0.8450 | 0.0922 | 0.4091 | 1.0965 | 1.0710 | 33 |
| rl:ft (pre + 6 min) | 0.8376 | 0.1010 | 0.3471 | 1.1260 | 1.2019 | 1 |
| rl:ft@32 | **0.7887** | 0.0961 | 0.3582 | 1.0271 | 1.2082 | 33 |

With the same 6 minutes, fine-tuning the pre-trained policy beats training
from scratch (0.838 vs 0.866 greedy, 0.789 vs 0.845 best-of-32). This is
some evidence that pre-training transfers. It is one design and one seed,
though.

## Training designs (seeds 0 and 1)

| method | mean proxy / grid | no cluster pass |
|---|---|---|
| random@32 | 0.989 | 0.937 |
| sa@20000 | 1.008 | 0.679 |
| rl:pre | 1.017 | 0.869 |
| rl:pre@32 | 0.886 | 0.875 |

## Caveats (read before quoting any number)

1. **The ranking depends on the scoring protocol.** Without the cluster
   pass, SA@20000 wins clearly (0.639 of grid against 0.839 for RL). SA
   optimises the proxy with clusters frozen, and the cluster pass then
   moves them and undoes much of that work. RL is trained on the post-pass
   reward. `sa-alt` is a first attempt at an SA that knows about the
   cluster pass. It helps (0.931), but it is not SA with the pass in the
   loop, which would cost about 20 ms per evaluation. Until that baseline
   exists, "RL beats SA" holds only under this protocol.
2. **Best-of-k selection uses the final metric.** RL@32 and random@32 pick
   their best sample using the same proxy they are scored on. After the
   cluster pass the proxy is quite noisy with respect to small macro moves,
   so part of the best-of-k gain is selection on noise. SA's final choice
   happens before the pass.
3. **Congestion dominates the proxy** (about 1.0 to 1.6 against density
   about 0.4 and wirelength about 0.1). The weights are untuned, so "better
   proxy" mostly means "lower top-5% RUDY".
4. **Wire-mask prior.** The policy logits include a learnable
   `-wm_scale * wiremask` term (initialised at 2), in the style of
   MaskPlace. The pure wire-mask heuristic alone reaches 0.982 of grid
   under the protocol, so the learned part is what takes RL from there to
   0.889.
5. These are small synthetic designs (11 to 19 macros, about 150 clusters),
   with one training seed and no confidence intervals. Nothing here is
   routed.

## Reproduce

```bash
python -m macromind.place.train --synthetic 8-20 --n-designs 6 --steps 10000000 --time-limit 1500 --episodes 16 --out runs/rl-v1
python -m macromind.place.train --synthetic 8-20 --design-seeds 1001 --init runs/rl-v1/policy.pt --steps 10000000 --time-limit 360 --episodes 32 --out runs/rl-v1-ft
python -m macromind.place.train --synthetic 8-20 --design-seeds 1001 --steps 10000000 --time-limit 360 --episodes 32 --out runs/rl-v1-scratch
python -m macromind.place.evaluate --ckpt pre=runs/rl-v1/policy.pt --synthetic 8-20 --design-seeds 1001 1002 1003 1004 \
    --k 32 --sa-budgets 33 20000 --sa-alt 20000 --json docs/results/rl-v1-heldout.json
```

## Next steps

* SA with the cluster pass in the loop (or the pass applied every N moves)
  as the fair strong baseline; repeat with several seeds and report
  confidence intervals.
* Tune proxy weights so congestion stops dominating; then check against
  routed ORFS results (SG7), since the proxy, not the policy, is the weakest
  link.
* Larger designs (50+ macros) and real DEFs through `cluster_cells`.
