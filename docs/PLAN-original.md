# MacroMind: learned macro placement and PPA prediction for OpenROAD

**Status:** plan for review. No code has been written. Nothing below starts until you say proceed.
**Date:** 2026-11-06
**Working name:** MacroMind (rename freely)

---

## 1. The one-paragraph pitch

Build an open, reproducible ML-for-EDA system that (a) places macros with a GNN-encoded, RL-trained policy, (b) predicts post-route congestion and timing from a placement in milliseconds instead of the tens of minutes a real route plus STA takes, and (c) plugs into OpenROAD-flow-scripts (ORFS) and OpenLane 2 as a drop-in replacement for the default macro placer. Then evaluate it honestly, on real post-route PPA, against strong baselines (OpenROAD's hier-RTLMP, simulated annealing, DREAMPlace mixed-size), on public designs plus a macro-heavy variant of your own tiny-gpu-extended. The deliverable is not "RL beats everything" (the literature says it usually does not); it is a rigorous answer to "when does a learned placer help, by how much, and at what compute cost", backed by a tool other people can run.

## 2. Why this is interesting, and the trap to avoid

- Google's AlphaChip (Mirhoseini et al., *Nature* 2021) claimed RL macro placement beat human experts on TPU blocks. Cheng, Kahng et al. ("Assessment of Reinforcement Learning for Macro Placement", ISPD 2023, and the TILOS `MacroPlacement` repo) re-ran it openly and found simulated annealing and commercial tools matched or beat it, and that the proxy cost RL optimises correlates weakly with real post-route PPA. Follow-ups (MaskPlace NeurIPS'22, ChiPFormer ICML'23, WireMask-BBO NeurIPS'23, EfficientPlace) improved sample efficiency but mostly report proxy metrics on ICCAD04, not routed PPA.
- That gap is the opportunity: an open system that closes the proxy-to-PPA gap with **learned congestion and timing predictors used as the reward**, and that reports **routed** WNS/TNS, wirelength, DRC count and power, is a genuinely publishable contribution and a strong portfolio piece that connects your ML work (nanoGPT Extended) and your hardware work (tiny-gpu-extended).
- The trap is spending months on RL and then comparing against a weak baseline on proxy cost. The plan front-loads baselines and the evaluation harness so every later result is measured the same way.

## 3. Goals and non-goals

### Goals
1. **G1 Evaluation harness first.** One command runs any placer on any benchmark through ORFS to detailed route and emits a JSON PPA record (routed WL, WNS, TNS, power, DRC count, GRT overflow, runtime). Every number in the project comes from this harness.
2. **G2 Learned predictors.** A congestion predictor (placement → GRT overflow map) and a timing predictor (placement → per-endpoint post-route slack) that run in under 100 ms and achieve Kendall tau ≥ 0.7 when ranking candidate placements by true routed PPA.
3. **G3 RL macro placer.** A GNN + policy network that places macros sequentially on a grid, trained with a reward built from G2's predictors plus wirelength/density, with pre-training across designs and per-design fine-tuning.
4. **G4 Flow integration.** A drop-in macro placement step for ORFS (Tcl hook replacing `macro_place`) and an OpenLane 2 custom `Step`, both reading/writing ODB and DEF.
5. **G5 Honest benchmark report.** Routed-PPA comparison vs hier-RTLMP, SA, DREAMPlace, and (where available) hand placements, with seeds, confidence intervals and compute cost, on at least 6 designs across 2 PDKs.
6. **G6 tiny-gpu-extended as a case study.** A synthesisable, macro-heavy configuration of your GPU (SRAM macros for caches, shared memory, register files) hardened through the flow with each placer.

### Non-goals
- Standard-cell placement by RL (OpenROAD's global placer handles cells; we only place macros and possibly soft clusters).
- Routing, CTS or power-grid synthesis innovation.
- Matching AlphaChip's TPU results (proprietary designs, not reproducible).
- Commercial tools. Everything must be open source and reproducible from a Docker image.

## 4. Architecture

```
            RTL / netlist (Yosys, yosys-slang for SystemVerilog)
                              │
                     ORFS: synth → floorplan
                              │  ODB
        ┌─────────────────────┴──────────────────────┐
        │              macromind (Python)             │
        │                                             │
        │  graph/      ODB → hypergraph → clustered   │
        │              graph (macros + std-cell       │
        │              clusters as soft nodes)        │
        │                                             │
        │  predict/    congestion net (GNN + U-Net)   │
        │              timing net (GNN on timing DAG) │
        │                                             │
        │  place/      RL env (grid, masks, reward)   │
        │              policy: GNN encoder + conv/    │
        │              transformer decoder, PPO       │
        │              baselines: SA, random, grid    │
        │                                             │
        │  legalize/   snap, halo, orientation,       │
        │              overlap removal, pin access    │
        └─────────────────────┬──────────────────────┘
                              │  placed macros (ODB / DEF)
              ORFS: global place cells → CTS → route → STA
                              │
                  harness/ → PPA JSON → report/
```

### 4.1 Netlist to graph (`graph/`)
- Read the floorplanned ODB through OpenROAD's Python API (`openroad -python`, `odb` module). Extract macros, std cells, nets, I/O pins, macro pin offsets.
- Cluster std cells into a few hundred to a few thousand soft nodes with OpenROAD's TritonPart (or hMETIS if licensing allows), following the AlphaChip/TILOS clustering recipe so results are comparable with the TILOS numbers.
- Node features: type, area, aspect ratio, pin count, fanin/fanout, logical hierarchy embedding, timing criticality from a pre-placement STA run. Edge features: net weight, net degree, criticality.
- Hypergraph to graph via star or clique expansion (ablate both).

### 4.2 Predictors (`predict/`)
- **Congestion:** input = RUDY map, pin-density map, macro-occupancy map (all as image channels) plus GNN node embeddings scattered onto the grid; output = per-gcell GRT overflow. Architecture: GNN encoder (GraphSAGE or GAT, 4 layers) + U-Net decoder. Prior art: RouteNet (ICCAD'18), CircuitNet baselines.
- **Timing:** GNN over the timing graph that predicts post-route endpoint arrival and slack from placement-stage features, in the style of TimingGCN (DAC'22) and its successors; aggregate to WNS/TNS.
- Both trained on (a) CircuitNet 2.0 for pre-training where its features overlap and (b) our own generated dataset (Section 5.2), which is the main source because it matches our PDKs and flow exactly.
- Calibrated outputs (ensembles or MC dropout) so the RL reward can penalise uncertain regions instead of being exploited.

### 4.3 RL placer (`place/`)
- **Environment:** a G×G grid (G = 32 to 128) over the core area. One macro placed per step, largest first (ablate ordering). Action = grid cell; invalid cells masked (overlap, halo, outside core, blockages). After all macros are placed, std-cell clusters are placed by a fast force-directed or DREAMPlace pass, as in AlphaChip.
- **Reward:** terminal reward = −(α·HPWL + β·density + γ·predicted congestion + δ·predicted TNS), with α..δ normalised per design. A small fraction of episodes (for example 1 in 200) runs the real ORFS route + STA to recalibrate the predictors online (active learning on the policy's own distribution, which addresses the proxy-exploitation problem).
- **Policy:** GNN encoder for the netlist (shared with predictors where useful), a current-state image encoder (placed macros, density, RUDY), and a deconvolution or transformer decoder producing logits over grid cells. Value head for PPO. Alternatives to ablate: MaskPlace-style pixel masks, ChiPFormer-style offline decision transformer.
- **Training regime:** pre-train across all training designs, then fine-tune per target design for a fixed budget. Report zero-shot, fine-tuned, and from-scratch, each with wall-clock and GPU-hours.
- **Legaliser:** deterministic snap to placement sites and routing tracks, orientation choice (R0/MX/MY/R180) by local HPWL, halo enforcement, and a final overlap-removal pass, so every placer (including baselines) goes through the same legaliser.

### 4.4 Flow integration (`flow/`)
- **ORFS:** a Tcl hook that replaces the `macro_place` (hier-RTLMP) call in `2_2_floorplan_macro` with `python -m macromind.place --odb in.odb --out out.odb`. Config switch `MACRO_PLACER=macromind|rtlmp|sa|dreamplace`.
- **OpenLane 2:** a custom `Step` subclass that runs before `OpenROAD.GlobalPlacement` and writes macro placements, plus a Flow definition that swaps it in. This is the path for sky130/gf180 users.
- Pinned Docker image: ORFS release tag + PyTorch + PyG, so results are reproducible.

### 4.5 Harness and reporting (`harness/`, `report/`)
- Job runner that fans out (design × placer × seed) runs on local cores or a cloud batch queue, caches by content hash, and writes PPA JSON.
- Report generator: tables with mean ± 95% CI over ≥ 5 seeds, Pareto plots (WNS vs WL vs power), proxy-vs-routed correlation scatter plots, layout images per design.

## 5. Datasets and benchmarks

### 5.1 Evaluation designs
| Design | Source | PDK(s) | Macros | Why |
|---|---|---|---|---|
| Ariane (CVA6) 133/136 macros | TILOS MacroPlacement, ORFS | NanGate45, ASAP7 | 133–136 | Direct comparison with the AlphaChip assessment paper |
| BlackParrot (quad core) | TILOS MacroPlacement | NanGate45 | ~220 | Large, many macros |
| MemPool tile / group | TILOS MacroPlacement | NanGate45 | 20–300 | Memory-dominated |
| NVDLA partition c | TILOS MacroPlacement | NanGate45 | 128 | Accelerator, relevant to ML hardware |
| bp_fe / bp_be / swerv_wrapper | ORFS designs | NanGate45 / sky130hd | 10–30 | Smaller, fast iteration |
| ICCAD04 IBM mixed-size | Public | (abstract) | 100s | Proxy-only comparison with MaskPlace/ChiPFormer/WireMask numbers |
| **tiny-gpu-extended (macro variant)** | Your repo | sky130hd, NanGate45 | 16–64 | Case study, see 5.3 |

GF12 testcases in the TILOS repo need a commercial PDK and are excluded.

### 5.2 Generated training data (our main dataset)
- For each training design, generate placements from a mix of: hier-RTLMP with randomised parameters, SA at different temperatures, random legal placements, and (later) the RL policy itself.
- Run each through ORFS to global route + STA (cheap label: GRT overflow, estimated parasitics STA) and a subset to detailed route (expensive label).
- Target: about 3,000 global-route labels and 500 detailed-route labels across 6 to 8 training designs, held out by design (never train and test on the same design).
- Store as PyG `Data` objects plus image tensors, versioned with DVC or a plain content-hash manifest.

### 5.3 tiny-gpu-extended macro variant
Your GPU today has no SRAM macros (shared memory, caches and register files are flip-flop arrays), and its SystemVerilog goes through sv2v for Icarus. Hardening it needs a small, separate side project inside this plan:
- Synthesise with Yosys + yosys-slang (or sv2v → Yosys) at a reduced configuration first to get a baseline area and timing.
- Replace `shared_memory`, `l1d_cache` data arrays and the I-cache with SRAM macros: sky130 OpenRAM macros (`sky130_sram_1kbyte_1rw1r_32x256_8` and friends) or DFFRAM for sky130; fakeram45 for NanGate45 (what ORFS already uses).
- Scale `NUM_CORES` up (for example 8 to 16 cores) so there are enough macros (16 to 64) for placement to matter.
- This does not change the GPU repo's main branch; it is a new config plus a wrapper, ideally on a branch in Didactic_GPU_Simulator or vendored into this project.

## 6. Baselines (all through the same legaliser and harness)
1. **hier-RTLMP** (OpenROAD's default `macro_place`), the "default flow" the brief asks us to beat.
2. **Simulated annealing** on the same proxy cost (the TILOS paper's strongest open baseline).
3. **DREAMPlace mixed-size** global placement with macro legalisation.
4. **Random legal** and **grid packing** sanity baselines.
5. **Hand/expert placement** where TILOS or ORFS ship one.
6. **RL with proxy-only reward** (no learned predictors), the ablation that tests the central claim.

## 7. Milestones

Each milestone ends with a gate you can check. Durations assume one person part-time with Claude doing most implementation; they are rough.

| # | Milestone | Deliverable | Gate | Est. |
|---|---|---|---|---|
| M0 | Repo, Docker, CI | `macromind` repo, pinned ORFS + PyTorch image, CI running unit tests and one tiny ORFS design (gcd) end-to-end | CI green; `gcd` routes in the container | 1 wk |
| M1 | Evaluation harness | Runner + PPA JSON + report generator; reproduces ORFS default results on 3 designs | Re-running default flow twice gives identical PPA (or documented nondeterminism bounds) | 2 wk |
| M2 | Baselines | SA, random, grid, DREAMPlace adapters, shared legaliser; first baseline table on 4 designs × 5 seeds | Table reproduces published TILOS trends for Ariane133 within reason | 2–3 wk |
| M3 | Graph + data generation | ODB → graph, clustering, dataset generator; first 1,000 GRT-labelled placements | Dataset manifest; label spot-checks match ORFS reports | 2–3 wk |
| M4 | Predictors | Congestion and timing nets, held-out-design evaluation | Kendall tau ≥ 0.7 ranking placements by routed TNS and overflow on unseen designs; ≤ 100 ms inference | 3–4 wk |
| M5 | RL placer v1 | PPO env, policy, proxy-only reward; single-design training | Beats random and grid; within 5% of SA on proxy cost on 2 designs | 3–4 wk |
| M6 | RL placer v2 | Predictor-based reward, online recalibration, multi-design pre-training + fine-tune | On held-out designs, routed TNS or WL better than hier-RTLMP on ≥ half of designs, or a clear documented negative result | 4–6 wk |
| M7 | Flow integration | ORFS hook + OpenLane 2 Step, docs, example configs | A user runs `make DESIGN_CONFIG=... MACRO_PLACER=macromind` and gets a routed design | 1–2 wk |
| M8 | tiny-gpu case study | Macro variant of your GPU hardened with every placer | Routed results table + layout images for tiny-gpu-macro | 2–3 wk |
| M9 | Report | Full benchmark report, ablations, compute accounting, write-up suitable for a workshop paper or blog | Every number regenerable from one command and a cache | 2 wk |

Total: roughly 6 to 8 months part-time. M0 to M4 alone (harness, baselines, predictors) is a self-contained, valuable result if RL is later dropped.

## 8. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| RL does not beat SA or hier-RTLMP on routed PPA (this is what the literature predicts) | High | High | Framing: the project's claim is the honest comparison and the predictor-as-reward idea. The ablation (proxy-only vs predictor reward) is a result either way. M4 predictors are useful alone, for example to rank SA candidates. |
| Policy exploits predictor errors | High | Medium | Ensembles with uncertainty penalty; online recalibration with real routes; always report routed, never predicted, PPA. |
| ORFS runtime makes data generation slow (Ariane ~1–3 h to detailed route) | High | High | Label most samples at global route only; detailed route a subset; cache aggressively; start on small designs. |
| Few designs with macros, so predictors overfit | Medium | High | Leave-one-design-out evaluation; synthetic macro-heavy variants (tiny-gpu scaled, MemPool configs); CircuitNet pre-training. |
| OpenROAD nondeterminism across runs/threads | Medium | Medium | Pin versions, single-thread for evaluation runs, 5+ seeds, report CIs. |
| SystemVerilog support for tiny-gpu in Yosys | Medium | Low | sv2v fallback (already used in that repo for Icarus). |
| sky130 OpenRAM macros have awkward pin access / LEF issues | Medium | Medium | Use fakeram45 on NanGate45 as the primary tiny-gpu run; sky130 is secondary. |
| Scope creep into std-cell placement or routing | Medium | Medium | Non-goals are explicit; any expansion needs your sign-off. |

## 9. Compute needs

This cloud container has no OpenROAD and no GPU, so real runs need a separate machine. Two realistic options:

| Resource | Use | Estimate |
|---|---|---|
| CPU (16–64 cores, 64–128 GB RAM) | ORFS runs for data generation and evaluation | ~3,000 GRT runs × ~10 min + ~500 DRT runs × ~60 min ≈ 1,000 core-hours for M3/M4; another ~1,000–1,500 core-hours for M6 and M9 evaluation |
| GPU (1 × 24 GB, e.g. RTX 4090 or A10G; A100 nice to have) | Predictor training, PPO training | Predictors: tens of GPU-hours. RL: ~50–200 GPU-hours across ablations (sample-efficient methods; AlphaChip-scale is out of scope) |
| Storage | ODB/DEF, reports, datasets | 200–500 GB |
| Rough cloud cost | Spot CPU + one GPU | about $500 to $1,500 total over the project, dominated by ORFS CPU time. A local workstation with a 24 GB GPU and 16+ cores covers it with no cloud spend. |

## 10. Repo layout (proposed)

```
macromind/
  docker/            pinned ORFS + PyTorch image
  macromind/
    graph/  predict/  place/  legalize/  flow/  harness/  report/
  configs/           per-design, per-PDK flow configs
  designs/           tiny-gpu-macro wrapper + SRAM configs
  data/              dataset manifests (not data)
  experiments/       one YAML per experiment, results cached by hash
  docs/              ADRs, PLAN.md, milestone reports
  tests/
```

Same discipline as tiny-gpu-extended: ADRs for big choices, a PLAN.md tracker with Verify commands per sub-goal, and independent audits after each milestone.

## 11. Decisions I need from you before starting

1. **Compute:** do you have a local workstation (cores + GPU), or should the plan assume cloud spot instances? This sets the dataset size.
2. **PDKs:** NanGate45 + ASAP7 as primary (best comparability with published work), with sky130 for the OpenLane 2 path and tiny-gpu? That is my recommendation.
3. **New repo:** a new GitHub repo (for example `macromind`) that you create, which I then attach. I cannot create repos from here.
4. **tiny-gpu changes:** OK to add a macro-variant config on a branch of Didactic_GPU_Simulator, or keep it vendored inside the new repo only?
5. **Scope cut if time is short:** my suggestion is to keep M0–M4 + M7 as the must-haves and treat RL (M5–M6) as the research bet.

## 12. Key references
- Mirhoseini et al., "A graph placement methodology for fast chip design", *Nature* 2021 (AlphaChip / Circuit Training).
- Cheng, Kahng, Kundu, Wang, Wang, "Assessment of Reinforcement Learning for Macro Placement", ISPD 2023; TILOS-AI-Institute/MacroPlacement repo.
- Lai et al., "MaskPlace", NeurIPS 2022; Lai et al., "ChiPFormer", ICML 2023; Shi et al., "Macro Placement by Wire-Mask-Guided Black-Box Optimization", NeurIPS 2023.
- Kahng et al., "RTL-MP" / Kahng, Varadarajan, Wang, "Hier-RTLMP", ISPD 2022/2023.
- Lin et al., "DREAMPlace", DAC 2019 / TCAD.
- Xie et al., "RouteNet", ICCAD 2018; Guo et al., "A Timing Engine Inspired Graph Neural Network Model for Pre-Routing Slack Prediction" (TimingGCN), DAC 2022.
- Chai et al., "CircuitNet 2.0", ICLR 2024.
- OpenROAD-flow-scripts and OpenLane 2 documentation.
