# Scope and decisions (2026-11-06)

User approved PLAN-original.md and asked Claude to use its own recommendations for open questions.

| Question | Decision |
|---|---|
| Compute | Develop in the cloud container (4 CPU cores, 15 GB RAM, no GPU). Run small designs end to end here; large designs (Ariane, BlackParrot) are scripted but only run where compute allows. GPU-scale RL is deferred. |
| PDKs | NanGate45 primary (fakeram45 macros, TILOS comparability). ASAP7 secondary. sky130 only for the OpenLane 2 path. |
| Repo | Local repo now; pushed to GitHub once the user creates one. |
| tiny-gpu | Vendored as a wrapper + macro config inside this repo. The user's GPU repo is not modified. |
| Must-haves | M0 to M4 and M7 (harness, baselines, graph, predictors, flow hook). RL (M5, M6) is the research bet. |

## Sub-goals (each has a Verify command; tests run throughout)

| ID | Sub-goal | Verify |
|---|---|---|
| SG0 | Repo skeleton, venv, pytest, CI workflow | `pytest -q` |
| SG1 | Toolchain: OpenROAD + Yosys + ORFS platforms installed by `scripts/setup_toolchain.sh`; gcd routes | `scripts/smoke_orfs.sh gcd` |
| SG2 | Core model: `Design` (macros, cells/clusters, nets, pins, die/core), LEF/DEF reader, Bookshelf reader, synthetic generator | `pytest tests/test_core*.py tests/test_io*.py` |
| SG3 | Proxy cost: HPWL, density, RUDY congestion, TILOS-style combined proxy | `pytest tests/test_cost*.py` |
| SG4 | Legaliser + baselines: random, grid packing, simulated annealing | `pytest tests/test_legalize*.py tests/test_baselines*.py` |
| SG5 | Graph builder: clustering + PyTorch graph tensors | `pytest tests/test_graph*.py` |
| SG6 | RL env + GNN policy + PPO (CPU), beats random/grid on synthetic designs | `pytest tests/test_place*.py` + `python -m macromind.place.train --smoke` |
| SG7 | Harness: ORFS runner, macro placement injection, PPA JSON | `pytest -m eda tests/test_harness*.py` |
| SG8 | Data generation + congestion/timing predictors | `pytest tests/test_predict*.py` |
| SG9 | tiny-gpu macro variant synthesised and placed | `scripts/run_tinygpu.sh` |
| SG10 | Results report: baselines vs RL on routed PPA | `python -m macromind.report` |
