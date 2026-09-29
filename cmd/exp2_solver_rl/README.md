# Experiment 2: model-designed curricula vs static environment pools

Section 4.2; Table 3 (and the middle panel of Figure 1).

Qwen3.5-9B and Qwen3.5-35B-A3B are trained as solvers with RL. The proposer is GPT-5.6-sol
(in Codex) or Claude Opus 5 (in Claude Code); it keeps its weights fixed and improves only
through its harness.

- **Tmax**: a static pool, fixed during training (`selfplay run --mode static`).
- **AutoEnvScaling**: the proposer refills the pool from the solver's latest rollouts at every
  refresh (`selfplay run --mode frontier`).
- **AutoEnvScaling (w/o solver feedback)**, Qwen3.5-35B-A3B only: past solver trajectories are
  removed from the proposer's workspace (`--no-solver-feedback`).

Each run is a Miles job in bridge mode (`autoenvscaling train run --preset exp2-<solver>`) fed by
the self-play loop (`autoenvscaling selfplay run`) through a shared bridge directory; the
launcher starts the Miles job in the background and the loop in the foreground. Evaluation:
held-out Terminal-Bench 2.1, avg@5, at matched training steps (`cmd/evals`).

```bash
export TMAX_ZIP=tasks.zip HF_CHECKPOINT=... REF_LOAD=... PUBLIC_BASE_URL=...
export AES_SOLVER_BASE_URL=... AES_SOLVER_MODEL=...     # base solver, for the initial pool
export MILES_DIR=... MEGATRON_DIR=...
cmd/exp2_solver_rl/run.sh qwen3.5-9b                     # or qwen3.5-35b-a3b
DRY_RUN=1 cmd/exp2_solver_rl/run.sh qwen3.5-35b-a3b
```

## Commands

| Step | Command |
|---|---|
| Tmax tasks from TMax-15K | `autoenvscaling tmax --zip tasks.zip --out D [--select S]` |
| Pools | `autoenvscaling selfplay seed --pool-dir P --source tmax --no-copy D/*` |
| Initial AutoEnvScaling pool | `autoenvscaling selfplay generate --parents-pool P --pool-dir Q --harness H --model M --assignments 256 [--no-solver-feedback]` |
| Miles updates per run | `autoenvscaling selfplay schedule --steps S --mode static\|frontier` |
| Miles job | `autoenvscaling train plan\|run --preset exp2-<solver> --bridge-dir B --hf-checkpoint ... --ref-load ... --save ... --num-rollout N` |
| Training loop | `autoenvscaling selfplay run --mode static\|frontier --pool-dir Q --run-dir R --bridge-dir B --steps S [--harness H --model M] [--no-solver-feedback]` |
| Evaluation | `cmd/evals/run.sh <served checkpoint> tb2.1` |

## Settings

| Setting | Value | Source |
|---|---|---|
| Solvers | Qwen3.5-9B, Qwen3.5-35B-A3B | Section 4.2 |
| Proposers | GPT-5.6-sol (Codex), Claude Opus 5 (Claude Code) | Section 4.2, Appendix D.2 |
| Group size / environments per batch | 16 / 16 | Section 4.2 |
| Maximum sequence length | 128k tokens (the `exp2-*` presets) | Section 4.2 |
| Admission | validation, then calibration: mean reward in [0.25, 0.75], std >= 0.1, 4 rollouts (8 near a band edge), up to 2 revisions | Section 3.2, Appendix G.4 |
| Solver sandbox | 4 CPUs, 8 GiB, Pi | Appendix D.2 |
| Evaluation | Terminal-Bench 2.1 avg@5 | Section 4.2 |

## Differences from the paper and gaps

- The paper regenerated the Tmax pool with each proposer, following the Tmax recipe, so each
  proposer had its own Tmax baseline. This release has no Tmax regeneration pipeline; the
  baseline pool comes from the released TMax-15K tasks, so there is one Tmax run per solver.
- How the initial AutoEnvScaling pool was built is not stated in the paper. The launcher
  generates `INITIAL_ENVS` (default 256, the pool size) environments with the proposer, parents
  drawn from the Tmax tasks and calibrated against the base solver.
- The number of matched solver updates is not stated in the paper; `STEPS` defaults to the
  preset's 300.
- Serving a Miles checkpoint as an OpenAI-compatible endpoint for evaluation is not part of this
  package.
