# Experiment 3: recursive self-improvement

Section 4.3 and Appendix G; Figures 6, 7 and 20 (and the right panel of Figure 1).

Qwen3.6-35B-A3B plays both roles: it designs training environments as proposer, trains on them
as solver, and the updated checkpoint designs the next round. All runs start from the
Cold-Start checkpoint (`cmd/cold_start`):

- **AutoEnvScaling**: `selfplay run --mode shared`, with proposer and solver rewards and
  separate advantage baselines per role.
- **AutoEnvScaling (w/o proposer training)**: `--no-proposer-training`, same shared weights,
  proposer loss removed. In the paper, training collapsed at step 160 as environment validation
  errors grew.
- **Tmax**: `selfplay run --mode static` on a Tmax pool.

Each run is a Miles job (`autoenvscaling train run --preset shared-policy`, bridge mode) fed by
the self-play loop through a shared bridge directory; the launcher starts the Miles job in the
background and the loop in the foreground.

```bash
export COLD_START_HF=... COLD_START_REF=... PUBLIC_BASE_URL=... MILES_DIR=... MEGATRON_DIR=...
export AES_PROPOSER_BASE_URL=... AES_SOLVER_BASE_URL=... AES_SOLVER_MODEL=...  # Cold-Start, served
SEED_TASKS="tasks/*" TMAX_TASKS=pools/tmax15k cmd/exp3_self_play/run.sh
DRY_RUN=1 cmd/exp3_self_play/run.sh
```

## Commands

| Step | Command |
|---|---|
| Parent pool | `aes seed --pool-dir P --source seed --no-copy <tasks>` |
| Initial pool | `aes generate --parents-pool P --pool-dir Q --harness pi --model <policy> --assignments 256` |
| Miles updates per run | `aes schedule --steps 288 --mode shared` (305 updates: 288 solver + 17 proposer) |
| Miles job | `autoenvscaling train plan\|run --preset shared-policy --bridge-dir B --hf-checkpoint ... --ref-load ... --save ... --num-rollout N` |
| Training loop | `aes run --mode shared\|static --pool-dir Q --run-dir R --bridge-dir B --steps 288 [--no-proposer-training]` |
| Tmax tasks | `autoenvscaling tmax --zip tasks.zip --out pools/tmax15k` |
| Evaluation | `cmd/evals/run.sh <served checkpoint> both` |

## Settings (Table 12 and Appendix G)

| Setting | Value |
|---|---|
| Learning rate | 1e-6 |
| Solver rollout batch | 16 environments x 16 trajectories |
| Proposer rollout group | 128 proposer trajectories, one shared assignment |
| Proposer generation interval (k) | every 16 solver updates |
| Active environment pool | 256 tasks, reviewed every 16 solver updates |
| Policy loss | DPPO, total-variation threshold 0.1; KL and entropy coefficients 0 |
| Advantages | mean-centred per group (no std division); zero-variance groups dropped |
| Proposer reward | -1 invalid, -0.25 valid but not calibrated, 1 admitted |
| Rollouts | oversample until 16 non-degenerate groups; unfinished trajectories carried across updates |
| Hardware | Miles on 4-8 B200 GPUs |
| Evaluation | Terminal-Bench 2.1 and 4.0, avg@5, every 32 updates |
| Evaluation range | 288 updates (full system, Tmax); 160 (ablation) |

Figure 7 compares rollout collection over updates 0-162, the range logged for both methods,
with matched hardware and rollout concurrency. Its Tmax curve continues to step 200 with a
log-linear extrapolation fitted to steps 93-162, marked in the data.

## Gaps

- How the initial pool was built is not stated in the paper; the launcher runs one generation
  round with the Cold-Start policy from `SEED_TASKS` parents.
- The paper's Tmax pool for this experiment is not released; `TMAX_TASKS` can be the TMax-15K
  conversion from `autoenvscaling tmax`.
- Serving saved checkpoints as OpenAI-compatible endpoints for evaluation is not part of this
  package.
- Not in this release yet: turning each run's `metrics.jsonl` into the data files behind
  Figures 7 and 20. The figures themselves render from the released numbers
  (`python analysis/make_all.py`).
