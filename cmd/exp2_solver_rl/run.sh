#!/usr/bin/env bash
# Experiment 2 (Section 4.2): solver RL on AutoEnvScaling curricula from frontier proposers vs
# a static Tmax pool. Produces Table 3 (and Figure 1, middle). One solver per invocation:
#
#   cmd/exp2_solver_rl/run.sh qwen3.5-9b         # or qwen3.5-35b-a3b
#
# Environment:
#   TMAX_ZIP        TMax-15K tasks.zip (allenai/TMax-15K on the Hugging Face Hub); TMAX_SELECT
#                   optionally names a JSON selection of task ids
#   HF_CHECKPOINT   Hugging Face weights of the solver; REF_LOAD the same in torch_dist format
#   PUBLIC_BASE_URL address under which sandboxes reach the Miles router
#   POLICY_MODEL    served name of the policy in Miles (default "policy")
#   AES_SOLVER_BASE_URL, AES_SOLVER_MODEL   the base solver, served, for calibrating the
#                   initial AutoEnvScaling pool
#   MILES_DIR, MEGATRON_DIR                 checked by `autoenvscaling train plan`
#   PROPOSERS       default "gpt-5.6-sol claude-opus-5"
#   STEPS           solver updates per run (default 300, as in the paper's Exp 2 runs)
#   INITIAL_ENVS    environments generated for the initial AutoEnvScaling pool (default 256)
source "$(dirname "$0")/../common.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$HERE/config.toml"

SOLVER="${1:?usage: cmd/exp2_solver_rl/run.sh qwen3.5-9b|qwen3.5-35b-a3b}"
case "$SOLVER" in
  qwen3.5-9b|qwen3.5-35b-a3b) PRESET="exp2-$SOLVER" ;;
  *) echo "unknown solver $SOLVER" >&2; exit 2 ;;
esac
read -r -a PROPOSERS <<< "${PROPOSERS:-gpt-5.6-sol claude-opus-5}"
STEPS="${STEPS:-300}"
INITIAL_ENVS="${INITIAL_ENVS:-256}"
WEIGHTS=(--hf-checkpoint "${HF_CHECKPOINT:-<hf checkpoint>}" --ref-load "${REF_LOAD:-<torch_dist>}")
BRIDGE_ARGS=(--policy-model "${POLICY_MODEL:-policy}" --public-base-url "${PUBLIC_BASE_URL:-<public url>}")
SELECT=()
if [[ -n "${TMAX_SELECT:-}" ]]; then SELECT=(--select "$TMAX_SELECT"); fi

start_log "exp2_solver_rl/$SOLVER"
require_env TMAX_ZIP HF_CHECKPOINT REF_LOAD PUBLIC_BASE_URL AES_SOLVER_BASE_URL AES_SOLVER_MODEL
N_STATIC="$(num_rollout "$CONFIG" "$STEPS" static)"
N_FRONTIER="$(num_rollout "$CONFIG" "$STEPS" frontier)"

# 1. The Tmax baseline pool, from the released TMax-15K tasks. The training pool references the
#    converted tasks in place; a second pool of the same tasks supplies parents to the proposers.
aes tmax --zip "${TMAX_ZIP:-<tasks.zip>}" --out "$AES_RUN_DIR/tmax-tasks" ${SELECT[@]+"${SELECT[@]}"}
for pool in tmax/pool parents; do
  aes seed --config "$CONFIG" --pool-dir "$AES_RUN_DIR/$pool" --source tmax --no-copy \
      "$AES_RUN_DIR"/tmax-tasks/*
done

# 2. Tmax: solver RL on the static pool (Miles in the background, the loop in the foreground).
miles_start "$PRESET" "$AES_RUN_DIR/tmax" "$N_STATIC" "${WEIGHTS[@]}"
aes run --config "$CONFIG" --mode static --pool-dir "$AES_RUN_DIR/tmax/pool" \
    --run-dir "$AES_RUN_DIR/tmax" --bridge-dir "$AES_RUN_DIR/tmax/bridge" --steps "$STEPS" \
    "${BRIDGE_ARGS[@]}"
miles_wait

# 3. AutoEnvScaling with each frontier proposer (fixed weights, evolving harness). The initial
#    pool is one generation round from the Tmax parents; during training the proposer refills
#    the pool from the solver's latest rollouts at every refresh.
VARIANTS=(autoenvscaling)
if [[ "$SOLVER" == "qwen3.5-35b-a3b" ]]; then VARIANTS+=(no-solver-feedback); fi
for proposer in "${PROPOSERS[@]}"; do
  harness="$(harness_for "$proposer")"
  for variant in "${VARIANTS[@]}"; do
    dir="$AES_RUN_DIR/$proposer/$variant"
    ablate=()
    # Ablation: past solver trajectories removed from the proposer's workspace.
    if [[ "$variant" == "no-solver-feedback" ]]; then ablate=(--no-solver-feedback); fi
    aes generate --config "$CONFIG" --parents-pool "$AES_RUN_DIR/parents" \
        --pool-dir "$dir/pool" --out-dir "$dir/initial" --harness "$harness" \
        --model "$proposer" --assignments "$INITIAL_ENVS" --round-id initial \
        ${ablate[@]+"${ablate[@]}"}
    miles_start "$PRESET" "$dir" "$N_FRONTIER" "${WEIGHTS[@]}"
    aes run --config "$CONFIG" --mode frontier --pool-dir "$dir/pool" --run-dir "$dir" \
        --bridge-dir "$dir/bridge" --steps "$STEPS" --harness "$harness" --model "$proposer" \
        "${BRIDGE_ARGS[@]}" ${ablate[@]+"${ablate[@]}"}
    miles_wait
  done
done

# 4. Evaluation: serve each final checkpoint with an OpenAI-compatible server and run
#    cmd/evals/run.sh <served id> tb2.1   (held-out Terminal-Bench 2.1, avg@5).
note "evaluate each run's final checkpoint with cmd/evals/run.sh <served id> tb2.1"
run "$PYTHON" "$AES_ROOT/analysis/make_all.py" --out-dir "$AES_RUN_DIR/outputs"
