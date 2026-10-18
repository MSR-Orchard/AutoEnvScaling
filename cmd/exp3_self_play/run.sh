#!/usr/bin/env bash
# Experiment 3 (Section 4.3): Qwen3.6-35B-A3B as both proposer and solver, trained on its own
# environments, vs the same model without proposer training and vs a static Tmax pool.
# Produces Figures 6, 7 and 20 (and Figure 1, right).
#
#   COLD_START_HF=... COLD_START_REF=... SEED_TASKS="<parent task dirs>" TMAX_TASKS=<dir> \
#     cmd/exp3_self_play/run.sh
#
# Environment:
#   COLD_START_HF, COLD_START_REF  the Cold-Start checkpoint (cmd/cold_start), Hugging Face and
#                   torch_dist formats; all three runs start from it
#   SEED_TASKS      parent Harbor task directories for the initial pool
#   TMAX_TASKS      Harbor tasks of the static baseline pool (from `autoenvscaling tmax`)
#   POLICY_MODEL    served name of the policy (default "policy")
#   PUBLIC_BASE_URL address under which sandboxes reach the Miles router
#   AES_PROPOSER_BASE_URL, AES_SOLVER_BASE_URL, AES_SOLVER_MODEL   the Cold-Start checkpoint,
#                   served, for generating and calibrating the initial pool
#   MILES_DIR, MEGATRON_DIR     checked by `autoenvscaling train plan`
#   STEPS           solver updates (default 288; the ablation collapsed at 160)
#   INITIAL_ENVS    environments generated for the initial pool (default 256)
source "$(dirname "$0")/../common.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$HERE/config.toml"

STEPS="${STEPS:-288}"
INITIAL_ENVS="${INITIAL_ENVS:-256}"
POLICY="${POLICY_MODEL:-policy}"
read -r -a SEEDS <<< "${SEED_TASKS:-<parent task dirs>}"
TMAX_TASKS="${TMAX_TASKS:-<tmax task dir>}"
WEIGHTS=(--hf-checkpoint "${COLD_START_HF:-<cold-start hf>}" --ref-load "${COLD_START_REF:-<cold-start torch_dist>}")
BRIDGE_ARGS=(--policy-model "$POLICY" --public-base-url "${PUBLIC_BASE_URL:-<public url>}")

start_log exp3_self_play
require_env COLD_START_HF COLD_START_REF SEED_TASKS TMAX_TASKS PUBLIC_BASE_URL \
            AES_PROPOSER_BASE_URL AES_SOLVER_BASE_URL AES_SOLVER_MODEL
N_SHARED="$(num_rollout "$CONFIG" "$STEPS" shared)"
# Without proposer training the loop still refreshes the pool but never updates the proposer,
# which is the frontier schedule.
N_NO_PROPOSER="$(num_rollout "$CONFIG" "$STEPS" frontier)"
N_STATIC="$(num_rollout "$CONFIG" "$STEPS" static)"

# 1. Initial pool: one generation round by the Cold-Start policy in the Pi harness, parents from
#    SEED_TASKS. Both self-play runs start from a copy of it.
aes seed --config "$CONFIG" --pool-dir "$AES_RUN_DIR/parents" --source seed --no-copy \
    "${SEEDS[@]}"
aes generate --config "$CONFIG" --parents-pool "$AES_RUN_DIR/parents" \
    --pool-dir "$AES_RUN_DIR/initial/pool" --out-dir "$AES_RUN_DIR/initial" --harness pi \
    --model "$POLICY" --assignments "$INITIAL_ENVS" --round-id initial

# 2. AutoEnvScaling and the ablation without proposer training (shared weights, proposer loss
#    removed). The solver trains every step; every 16 solver steps the pool is reviewed, the
#    policy generates replacements (128 trajectories per assignment) and, in the full run,
#    trains on r_P.
for arm in autoenvscaling no-proposer-training; do
  dir="$AES_RUN_DIR/$arm"
  run mkdir -p "$dir"
  run cp -R "$AES_RUN_DIR/initial/pool" "$dir/pool"
  if [[ "$arm" == "autoenvscaling" ]]; then n="$N_SHARED"; ablate=();
  else n="$N_NO_PROPOSER"; ablate=(--no-proposer-training); fi
  miles_start shared-policy "$dir" "$n" "${WEIGHTS[@]}"
  aes run --config "$CONFIG" --mode shared --pool-dir "$dir/pool" --run-dir "$dir" \
      --bridge-dir "$dir/bridge" --steps "$STEPS" "${BRIDGE_ARGS[@]}" ${ablate[@]+"${ablate[@]}"}
  miles_wait
done

# 3. Tmax: solver RL on the static pool from the same checkpoint.
aes seed --config "$CONFIG" --pool-dir "$AES_RUN_DIR/tmax/pool" --source tmax --no-copy \
    "$TMAX_TASKS"/*
miles_start shared-policy "$AES_RUN_DIR/tmax" "$N_STATIC" "${WEIGHTS[@]}"
aes run --config "$CONFIG" --mode static --pool-dir "$AES_RUN_DIR/tmax/pool" \
    --run-dir "$AES_RUN_DIR/tmax" --bridge-dir "$AES_RUN_DIR/tmax/bridge" --steps "$STEPS" \
    "${BRIDGE_ARGS[@]}"
miles_wait

# 4. Figure 6: every 32 updates, serve the saved checkpoint and run
#    cmd/evals/run.sh <served id> both      (Terminal-Bench 2.1 and 4.0, avg@5).
note "evaluate the checkpoints saved every 32 updates with cmd/evals/run.sh <served id> both"
# Figures 7 and 20: each run's metrics.jsonl records timestamped solver_update events (groups
# kept, environments tried, grouping statistics) and generate events (the round's valid_rate).
not_in_release "Figure 7 and 20 data" "conversion of each run's metrics.jsonl into the figure data files"
run "$PYTHON" "$AES_ROOT/analysis/make_all.py" --out-dir "$AES_RUN_DIR/outputs"
