#!/usr/bin/env bash
# Cold-Start (Section 4.3, Appendix H): fine-tune Qwen3.6-35B-A3B on 600 high-quality proposer
# trajectories so it can start self-play. Produces Figures 21-22 and Table 13.
#
#   SEED_TASKS="<parent task dirs>" BASE_HF=... BASE_REF=... cmd/cold_start/run.sh
#
# Environment:
#   SEED_TASKS      parent Harbor task directories for the proposer episodes
#   TEACHERS        default "claude-opus-5 deepseek-v4-flash kimi-k3"
#   N_EPISODES      teacher episodes per model (default 300)
#   N_TRAJECTORIES  fine-tuning trajectories kept (default 600)
#   BASE_HF, BASE_REF   Qwen3.6-35B-A3B weights (Hugging Face and torch_dist)
#   AES_PROPOSER_BASE_URL, AES_PROPOSER_API_KEY   endpoint for Pi teachers (and the served
#                   base / Cold-Start models in step 3)
#   AES_SOLVER_BASE_URL, AES_SOLVER_MODEL         solver for calibration
#   MILES_DIR, MEGATRON_DIR                       checked by `autoenvscaling train plan`
#   BASE_MODEL, COLD_START_MODEL   served names of the two checkpoints for step 3
source "$(dirname "$0")/../common.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$HERE/config.toml"

read -r -a TEACHERS <<< "${TEACHERS:-claude-opus-5 deepseek-v4-flash kimi-k3}"
read -r -a SEEDS <<< "${SEED_TASKS:-<parent task dirs>}"
N_EPISODES="${N_EPISODES:-300}"
N_TRAJECTORIES="${N_TRAJECTORIES:-600}"

start_log cold_start
require_env SEED_TASKS BASE_HF BASE_REF AES_SOLVER_BASE_URL AES_SOLVER_MODEL

# 1. Proposer episodes from the teachers, each in its coding agent. Admitted episodes (r_P = 1)
#    are the candidates for fine-tuning.
aes selfplay --config "$CONFIG" seed --pool-dir "$AES_RUN_DIR/parents" --source seed --no-copy \
    "${SEEDS[@]}"
for teacher in "${TEACHERS[@]}"; do
  aes selfplay --config "$CONFIG" generate --parents-pool "$AES_RUN_DIR/parents" \
      --pool-dir "$AES_RUN_DIR/teachers/$teacher/pool" --out-dir "$AES_RUN_DIR/teachers/$teacher" \
      --harness "$(harness_for "$teacher")" --model "$teacher" --assignments "$N_EPISODES" \
      --round-id teacher
done

# 2. Keep admitted, deduplicated episodes (closest to the middle of the calibration band first)
#    and fine-tune the base model on them.
aes export sft "$AES_RUN_DIR"/teachers/*/teacher/proposer_trajectories.jsonl \
    --limit "$N_TRAJECTORIES" --out "$AES_RUN_DIR/sft.jsonl"
TRAIN=(--preset cold-start --prompt-data "$AES_RUN_DIR/sft.jsonl"
       --hf-checkpoint "${BASE_HF:-<base hf>}" --ref-load "${BASE_REF:-<base torch_dist>}"
       --save "$AES_RUN_DIR/checkpoint" --log "$AES_RUN_DIR/miles.log")
aes train plan "${TRAIN[@]}"
aes train run "${TRAIN[@]}"

# 3. Environment generation before and after (Figures 21-22, Table 13 acceptance). With the same
#    parent pool and seed, both models get the same assignments. Serve each checkpoint at
#    AES_PROPOSER_BASE_URL under its name first.
note "serve the base and Cold-Start checkpoints before step 3; set BASE_MODEL and COLD_START_MODEL"
for model in "${BASE_MODEL:-<served base>}" "${COLD_START_MODEL:-<served cold-start>}"; do
  for suite in 60 136; do
    aes selfplay --config "$CONFIG" generate --parents-pool "$AES_RUN_DIR/parents" \
        --pool-dir "$AES_RUN_DIR/eval$suite/$model/pool" --out-dir "$AES_RUN_DIR/eval$suite/$model" \
        --harness pi --model "$model" --assignments "$suite" --seed "$suite" \
        --round-id "eval$suite"
  done
done

# 4. Solving before and after (Table 13): cmd/evals/run.sh <served id> tb2.1
#    (avg@5 and pass@5 over the 89 Terminal-Bench 2.1 tasks).
note "evaluate both checkpoints with cmd/evals/run.sh <served id> tb2.1"
run "$PYTHON" "$AES_ROOT/analysis/make_all.py" --out-dir "$AES_RUN_DIR/outputs"
