#!/usr/bin/env bash
# Experiment 1 (Section 4.1): Prompt vs Prompt + Env Feedback vs AutoEnvScaling across eight
# proposer models. Produces Table 2 and Figures 4-5.
#
#   ASSIGNMENTS=<assignments.jsonl> SEED_TASKS="<parent Harbor task dirs>" cmd/exp1_authoring/run.sh
#
# Environment:
#   ASSIGNMENTS    assignments for the two prompt-based arms (JSON lines; see `baselines --help`)
#   SEED_TASKS     parent Harbor task directories the AutoEnvScaling arm draws assignments from
#   MODELS         space-separated proposer ids (default: the paper's eight)
#   METHODS        subset of "prompt feedback harness" (default: all three)
#   N_PROMPT       assignments per prompt-based cell (default 400)
#   N_HARNESS      assignments per AutoEnvScaling cell (default 200)
#   USD_IN, USD_OUT  prices per million input / output tokens, for cost per valid environment
#   AES_PROPOSER_BASE_URL, AES_PROPOSER_API_KEY   endpoint for the prompt arms and Pi proposers
#   AES_SOLVER_BASE_URL, AES_SOLVER_MODEL         solver for the pipeline's calibration step
source "$(dirname "$0")/../common.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$HERE/config.toml"

read -r -a MODELS <<< "${MODELS:-claude-opus-5 gpt-6-astra gpt-5.6-sol gpt-5.6-terra gpt-5.6-luna gemini-3.8-flash grok-4.6 deepseek-v4-flash}"
read -r -a METHODS <<< "${METHODS:-prompt feedback harness}"
read -r -a SEEDS <<< "${SEED_TASKS:-<parent task dirs>}"
ASSIGNMENTS="${ASSIGNMENTS:-<assignments.jsonl>}"
PRICES=()
if [[ -n "${USD_IN:-}" && -n "${USD_OUT:-}" ]]; then
  PRICES=(--usd-per-mtok-in "$USD_IN" --usd-per-mtok-out "$USD_OUT")
fi
has() { [[ " ${METHODS[*]} " == *" $1 "* ]]; }

start_log exp1_authoring

# 1. Prompt and Prompt + Env Feedback: the model is called directly, without a coding agent.
#    The feedback arm reuses the prompt arm's first round from the same --out directory.
for model in "${MODELS[@]}"; do
  for arm in prompt feedback; do
    has "$arm" || continue
    aes baselines generate --config "$CONFIG" --arm "$arm" --model "$model" \
        --assignments "$ASSIGNMENTS" --limit "${N_PROMPT:-400}" \
        --out "$AES_RUN_DIR/baselines/$model"
  done
  if has prompt || has feedback; then
    aes baselines summarize --out "$AES_RUN_DIR/baselines/$model" ${PRICES[@]+"${PRICES[@]}"}
  fi
done

# 2. AutoEnvScaling: the model works in its coding agent inside the proposer sandbox. The host
#    validates each submitted task (and calibrates valid ones); the round summary reports
#    valid_rate, and every episode's checks are in proposer_trajectories.jsonl.
if has harness; then
  aes seed --config "$CONFIG" --pool-dir "$AES_RUN_DIR/parents" --source seed \
      --no-copy "${SEEDS[@]}"
  for model in "${MODELS[@]}"; do
    aes generate --config "$CONFIG" --parents-pool "$AES_RUN_DIR/parents" \
        --pool-dir "$AES_RUN_DIR/harness/$model/pool" --out-dir "$AES_RUN_DIR/harness/$model" \
        --harness "$(harness_for "$model")" --model "$model" \
        --assignments "${N_HARNESS:-200}" --round-id "exp1-$model"
  done
fi

# 3. Figure 5 needs solver turns and pass rate on every arm's admitted environments;
#    `autoenvscaling solver <task_dir> -n 4` measures one task at a time.
not_in_release "Figure 5 measurements" "a batch runner over the admitted tasks of all three arms"

# 4. Table 11 compares AutoEnvScaling with and without the reference-solution self-check.
not_in_release "Table 11 self-check ablation" "a switch that disables the proposer's self-check"

# 5. Tables and figures from the released numbers.
run "$PYTHON" "$AES_ROOT/analysis/make_all.py" --out-dir "$AES_RUN_DIR/outputs"
