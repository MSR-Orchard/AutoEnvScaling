#!/usr/bin/env bash
# Terminal-Bench 2.1 / 4.0 evaluation of one model checkpoint: avg@5 with the Pi agent.
#
#   AES_EVAL_BASE_URL=... AES_EVAL_API_KEY=... cmd/evals/run.sh <served model id> [tb2.1|tb4|both]
#
# Extra flags for `autoenvscaling eval run` can be passed in EVAL_ARGS, e.g.
# EVAL_ARGS="--sampling qwen-solving --concurrency 32".
source "$(dirname "$0")/../common.sh"

MODEL="${1:?usage: cmd/evals/run.sh <served model id> [tb2.1|tb4|both]}"
WHICH="${2:-both}"
K="${K:-5}"
HARBOR_ENV="${HARBOR_ENV:-modal}"
read -r -a EXTRA <<< "${EVAL_ARGS:-}"

start_log evals
require_env AES_EVAL_BASE_URL AES_EVAL_API_KEY

case "$WHICH" in
  both) BENCHMARKS=(tb2.1 tb4) ;;
  tb2.1|tb4) BENCHMARKS=("$WHICH") ;;
  *) echo "unknown benchmark $WHICH" >&2; exit 2 ;;
esac

# One cheap request first: a wrong URL or model id otherwise shows up as hundreds of failed trials.
aes eval preflight --model "$MODEL"

for bench in "${BENCHMARKS[@]}"; do
  aes eval run --benchmark "$bench" --model "$MODEL" -k "$K" \
      --env "$HARBOR_ENV" --jobs-dir "$AES_RUN_DIR/jobs" ${EXTRA[@]+"${EXTRA[@]}"}
done
