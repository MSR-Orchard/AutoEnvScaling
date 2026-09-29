#!/usr/bin/env bash
# Cross-domain environment generation (Appendix C): three coding agents generate Harbor
# environments in seven domains. Produces Table 8 (and the companion acceptance figure).
#
#   SEED_TASKS="<terminal parent task dirs>" cmd/cross_domain/run.sh
#
# Only the terminal domain runs with this release. Environment:
#   SEED_TASKS      parent Harbor task directories for the terminal domain
#   PROPOSERS       default "gpt-6-astra gpt-5.6-sol claude-opus-5"
#   N_TASKS         assignments per proposer (default 75)
#   AES_SOLVER_BASE_URL, AES_SOLVER_MODEL   solver for the pipeline's calibration step
source "$(dirname "$0")/../common.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$HERE/config.toml"

read -r -a PROPOSERS <<< "${PROPOSERS:-gpt-6-astra gpt-5.6-sol claude-opus-5}"
read -r -a SEEDS <<< "${SEED_TASKS:-<parent task dirs>}"

start_log cross_domain
require_env SEED_TASKS AES_SOLVER_BASE_URL AES_SOLVER_MODEL

# Terminal: the standard pipeline, up to two revisions after the initial attempt.
aes selfplay --config "$CONFIG" seed --pool-dir "$AES_RUN_DIR/parents" --source seed --no-copy \
    "${SEEDS[@]}"
for proposer in "${PROPOSERS[@]}"; do
  aes selfplay --config "$CONFIG" generate --parents-pool "$AES_RUN_DIR/parents" \
      --pool-dir "$AES_RUN_DIR/terminal/$proposer/pool" --out-dir "$AES_RUN_DIR/terminal/$proposer" \
      --harness "$(harness_for "$proposer")" --model "$proposer" --assignments "${N_TASKS:-75}" \
      --round-id terminal
done

for domain in swe tool browser desktop-cua kernelbench professional-workflow; do
  not_in_release "domain $domain" "domain assignments and its runtime (repositories, MCP services, local web apps, a graphical desktop, a T4 GPU, office documents)"
done
not_in_release "the appendix admission protocol" "the do-nothing-earns-zero and reset checks"

run "$PYTHON" "$AES_ROOT/analysis/make_all.py" --out-dir "$AES_RUN_DIR/outputs"
