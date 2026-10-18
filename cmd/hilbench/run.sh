#!/usr/bin/env bash
# HiL-Bench (Section 4.5): generate HiL-style training tasks, train Qwen3.6-35B-A3B to ask for
# missing requirements, and score it. Produces Tables 4 and 5.
#
#   HIL_BENCH_ROOT=<HiL-Bench checkout> cmd/hilbench/run.sh
#
# Environment:
#   HIL_BENCH_ROOT   HiL-Bench checkout at the pinned commit (for the ask-human server)
#   HIL_TASKS        authored HiL task directories (each with ask-human-data/blocker_registry.json)
#   HIL_SFT_ROWS     cold-start rows (messages / tools JSON lines) from asking trajectories
#   BASE_HF, BASE_REF   Qwen3.6-35B-A3B weights (Hugging Face and torch_dist)
#   EPISODES         episodes.jsonl files to score
#   MILES_DIR, MEGATRON_DIR   checked by `autoenvscaling train plan`
source "$(dirname "$0")/../common.sh"

start_log hilbench
require_env HIL_BENCH_ROOT

# 0. The ask-human server is HiL-Bench's own, checked against the pinned hash.
run "$PYTHON" -c 'import sys; from pathlib import Path
from autoenvscaling.evals.hilbench.ask_human import upstream_server
print(upstream_server(Path(sys.argv[1])))' "${HIL_BENCH_ROOT:-<HiL-Bench checkout>}"

# 1. Authoring: the HiL prompt for episode i (domain hint i) ...
aes hilbench prompt --index 0
# ... run by Claude Opus 5 in Claude Code through the proposer pipeline.
not_in_release "HiL task generation" "a HiL setting for the proposer pipeline (aes generate uses the standard authoring prompt)"

# 2. Derive the full_info, ask_human and baseline variants (string operations, no model call).
if [[ -n "${HIL_TASKS:-}" || "$DRY_RUN" == "1" ]]; then
  read -r -a TASKS <<< "${HIL_TASKS:-<authored HiL task dirs>}"
  aes hilbench derive --out-root "$AES_RUN_DIR/variants" "${TASKS[@]}"
fi

# 3. Cold start: fine-tune on 75 Claude Opus 5 trajectories that ask about missing requirements.
not_in_release "cold-start rows" "exporting asking trajectories from HiL episodes as SFT rows"
if [[ -n "${HIL_SFT_ROWS:-}" || "$DRY_RUN" == "1" ]]; then
  # The cold-start preset is the Qwen3.6-35B-A3B fine-tuning configuration; its epochs and
  # learning rate are placeholders.
  TRAIN=(--preset cold-start --prompt-data "${HIL_SFT_ROWS:-<sft rows>}"
         --hf-checkpoint "${BASE_HF:-<base hf>}" --ref-load "${BASE_REF:-<base torch_dist>}"
         --save "$AES_RUN_DIR/cold-start" --log "$AES_RUN_DIR/miles.log")
  aes train plan "${TRAIN[@]}"
  aes train run "${TRAIN[@]}"
fi

# 4. RL for up to 150 steps with the shaped clarification reward, on the 100 official training
#    tasks or on the generated ones.
not_in_release "HiL RL" "a Harbor runner for HiL-Bench episodes with the ask-human sidecar, and the shaped reward in the training loop"

# 5. Score held-out ask_human episodes: task success, Ask-F1, ask rate.
if [[ -n "${EPISODES:-}" || "$DRY_RUN" == "1" ]]; then
  read -r -a EPISODE_FILES <<< "${EPISODES:-<episodes.jsonl>}"
  aes hilbench report "${EPISODE_FILES[@]}"
fi
run "$PYTHON" "$AES_ROOT/analysis/make_all.py" --out-dir "$AES_RUN_DIR/outputs"
