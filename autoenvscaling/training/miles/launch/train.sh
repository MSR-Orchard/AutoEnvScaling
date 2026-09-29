#!/usr/bin/env bash
# Start one Miles training job on this node. Run it inside the Miles container
# (github.com/radixark/miles, patched as described in third_party/miles/README.md).
#
#   train.sh <arguments for Miles' train.py>
#
# The Python launcher (python -m autoenvscaling.training.miles) builds the arguments from a
# MilesConfig and sets the variables below. Required:
#   MILES_DIR          Miles checkout (with the patch applied)
#   MEGATRON_DIR       Megatron-LM checkout that Miles was built against
# Optional:
#   MODEL_ARGS_SCRIPT  model architecture script relative to MILES_DIR, e.g.
#                      scripts/models/qwen3.6-35B-A3B.sh; it must define MODEL_ARGS
#   AES_ROOT           this repository, added to PYTHONPATH so Ray workers import autoenvscaling
#   AES_BRIDGE_DIR     bridge control directory (bridge mode)
#   AES_FORWARD_ENV    space-separated names of extra variables to pass to Ray workers
#   NUM_GPUS (8), MASTER_ADDR (127.0.0.1), RAY_DASHBOARD_PORT (8265), TRAIN_ENTRY (train.py)
#
# Exits non-zero unless the Ray job ends in SUCCEEDED.
set -euo pipefail
set +x   # never trace: the runtime environment may carry credentials

: "${MILES_DIR:?set MILES_DIR to the Miles checkout}"
: "${MEGATRON_DIR:?set MEGATRON_DIR to the Megatron-LM checkout}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
TRAIN_ENTRY="${TRAIN_ENTRY:-train.py}"
export PYTHONUNBUFFERED=1

MODEL_ARGS=()
if [ -n "${MODEL_ARGS_SCRIPT:-}" ]; then
  # shellcheck disable=SC1090
  source "${MILES_DIR}/${MODEL_ARGS_SCRIPT}"
fi

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [ "${NVLINK_COUNT:-0}" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi

export no_proxy="127.0.0.1,localhost,${MASTER_ADDR}"
export NO_PROXY="${no_proxy}"

# Start Ray from MILES_DIR so worker imports do not pick up a stray package from the caller's cwd.
cd "${MILES_DIR}"
ray stop --force >/dev/null 2>&1 || true
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats --dashboard-host=127.0.0.1 --dashboard-port="${RAY_DASHBOARD_PORT}"

RUNTIME_ENV_FILE="$(mktemp "${TMPDIR:-/tmp}/aes-miles-runtime-env.XXXXXX.json")"
chmod 600 "${RUNTIME_ENV_FILE}"
trap 'rm -f -- "${RUNTIME_ENV_FILE}"' EXIT
export MEGATRON_DIR MILES_DIR HAS_NVLINK
export AES_ROOT="${AES_ROOT:-}" AES_BRIDGE_DIR="${AES_BRIDGE_DIR:-}" AES_FORWARD_ENV="${AES_FORWARD_ENV:-}"
python3 - "${RUNTIME_ENV_FILE}" <<'PY'
import json
import os
import sys

paths = [os.environ["MEGATRON_DIR"], os.environ["MILES_DIR"]]
if os.environ.get("AES_ROOT"):
    paths.append(os.environ["AES_ROOT"])
env = {
    "PYTHONPATH": ":".join(paths),
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": os.environ.get("HAS_NVLINK", "0"),
    "no_proxy": os.environ["no_proxy"],
    "NO_PROXY": os.environ["NO_PROXY"],
}
names = ["AES_BRIDGE_DIR", "AES_BRIDGE_POLL_S", "WANDB_API_KEY"] + os.environ.get("AES_FORWARD_ENV", "").split()
for name in names:
    if os.environ.get(name):
        env[name] = os.environ[name]
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump({"env_vars": env}, fh)
PY

JOB_ID="aes_miles_$(date +%Y%m%d_%H%M%S)_$$"
ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --submission-id="${JOB_ID}" --runtime-env="${RUNTIME_ENV_FILE}" --no-wait \
  -- python3 -u "${MILES_DIR}/${TRAIN_ENTRY}" "${MODEL_ARGS[@]}" "$@"

ray job logs --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" --follow "${JOB_ID}" || true

# `ray job logs --follow` exits zero even when the job failed, so ask Ray for the final status.
STATUS="$(python3 - "http://127.0.0.1:${RAY_DASHBOARD_PORT}" "${JOB_ID}" <<'PY'
import sys
from ray.job_submission import JobSubmissionClient
print(JobSubmissionClient(sys.argv[1]).get_job_status(sys.argv[2]).value)
PY
)"
echo "Ray job ${JOB_ID}: ${STATUS}"
ray stop --force >/dev/null 2>&1 || true
[ "${STATUS}" = "SUCCEEDED" ]
