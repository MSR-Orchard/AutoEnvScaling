#!/usr/bin/env bash
# Run train.sh inside a Miles container image with Apptainer, for hosts without Docker.
#
#   apptainer.sh <arguments for Miles' train.py>
#
# Required:
#   MILES_IMAGE     an Apptainer image or sandbox directory built from the Miles Docker image
# Optional:
#   AES_BINDS       comma-separated host:container bind mounts (model, data and output
#                   directories, the bridge control directory, this repository)
# Every variable train.sh reads is passed through unchanged (see train.sh).
set -euo pipefail
: "${MILES_IMAGE:?set MILES_IMAGE to the Miles Apptainer image}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

BIND_ARGS=()
IFS=',' read -r -a BINDS <<< "${AES_BINDS:-}"
for bind in "${BINDS[@]}"; do
  [ -n "${bind}" ] && BIND_ARGS+=(--bind "${bind}")
done

exec apptainer exec --nv --writable-tmpfs "${BIND_ARGS[@]}" --bind "${HERE}:${HERE}" \
  "${MILES_IMAGE}" bash "${HERE}/train.sh" "$@"
