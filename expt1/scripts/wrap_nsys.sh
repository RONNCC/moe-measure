#!/bin/bash
set -euo pipefail

PROFILE_RANK="${PROFILE_RANK:-0}"
RANK_ID="${RANK:-${SLURM_PROCID:-0}}"
NSYS_OUT_BASE="${NSYS_OUT_BASE:-nsys-report}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt}"
NSYS_EXTRA_ARGS="${NSYS_EXTRA_ARGS:-}"

if [[ "$RANK_ID" == "$PROFILE_RANK" ]]; then
  echo "[nsys] profiling rank $RANK_ID -> ${NSYS_OUT_BASE}-rank${RANK_ID}"
  # shellcheck disable=SC2086
  exec nsys profile --force-overwrite true --sample=none --trace "$NSYS_TRACE" -o "${NSYS_OUT_BASE}-rank${RANK_ID}" $NSYS_EXTRA_ARGS "$@"
else
  exec "$@"
fi
