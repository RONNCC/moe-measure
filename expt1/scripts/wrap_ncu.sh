#!/bin/bash
set -euo pipefail

PROFILE_RANK="${PROFILE_RANK:-0}"
RANK_ID="${RANK:-${SLURM_PROCID:-0}}"
NCU_OUT_BASE="${NCU_OUT_BASE:-ncu-report}"
NCU_SET="${NCU_SET:-full}"
NCU_EXTRA_ARGS="${NCU_EXTRA_ARGS:-}"

if [[ "$RANK_ID" == "$PROFILE_RANK" ]]; then
  echo "[ncu] profiling rank $RANK_ID -> ${NCU_OUT_BASE}-rank${RANK_ID}"
  # shellcheck disable=SC2086
  exec ncu --target-processes all --set "$NCU_SET" --force-overwrite -o "${NCU_OUT_BASE}-rank${RANK_ID}" $NCU_EXTRA_ARGS "$@"
else
  exec "$@"
fi
