#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/study.qwen3_30b_a3b.initial.yaml}"
MODE="${1:-submit}"

cd "$ROOT_DIR"

case "$MODE" in
  dry-run)
    python3 scripts/submit_slurm_study.py --config "$CONFIG_PATH" --dry-run
    ;;
  submit)
    python3 scripts/submit_slurm_study.py --config "$CONFIG_PATH"
    ;;
  *)
    echo "usage: $0 [dry-run|submit]"
    exit 1
    ;;
esac
