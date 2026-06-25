#!/bin/bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$ROOT_DIR/scripts/_run_study.sh" "${1:-help}" \
  "$ROOT_DIR/configs/study.qwen3_30b_a3b.full-twonode-limited.yaml"
