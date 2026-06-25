#!/bin/bash
# Generic launcher — set CONFIG_PATH before calling, or pass as second arg.
# usage: bash scripts/_run_study.sh [dry-run|submit|aggregate] <config_path>
set -euo pipefail

MODE="${1:-help}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="${2:-${CONFIG_PATH:-}}"

if [[ -z "$CONFIG_PATH" ]]; then
  echo "usage: $0 [dry-run|submit|aggregate] <config.yaml>"
  exit 1
fi

export VLLM_SPEC="${VLLM_SPEC:-vllm}"

_setup() {
  cd "$ROOT_DIR"
  module purge
  module load gcc/12.3.0 python/3.11 cuda/12.1.1
  ENV_DIR="$HOME/scratch/moe-breakdown-login-venv"
  SKIP_VLLM=1
  export ENV_DIR SKIP_VLLM
  source scripts/setup_ice_env.sh
}

case "$MODE" in
  dry-run)
    _setup
    python3 scripts/submit_slurm_study.py --config "$CONFIG_PATH" --dry-run
    ;;
  submit)
    _setup
    python3 scripts/submit_slurm_study.py --config "$CONFIG_PATH"
    ;;
  aggregate)
    _setup
    STUDY_ROOT="$(python3 -c "
import yaml, os
cfg = yaml.safe_load(open('$CONFIG_PATH'))
root = os.path.expanduser(cfg['output_root'])
print(os.path.join(root, cfg['study_name']))
")"
    echo "[aggregate] study root: $STUDY_ROOT"
    python3 scripts/aggregate_results.py --study-root "$STUDY_ROOT"
    ;;
  help|*)
    echo "usage: $0 [dry-run|submit|aggregate] <config.yaml>"
    exit 0
    ;;
esac
