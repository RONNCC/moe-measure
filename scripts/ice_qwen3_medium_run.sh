#!/bin/bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$ROOT_DIR/scripts/_run_study.sh" "${1:-help}" \
    "$ROOT_DIR/configs/study.qwen3_30b_a3b.medium.yaml"

case "$MODE" in
  dry-run)
    cd "$ROOT_DIR"
    module purge
    module load gcc/12.3.0 python/3.11 cuda/12.1.1
    ENV_DIR="$HOME/scratch/moe-breakdown-login-venv"
    SKIP_VLLM=1
    export ENV_DIR SKIP_VLLM
    source scripts/setup_ice_env.sh
    python3 scripts/submit_slurm_study.py --config "$CONFIG_PATH" --dry-run
    ;;
  submit)
    cd "$ROOT_DIR"
    module purge
    module load gcc/12.3.0 python/3.11 cuda/12.1.1
    ENV_DIR="$HOME/scratch/moe-breakdown-login-venv"
    SKIP_VLLM=1
    export ENV_DIR SKIP_VLLM
    source scripts/setup_ice_env.sh
    python3 scripts/submit_slurm_study.py --config "$CONFIG_PATH"
    ;;
  aggregate)
    cd "$ROOT_DIR"
    module purge
    module load gcc/12.3.0 python/3.11 cuda/12.1.1
    ENV_DIR="$HOME/scratch/moe-breakdown-login-venv"
    SKIP_VLLM=1
    export ENV_DIR SKIP_VLLM
    source scripts/setup_ice_env.sh
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
    echo "usage: $0 [dry-run|submit|aggregate]"
    exit 0
    ;;
esac
