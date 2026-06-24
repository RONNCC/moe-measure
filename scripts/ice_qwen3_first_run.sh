#!/bin/bash
set -euo pipefail

MODE="${1:-help}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/study.qwen3_30b_a3b.local.yaml}"
DEFAULT_CONFIG="$ROOT_DIR/configs/study.qwen3_30b_a3b.initial.yaml"
export VLLM_SPEC="${VLLM_SPEC:-vllm}"

ensure_local_config() {
  if [[ ! -f "$CONFIG_PATH" ]]; then
    cp "$DEFAULT_CONFIG" "$CONFIG_PATH"
  fi
}

case "$MODE" in
  dry-run)
    cd "$ROOT_DIR"
    module purge
    module load gcc/12.3.0 python/3.11 cuda/12.1.1
    ensure_local_config
    ENV_DIR="$HOME/scratch/moe-breakdown-login-venv"
    SKIP_VLLM=1
    export ENV_DIR SKIP_VLLM
    source scripts/setup_ice_env.sh
    CONFIG_PATH="$CONFIG_PATH" bash scripts/submit_qwen3_30b_a3b_initial.sh dry-run
    ;;
  submit)
    cd "$ROOT_DIR"
    module purge
    module load gcc/12.3.0 python/3.11 cuda/12.1.1
    ensure_local_config
    ENV_DIR="$HOME/scratch/moe-breakdown-login-venv"
    SKIP_VLLM=1
    export ENV_DIR SKIP_VLLM
    source scripts/setup_ice_env.sh
    CONFIG_PATH="$CONFIG_PATH" bash scripts/submit_qwen3_30b_a3b_initial.sh submit
    ;;
  setup)
    cd "$ROOT_DIR"
    source scripts/setup_ice_env.sh
    ;;
  profile-none)
    cd "$ROOT_DIR"
    source scripts/setup_ice_env.sh
    export PROFILE_TOOL=none
    bash scripts/profile_qwen3_30b_a3b_two_conditions.sh
    ;;
  profile-nsys)
    cd "$ROOT_DIR"
    source scripts/setup_ice_env.sh
    export PROFILE_TOOL=nsys
    bash scripts/profile_qwen3_30b_a3b_two_conditions.sh
    ;;
  profile-ncu)
    cd "$ROOT_DIR"
    source scripts/setup_ice_env.sh
    export PROFILE_TOOL=ncu
    bash scripts/profile_qwen3_30b_a3b_two_conditions.sh
    ;;
  help|*)
    cat <<'EOF'
Usage:
  bash scripts/ice_qwen3_first_run.sh dry-run
  bash scripts/ice_qwen3_first_run.sh submit
  bash scripts/ice_qwen3_first_run.sh setup
  bash scripts/ice_qwen3_first_run.sh profile-none
  bash scripts/ice_qwen3_first_run.sh profile-nsys
  bash scripts/ice_qwen3_first_run.sh profile-ncu

Notes:
  - dry-run / submit are intended for the login node.
  - setup / profile-* are intended to run inside a GPU allocation.
  - local config path defaults to configs/study.qwen3_30b_a3b.local.yaml
EOF
    ;;
esac
