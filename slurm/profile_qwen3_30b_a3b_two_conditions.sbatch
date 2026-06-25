#!/bin/bash
#SBATCH --job-name=qwen3-a3b-profile
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=qwen3-a3b-profile-%j.out
#SBATCH --error=qwen3-a3b-profile-%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/scratch/moe-breakdown}"
VENV="${VENV:-}"
UV_ENV_DIR="${UV_ENV_DIR:-${TMPDIR:-}/moe-breakdown-venv}"
MODULES="${MODULES:-gcc/12.3.0 python/3.11 cuda/12.1.1}"
PROFILE_TOOL="${PROFILE_TOOL:-nsys}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"

ROOT_DIR="$WORKDIR"
cd "$ROOT_DIR"

module purge || true
if [[ -n "$MODULES" ]]; then
  # shellcheck disable=SC2086
  module load $MODULES
fi

if [[ -n "${TMPDIR:-}" ]]; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$TMPDIR/uv-cache}"
  export HF_HOME="${HF_HOME:-$TMPDIR/hf-cache}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$TMPDIR/hf-cache}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$TMPDIR/triton-cache}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$TMPDIR/torchinductor-cache}"
  mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
else
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/scratch/.cache/uv}"
  mkdir -p "$UV_CACHE_DIR"
fi

if [[ -n "$UV_ENV_DIR" ]]; then
  bash "$ROOT_DIR/scripts/bootstrap_uv_env.sh" "$UV_ENV_DIR"
  VENV="$UV_ENV_DIR"
fi

if [[ -n "$VENV" ]]; then
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPUS_PER_TASK}"
export PROFILE_TOOL

# Active script uses up to 2 GPUs on one A100 node.
# If you later uncomment multi-node / larger EP examples, adjust this file accordingly.

bash "$ROOT_DIR/scripts/profile_qwen3_30b_a3b_two_conditions.sh"
