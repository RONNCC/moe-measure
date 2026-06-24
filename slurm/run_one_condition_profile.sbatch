#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

set -euo pipefail

: "${WORKDIR:?WORKDIR is required}"
: "${TP_SIZE:?TP_SIZE is required}"
: "${EP_SIZE:?EP_SIZE is required}"
: "${CONDITION_ARGS:?CONDITION_ARGS is required}"

WORLD_SIZE=$((TP_SIZE * EP_SIZE))
PROFILE_TOOL="${PROFILE_TOOL:-none}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
VENV="${VENV:-}"
MODULES="${MODULES:-}"
UV_ENV_DIR="${UV_ENV_DIR:-}"

cd "$WORKDIR"

module purge || true
if [[ -n "$MODULES" ]]; then
  # shellcheck disable=SC2086
  module load $MODULES
fi

if [[ -n "$UV_ENV_DIR" ]]; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/scratch/.cache/uv}"
  mkdir -p "$UV_CACHE_DIR"
  bash scripts/bootstrap_uv_env.sh "$UV_ENV_DIR"
  VENV="$UV_ENV_DIR"
fi

if [[ -n "$VENV" ]]; then
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPUS_PER_TASK}"
export MASTER_PORT="${MASTER_PORT:-29500}"

if [[ -z "${MASTER_ADDR:-}" && -n "${SLURM_NODELIST:-}" ]]; then
  export MASTER_ADDR
  MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
fi

case "$PROFILE_TOOL" in
  ncu)
    WRAPPER="bash scripts/wrap_ncu.sh"
    ;;
  nsys)
    WRAPPER="bash scripts/wrap_nsys.sh"
    ;;
  none)
    WRAPPER=""
    ;;
  *)
    echo "Unsupported PROFILE_TOOL=$PROFILE_TOOL"
    exit 1
    ;;
esac

COMMAND="python3 scripts/run_one_condition.py $CONDITION_ARGS"
if [[ -n "$WRAPPER" ]]; then
  COMMAND="$WRAPPER $COMMAND"
fi

echo "[profile] PROFILE_TOOL=$PROFILE_TOOL"
echo "[profile] COMMAND=$COMMAND"

srun \
  --ntasks="$WORLD_SIZE" \
  --gpus-per-task=1 \
  --cpus-per-task="$CPUS_PER_TASK" \
  bash -lc "$COMMAND"
