#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

set -euo pipefail

if [[ -z "${STUDY_CONFIG:-}" ]]; then
  echo "STUDY_CONFIG is required"
  exit 1
fi
if [[ -z "${TP_SIZE:-}" || -z "${EP_SIZE:-}" ]]; then
  echo "TP_SIZE and EP_SIZE are required"
  exit 1
fi

WORLD_SIZE=$((TP_SIZE * EP_SIZE))
WORKDIR="${WORKDIR:-$PWD}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
VENV="${VENV:-}"
MODULES="${MODULES:-}"
OUT_DIR="${OUT_DIR:-}"
if [[ -z "$OUT_DIR" && -n "${OUT_DIR_BASE:-}" ]]; then
  OUT_DIR="${OUT_DIR_BASE}/${SLURM_JOB_ID}_tp${TP_SIZE}-ep${EP_SIZE}"
  mkdir -p "$OUT_DIR"
fi
UV_ENV_DIR="${UV_ENV_DIR:-${TMPDIR:-}/moe-breakdown-deepep-venv}"

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
  bash "$ROOT_DIR/expt-3-deeptest/scripts/bootstrap_uv_env.sh" "$UV_ENV_DIR"
  VENV="$UV_ENV_DIR"
fi

if [[ -n "$VENV" ]]; then
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
fi

# Point deep_ep.so to the NCCL 2.30+ GIN library at runtime.
if python -c "import nvidia.nccl" 2>/dev/null; then
  NCCL_LIB_DIR=$(python -c "import nvidia.nccl, os; base = getattr(nvidia.nccl, '__file__', None) or list(nvidia.nccl.__path__)[0]; print(os.path.join(os.path.dirname(base), 'lib'))")
  export LD_LIBRARY_PATH="${NCCL_LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  echo "[slurm] LD_LIBRARY_PATH set for NCCL: $NCCL_LIB_DIR"
fi

# NVSHMEM: ibrc and ibdevx transports crash on PACE ICE (exit 139). Stubs at
# /home/hice1/sghose7/nvshmem-stubs/ intercept dlopen with nvshmemt_init()→1 to
# skip them. ibgda (IBGDA/GPU Direct RDMA) is NOT stubbed — gdrdrv and
# nvidia_peermem are loaded on compute nodes, so ibgda can load for LL multi-node.
# NVSHMEM_QP_DEPTH: low_latency_dispatch asserts qp_depth >= (tokens_per_rank+1)*2;
# 32768 covers our 8192-token worst-case with margin.
# NVSHMEM_BOOTSTRAP=uid avoids PMI/MPI dependency in SLURM srun launches.
NVSHMEM_STUBS_DIR="${NVSHMEM_STUBS_DIR:-/home/hice1/sghose7/nvshmem-stubs}"
export LD_LIBRARY_PATH="${NVSHMEM_STUBS_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NVSHMEM_BOOTSTRAP="${NVSHMEM_BOOTSTRAP:-uid}"
export NVSHMEM_QP_DEPTH="${NVSHMEM_QP_DEPTH:-32768}"
export NVSHMEM_DEBUG="${NVSHMEM_DEBUG:-WARN}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPUS_PER_TASK}"
export MASTER_PORT="${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 10000))}"

if [[ -z "${MASTER_ADDR:-}" && -n "${SLURM_NODELIST:-}" ]]; then
  export MASTER_ADDR
  MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
fi

echo "[slurm] WORKDIR=$WORKDIR"
echo "[slurm] STUDY_CONFIG=$STUDY_CONFIG"
echo "[slurm] TP_SIZE=$TP_SIZE EP_SIZE=$EP_SIZE WORLD_SIZE=$WORLD_SIZE"
echo "[slurm] OUT_DIR=$OUT_DIR"
echo "[slurm] TRANSPORT_CONDITION=${TRANSPORT_CONDITION:-nvlink_default}"
echo "[slurm] NVSHMEM stubs dir: ${NVSHMEM_STUBS_DIR} (stubs ibrc/ibdevx only; ibgda is real)"

ARGS=(
  --config "$STUDY_CONFIG"
  --tp-size "$TP_SIZE"
  --ep-size "$EP_SIZE"
)
if [[ -n "$OUT_DIR" ]]; then
  ARGS+=(--out-dir "$OUT_DIR")
fi

# All GPUs from the job allocation remain visible to all tasks (no --gpus-per-task).
# Each rank selects its device via torch.cuda.set_device(local_rank) in distributed.py.
srun \
  --ntasks="$WORLD_SIZE" \
  --cpus-per-task="$CPUS_PER_TASK" \
  python3 "$ROOT_DIR/expt-3-deeptest/scripts/run_direct_moe_sweep.py" "${ARGS[@]}"
