#!/bin/bash
#SBATCH --job-name=build-deepep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --qos=coc-ice
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

DEEPEP_BUILD_DIR="${DEEPEP_BUILD_DIR:-$HOME/scratch/deepep-build}"
BUILD_VENV="${BUILD_VENV:-/storage/ice1/0/2/sghose7/moe-breakdown-venv}"
NVSHMEM_DIR="${NVSHMEM_DIR:-$DEEPEP_BUILD_DIR/workspace/nvshmem}"
DEEPEP_SRC="${DEEPEP_SRC:-$DEEPEP_BUILD_DIR/workspace/DeepEP}"
WHEEL_OUT="${WHEEL_OUT:-$DEEPEP_BUILD_DIR/workspace/dist}"

echo "[build-deepep] DEEPEP_SRC=$DEEPEP_SRC"
echo "[build-deepep] WHEEL_OUT=$WHEEL_OUT"

module purge || true
module load gcc/12.3.0 python/3.11 cuda/13.0.1

CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-}}"
if [[ -z "$CUDA_HOME" ]]; then
  NVCC_BIN=$(which nvcc 2>/dev/null || true)
  [[ -n "$NVCC_BIN" ]] && CUDA_HOME=$(dirname "$(dirname "$NVCC_BIN")")
fi
if [[ -z "$CUDA_HOME" ]]; then
  echo "ERROR: cannot determine CUDA_HOME" >&2; exit 1
fi

echo "[build-deepep] CUDA_HOME=$CUDA_HOME"
echo "[build-deepep] nvcc: $(nvcc --version | grep release)"

source "$BUILD_VENV/bin/activate"

echo "[build-deepep] upgrading nvidia-nccl-cu13 to >=2.30.4"
uv pip install "nvidia-nccl-cu13>=2.30.4" --no-deps

NCCL_ROOT=$(python -c "import nvidia.nccl; print(nvidia.nccl.__path__[0])")
echo "[build-deepep] NCCL root: $NCCL_ROOT"

export LIBRARY_PATH="${CUDA_HOME}/lib64/stubs:${NCCL_ROOT}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${NCCL_ROOT}/lib:${LD_LIBRARY_PATH:-}"

cd "$DEEPEP_SRC"
rm -rf build dist *.egg-info

export NVSHMEM_DIR
mkdir -p "$WHEEL_OUT"

echo "[build-deepep] building wheel"
python setup.py bdist_wheel --dist-dir "$WHEEL_OUT"

echo "[build-deepep] done:"
ls -lh "$WHEEL_OUT"/*.whl
