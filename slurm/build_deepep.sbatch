#!/bin/bash
#SBATCH --job-name=build-deepep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --qos=coc-ice
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Build a DeepEP wheel against NCCL 2.30.7 (required for GIN APIs).
# No GPU needed — nccl.cu is host code only; GPU kernels are JIT compiled at runtime.
# Run from the repo root:
#   sbatch slurm/build_deepep.sbatch

set -euo pipefail

DEEPEP_BUILD_DIR="${DEEPEP_BUILD_DIR:-$HOME/scratch/deepep-build}"
BUILD_VENV="${BUILD_VENV:-/storage/ice1/0/2/sghose7/moe-breakdown-venv}"
NVSHMEM_DIR="${NVSHMEM_DIR:-$DEEPEP_BUILD_DIR/workspace/nvshmem}"
DEEPEP_SRC="${DEEPEP_SRC:-$DEEPEP_BUILD_DIR/workspace/DeepEP}"
WHEEL_OUT="${WHEEL_OUT:-$DEEPEP_BUILD_DIR/workspace/dist}"

echo "[build-deepep] DEEPEP_BUILD_DIR=$DEEPEP_BUILD_DIR"
echo "[build-deepep] BUILD_VENV=$BUILD_VENV"
echo "[build-deepep] NVSHMEM_DIR=$NVSHMEM_DIR"

module purge || true
module load gcc/12.3.0 python/3.11 cuda/13.0.1

echo "[build-deepep] nvcc: $(nvcc --version | grep release)"
echo "[build-deepep] python: $(which python3)"

source "$BUILD_VENV/bin/activate"

# Upgrade NCCL — 2.28.9 ships without GIN APIs (ncclGinRequest_t etc.), 2.30.4+ has them.
echo "[build-deepep] upgrading nvidia-nccl-cu13 to >=2.30.4"
uv pip install "nvidia-nccl-cu13>=2.30.4" --no-deps

python -c "import torch; print('[build-deepep] PyTorch CUDA:', torch.version.cuda, '| NCCL:', torch.cuda.nccl.version())"

# Clean stale build artifacts from the previous (failed) attempt.
cd "$DEEPEP_SRC"
rm -rf build dist *.egg-info

export NVSHMEM_DIR
mkdir -p "$WHEEL_OUT"

echo "[build-deepep] building wheel in $WHEEL_OUT"
python setup.py bdist_wheel --dist-dir "$WHEEL_OUT"

echo "[build-deepep] done:"
ls -lh "$WHEEL_OUT"/*.whl
