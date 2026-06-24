#!/bin/bash
set -euo pipefail

# Pinned ICE defaults. Override via env vars only if you intentionally
# want to change them.
GCC_MODULE="${GCC_MODULE:-gcc/12.3.0}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.1.1}"
UV_SPEC="${UV_SPEC:-uv}"
VLLM_SPEC="${VLLM_SPEC:-vllm}"
SKIP_VLLM="${SKIP_VLLM:-0}"
WORKDIR="${WORKDIR:-$HOME/scratch/moe-breakdown}"
ENV_DIR="${ENV_DIR:-${TMPDIR:-$HOME/scratch}/moe-breakdown-venv}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${TMPDIR:-$HOME/scratch/.cache}/uv-cache}"
HF_HOME="${HF_HOME:-${TMPDIR:-$HOME/scratch}/hf-cache}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR:-$HOME/scratch}/triton-cache}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${TMPDIR:-$HOME/scratch}/torchinductor-cache}"

mkdir -p "$WORKDIR" "$UV_CACHE_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
cd "$WORKDIR"

module purge || true
module load "$GCC_MODULE" "$PYTHON_MODULE" "$CUDA_MODULE"

if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user "$UV_SPEC"
fi
export PATH="$HOME/.local/bin:$PATH"

export UV_CACHE_DIR HF_HOME HF_HUB_CACHE TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR VLLM_SPEC SKIP_VLLM

bash scripts/bootstrap_uv_env.sh "$ENV_DIR"
# shellcheck disable=SC1090
source "$ENV_DIR/bin/activate"

if [[ "${SKIP_VLLM}" == "1" ]]; then
  echo "[ice-setup] CLI-only env ready"
else
  echo "[ice-setup] full runtime env ready"
fi

echo "[ice-setup] WORKDIR=$WORKDIR"
echo "[ice-setup] ENV_DIR=$ENV_DIR"
echo "[ice-setup] UV_CACHE_DIR=$UV_CACHE_DIR"
echo "[ice-setup] HF_HOME=$HF_HOME"
echo "[ice-setup] CUDA_MODULE=$CUDA_MODULE"
if [[ "${SKIP_VLLM}" != "1" ]]; then
  echo "[ice-setup] VLLM_SPEC=$VLLM_SPEC"
fi
python3 - <<'PY'
import sys
print('[ice-setup] python=', sys.version)
if True:
    try:
        import yaml
        print('[ice-setup] pyyaml=', getattr(yaml, '__version__', 'unknown'))
    except Exception as e:
        print('[ice-setup] yaml import failed:', e)
PY
if [[ "${SKIP_VLLM}" != "1" ]]; then
python3 - <<'PY'
try:
    import torch
    print('[ice-setup] torch=', torch.__version__)
except Exception as e:
    print('[ice-setup] torch import failed:', e)
try:
    import vllm
    print('[ice-setup] vllm=', getattr(vllm, '__version__', 'unknown'))
except Exception as e:
    print('[ice-setup] vllm import failed:', e)
PY
fi
