#!/bin/bash
set -euo pipefail

ENV_DIR="${1:-}"
if [[ -z "$ENV_DIR" ]]; then
  echo "usage: $0 <env-dir>"
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found in PATH"
  exit 1
fi

mkdir -p "$(dirname "$ENV_DIR")"
if [[ ! -d "$ENV_DIR" ]]; then
  uv venv "$ENV_DIR" --python python3.11
fi

SKIP_VLLM="${SKIP_VLLM:-0}"
READY_MARKER="$ENV_DIR/.fused_moe_kernel_study_ready"
if [[ "$SKIP_VLLM" == "1" ]]; then
  READY_MARKER="$ENV_DIR/.fused_moe_kernel_study_ready_cli"
fi
# shellcheck disable=SC1090
source "$ENV_DIR/bin/activate"

if [[ "${FORCE_UV_REINSTALL:-0}" != "1" && -f "$READY_MARKER" ]]; then
  echo "[uv] reusing existing environment at $ENV_DIR"
  exit 0
fi

uv pip install --upgrade pip setuptools wheel
uv pip install -e .
uv pip install pyyaml
if [[ "$SKIP_VLLM" != "1" ]]; then
  VLLM_SPEC="${VLLM_SPEC:-vllm}"
  uv pip install "$VLLM_SPEC"
fi

# DeepEP wheel: install NCCL 2.30.4+ (required for GIN APIs) then the wheel.
# The wheel is a pre-built .so; no nvcc needed here.
# Set LD_LIBRARY_PATH in your sbatch after activation so deep_ep.so finds libnccl.so.2.
DEEPEP_WHEEL="${DEEPEP_WHEEL:-}"
if [[ -n "$DEEPEP_WHEEL" ]]; then
  # Expand glob in case the config uses a wildcard (e.g. deep_ep-*.whl).
  DEEPEP_WHEEL_PATH=$(ls $DEEPEP_WHEEL 2>/dev/null | head -1 || true)
  if [[ -z "$DEEPEP_WHEEL_PATH" ]]; then
    echo "[uv] ERROR: DEEPEP_WHEEL set to '$DEEPEP_WHEEL' but no matching file found" >&2
    exit 1
  fi
  echo "[uv] installing nvidia-nccl-cu13>=2.30.4 for DeepEP GIN backend"
  uv pip install "nvidia-nccl-cu13>=2.30.4" --no-deps
  echo "[uv] installing DeepEP wheel: $DEEPEP_WHEEL_PATH"
  uv pip install --no-deps "$DEEPEP_WHEEL_PATH"
fi

touch "$READY_MARKER"
