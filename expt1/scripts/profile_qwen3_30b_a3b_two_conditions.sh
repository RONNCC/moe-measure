#!/bin/bash
set -euo pipefail

# This script is meant for an interactive allocation or a Slurm batch job.
# It profiles two concrete Qwen3-30B-A3B MoE kernel conditions first,
# then leaves several more conditions commented out below for later use.
#
# Supported modes:
#   PROFILE_TOOL=nsys  (default)
#   PROFILE_TOOL=ncu
#   PROFILE_TOOL=none
#
# Local single-node usage (if you already have 4 visible GPUs):
#   PROFILE_TOOL=nsys bash scripts/profile_qwen3_30b_a3b_two_conditions.sh
#
# On ICE, more common is to run this inside an interactive/batch allocation.
# Active conditions below are pinned to fit on a single 2-GPU A100 node.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PROFILE_TOOL="${PROFILE_TOOL:-nsys}"
PROFILE_RANK="${PROFILE_RANK:-0}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/runs/qwen3-30b-a3b-profiles}"
mkdir -p "$RESULT_ROOT"

launch_condition() {
  local label="$1"
  local tp="$2"
  local ep="$3"
  local tokens="$4"
  local alpha="$5"

  local world_size=$((tp * ep))
  local out_dir="$RESULT_ROOT/$label"
  mkdir -p "$out_dir"

  local wrapper=""
  case "$PROFILE_TOOL" in
    ncu)
      export NCU_OUT_BASE="$out_dir/ncu"
      export NCU_SET="${NCU_SET:-full}"
      export NCU_EXTRA_ARGS="${NCU_EXTRA_ARGS:---nvtx --nvtx-include moe_kernel_timed}"
      wrapper="bash $ROOT_DIR/scripts/wrap_ncu.sh"
      ;;
    nsys)
      export NSYS_OUT_BASE="$out_dir/nsys"
      export NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt}"
      export NSYS_EXTRA_ARGS="${NSYS_EXTRA_ARGS:---capture-range=nvtx}"
      wrapper="bash $ROOT_DIR/scripts/wrap_nsys.sh"
      ;;
    none)
      wrapper=""
      ;;
    *)
      echo "Unsupported PROFILE_TOOL=$PROFILE_TOOL"
      exit 1
      ;;
  esac

  echo "[profile] label=$label tp=$tp ep=$ep tokens=$tokens alpha=$alpha tool=$PROFILE_TOOL"

  local cmd="python3 $ROOT_DIR/scripts/run_one_condition.py \
    --shape-name qwen3_30b_a3b_moe \
    --hidden-size 2048 \
    --intermediate-size 768 \
    --num-experts 128 \
    --topk 8 \
    --dtype bfloat16 \
    --activation silu \
    --all2all-backend deepep_low_latency \
    --tp-size $tp \
    --ep-size $ep \
    --num-tokens $tokens \
    --alpha $alpha \
    --warmup-iters 10 \
    --measure-iters 40 \
    --collect-buckets \
    --bucket-profile-iters 3 \
    --bucket-full-events \
    --output-root runs \
    --out-dir $out_dir \
    --json-out $out_dir/measurement.json"

  if [[ -n "$wrapper" ]]; then
    cmd="$wrapper $cmd"
  fi

  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun --ntasks="$world_size" --gpus-per-task=1 --cpus-per-task="${CPUS_PER_TASK:-8}" bash -lc "$cmd"
  else
    if [[ "$PROFILE_TOOL" == "none" ]]; then
      torchrun --nproc-per-node="$world_size" python3 "$ROOT_DIR/scripts/run_one_condition.py" \
        --shape-name qwen3_30b_a3b_moe \
        --hidden-size 2048 \
        --intermediate-size 768 \
        --num-experts 128 \
        --topk 8 \
        --dtype bfloat16 \
        --activation silu \
        --all2all-backend deepep_low_latency \
        --tp-size "$tp" \
        --ep-size "$ep" \
        --num-tokens "$tokens" \
        --alpha "$alpha" \
        --warmup-iters 10 \
        --measure-iters 40 \
        --collect-buckets \
        --bucket-profile-iters 3 \
        --bucket-full-events \
        --output-root runs \
        --out-dir "$out_dir" \
        --json-out "$out_dir/measurement.json"
    else
      torchrun --nproc-per-node="$world_size" bash "$ROOT_DIR/scripts/wrap_${PROFILE_TOOL}.sh" python3 scripts/run_one_condition.py \
        --shape-name qwen3_30b_a3b_moe \
        --hidden-size 2048 \
        --intermediate-size 768 \
        --num-experts 128 \
        --topk 8 \
        --dtype bfloat16 \
        --activation silu \
        --all2all-backend deepep_low_latency \
        --tp-size "$tp" \
        --ep-size "$ep" \
        --num-tokens "$tokens" \
        --alpha "$alpha" \
        --warmup-iters 10 \
        --measure-iters 40 \
        --collect-buckets \
        --bucket-profile-iters 3 \
        --bucket-full-events \
        --output-root runs \
        --out-dir "$out_dir" \
        --json-out "$out_dir/measurement.json"
    fi
  fi
}

# ---------------------------------------------------------------------------
# Active initial conditions: run these first to validate code path.
# ---------------------------------------------------------------------------

# Condition 1: baseline, no inter-GPU expert traffic expected.
launch_condition "01-ep1-balanced-tok512-alpha1" 1 1 512 1.0

# Condition 2: EP-enabled communication case, should exercise cross-rank MoE dispatch.
launch_condition "02-ep2-imbalanced-tok1024-alpha4" 1 2 1024 4.0

# ---------------------------------------------------------------------------
# Additional conditions for later study expansion.
# Uncomment once the first two conditions work.
# ---------------------------------------------------------------------------

# launch_condition "03-ep4-balanced-tok1024-alpha1" 1 4 1024 1.0
# launch_condition "04-ep4-heavy-imbalance-tok1024-alpha8" 1 4 1024 8.0
# launch_condition "05-ep4-balanced-tok2048-alpha1" 1 4 2048 1.0
# launch_condition "06-ep4-imbalanced-tok2048-alpha4" 1 4 2048 4.0
# launch_condition "07-ep8-balanced-tok1024-alpha1" 1 8 1024 1.0
# launch_condition "08-ep8-imbalanced-tok1024-alpha4" 1 8 1024 4.0
# launch_condition "09-tp2ep4-balanced-tok1024-alpha1" 2 4 1024 1.0
# launch_condition "10-tp2ep4-imbalanced-tok1024-alpha4" 2 4 1024 4.0
