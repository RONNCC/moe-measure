#!/bin/bash
# PACE-ICE-tuned Slurm script for moe-breakdown.
#
# Usage:
#   # First-time setup (do once after logging in):
#   mkdir -p ~/scratch/.cache ~/scratch/.conda
#   ln -sfn ~/scratch/.cache ~/.cache
#   ln -sfn ~/scratch/.conda ~/.conda
#   cd ~/scratch && git clone <repo-url> moe-breakdown
#
#   # Run a job:
#   cd ~/scratch/moe-breakdown
#   MODEL="Qwen/Qwen1.5-MoE-A2.7B" sbatch examples/run_slurm_pace_ice.sbatch
#
#   # Or with overrides:
#   sbatch --export=ALL,MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1",GPU_TYPE="a100" \
#          examples/run_slurm_pace_ice.sbatch

#SBATCH --job-name=moe-breakdown
#SBATCH --qos=coc-ice                  # 'pace-ice' for PACE-ICE; 'coc-grade' if you're a TA
#SBATCH --gres=gpu:rtx6000:1           # 1 GPU is enough for any MoE up to ~30B params.
                                      # Swap to a100 / v100 / h100 if free.
                                      # Multi-GPU: --gres=gpu:rtx6000:2 etc.
#SBATCH --cpus-per-task=8              # 4 minimum, 8 comfortable
#SBATCH --mem=16G                      # 16 GB plenty for 670M-7B models;
                                      # scale up: 32 GB for 14B, 64 GB for Mixtral
#SBATCH --time=01:00:00                # 1 h for 670M; 2 h for larger models
#SBATCH --output=moe-breakdown-%j.out
#SBATCH --error=moe-breakdown-%j.err

set -euo pipefail

# ---------- environment ----------
module purge
# Adjust to whatever `module avail cuda` shows on ICE:
module load gcc/12.3.0 python/3.11 cuda/12.4

# ---------- storage setup (idempotent) ----------
# The framework + HF cache + a MoE checkpoint can easily exceed ICE's
# 5-15 GB home quota.  Symlink caches into scratch on every run.
mkdir -p ~/scratch/.cache ~/scratch/.conda
ln -sfn ~/scratch/.cache ~/.cache
ln -sfn ~/scratch/.conda ~/.conda

# HF cache goes to scratch too (don't pollute home)
export HF_HOME=~/scratch/hf_cache_${SLURM_JOB_ID}
mkdir -p "$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"

# Runs go to scratch (results get scp'd back at the end)
RUNS_DIR=~/scratch/moe-breakdown-runs
mkdir -p "$RUNS_DIR"
TS=$(date +%Y-%m-%d-%H%M)
OUT_DIR="$RUNS_DIR/${SLURM_JOB_ID}-${TS}"
mkdir -p "$OUT_DIR"

# ---------- venv ----------
# Create or reuse a venv in scratch so we don't bloat home.
VENV=~/scratch/moe-breakdown-venv
if [[ ! -d "$VENV" ]]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install -e ".[hf]"
fi
source "$VENV/bin/activate"

# ---------- run ----------
# Default: tiny model (smoke test of the pipeline)
MODEL="${MODEL:-}"
TOKENS="${TOKENS:-32}"
PASSES="${PASSES:-3}"
WARMUP="${WARMUP:-1}"
BITS="${BITS:-}"              # if set (4 or 8), use the quantized path
VLLM_TP="${VLLM_TP:-}"          # if set (e.g. 2), launch vLLM with TP=N
                                # and profile against it -- gives REAL 2-GPU data
VLLM_PORT="${VLLM_PORT:-8000}"

cd ~/scratch/moe-breakdown

if [[ -z "$MODEL" ]]; then
    echo "[moe-breakdown] no MODEL specified; running tiny backend"
    python3 scripts/run_breakdown.py \
        --backend tiny \
        --passes "$PASSES" \
        --out "$OUT_DIR"
elif [[ -n "$BITS" ]]; then
    echo "[moe-breakdown] profiling $MODEL with ${BITS}-bit quantization"
    python3 examples/run_quantized_qwen.py \
        --model "$MODEL" \
        --bits "$BITS" \
        --max-new-tokens "$TOKENS" \
        --n-passes "$PASSES" \
        --warmup "$WARMUP" \
        --out "$OUT_DIR"
elif [[ -n "$VLLM_TP" ]]; then
    # Multi-GPU mode: launch vLLM with tensor-parallel=$VLLM_TP in the
    # background, wait for it to come up, then profile against it.
    # Use ALL the GPUs we requested (must match --gres).
    echo "[moe-breakdown] launching vLLM with TP=$VLLM_TP"
    vllm serve "$MODEL" \
        --tensor-parallel-size "$VLLM_TP" \
        --port "$VLLM_PORT" \
        --max-model-len 4096 \
        > "$OUT_DIR/vllm.log" 2>&1 &
    VLLM_PID=$!
    echo "[moe-breakdown] vLLM PID=$VLLM_PID, waiting for it to come up ..."

    # Wait up to 5 minutes for vLLM to start serving
    for i in $(seq 1 60); do
        if curl -sf "http://localhost:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
            echo "[moe-breakdown] vLLM is up (took ~${i}*5 sec)"
            break
        fi
        sleep 5
    done
    if ! curl -sf "http://localhost:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
        echo "[moe-breakdown] ERROR: vLLM failed to start; see $OUT_DIR/vllm.log"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi

    # Profile against it
    echo "[moe-breakdown] profiling against vLLM (TP=$VLLM_TP)"
    python3 scripts/run_breakdown.py \
        --backend vllm \
        --model "$MODEL" \
        --base-url "http://localhost:$VLLM_PORT" \
        --tokens "$TOKENS" \
        --passes "$PASSES" \
        --out "$OUT_DIR" \
        --title "PACE-ICE job ${SLURM_JOB_ID} -- $MODEL via vLLM (TP=$VLLM_TP)"

    # Cleanup
    echo "[moe-breakdown] tearing down vLLM"
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
else
    echo "[moe-breakdown] profiling $MODEL"
    python3 scripts/run_breakdown.py \
        --backend transformers \
        --model "$MODEL" \
        --tokens "$TOKENS" \
        --passes "$PASSES" \
        --warmup "$WARMUP" \
        --out "$OUT_DIR" \
        --title "PACE-ICE job ${SLURM_JOB_ID} -- ${MODEL}"
fi

# ---------- summary ----------
echo
echo "[moe-breakdown] artifacts in $OUT_DIR:"
ls -la "$OUT_DIR"
echo
echo "To fetch the chart back to your laptop:"
echo "  scp <gt-username>@login-ice.pace.gatech.edu:$OUT_DIR/breakdown.png ."
echo
echo "To fetch the placement JSON (for expert-rack decisions):"
echo "  scp <gt-username>@login-ice.pace.gatech.edu:$OUT_DIR/placement.json ."
