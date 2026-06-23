#!/bin/bash
# Example Slurm batch script for running moe-breakdown on a cluster.
#
# Usage:
#   sbatch examples/run_slurm.sbatch
#   # or with overrides:
#   sbatch --export=ALL,MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1" \
#          examples/run_slurm.sbatch
#
# Outputs land in $SCRATCH/$SLURM_JOB_ID/breakdown/ and you can `scp` them
# back to your laptop.

#SBATCH --job-name=moe-breakdown
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=moe-breakdown-%j.out
#SBATCH --error=moe-breakdown-%j.err

set -euo pipefail

# ---------- environment ----------
module purge
module load cuda/12.3 python/3.11

# Optional: virtualenv / conda env with moe-breakdown installed
source ~/moe-breakdown-venv/bin/activate

# Pick the model.  Default: tiny in-tree model (works without network).
MODEL="${MODEL:-}"

# Where to put artifacts.  Defaults to ./runs/<backend>-<jobid>/ inside the
# repo checkout; override with the RUNS_DIR env var to use $SCRATCH on a
# shared filesystem.
RUNS_DIR="${RUNS_DIR:-$(dirname "$(readlink -f "$0")")/../runs}"
mkdir -p "$RUNS_DIR"
TS="$(date +%Y-%m-%d-%H%M)"
OUT_DIR="${RUNS_DIR}/${SLURM_JOB_ID}-${TS}"
mkdir -p "$OUT_DIR"

# ---------- run ----------
if [[ -z "$MODEL" ]]; then
    echo "[moe-breakdown] no MODEL specified; running tiny backend"
    python3 scripts/run_breakdown.py \
        --backend tiny \
        --passes 5 \
        --out "$OUT_DIR"
else
    export HF_HUB_CACHE="${SCRATCH:-/tmp}/hf_cache_${SLURM_JOB_ID}"
    mkdir -p "$HF_HUB_CACHE"

    python3 scripts/run_breakdown.py \
        --backend transformers \
        --model "$MODEL" \
        --tokens "${TOKENS:-32}" \
        --passes "${PASSES:-3}" \
        --warmup "${WARMUP:-1}" \
        --out "$OUT_DIR" \
        --title "Slurm job ${SLURM_JOB_ID} -- ${MODEL}"
fi

# ---------- summary ----------
echo "[moe-breakdown] artifacts in $OUT_DIR:"
ls -la "$OUT_DIR"
echo
echo "To fetch the chart back to your laptop:"
echo "  scp ${HOSTNAME}:${OUT_DIR}/breakdown.png ."
