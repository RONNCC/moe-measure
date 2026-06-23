# Running moe-breakdown — Docker, Slurm, and custom models

A quick-reference guide for running `moe-breakdown` against different MoE
models in different environments.

## Table of contents

1. [Pick a model](#pick-a-model)
2. [Run locally (no Docker)](#run-locally-no-docker)
3. [Run in Docker](#run-in-docker)
4. [Run on Slurm](#run-on-slurm)
5. [Troubleshooting](#troubleshooting)

---

## Pick a model

Any HuggingFace MoE checkpoint works.  Here are the ones tested:

| Model | Size | Experts | Architecture | Notes |
|---|---|---|---|---|
| **`PrimeIntellect/qwen3-moe-tiny`** | 670M | 16 (top-4) | Qwen3 MoE | **Recommended default** — real MoE on a single GPU |
| `yujiepan/phi-moe-tiny-random` | 2.5M | 8 (top-2) | Phi MoE | Tiny, runs anywhere (CPU even) |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | 46.7B | 8 (top-2) | Mixtral | The reference MoE |
| `Qwen/Qwen1.5-MoE-A2.7B` | 14.3B | 60 (top-4) | Qwen MoE | Many experts, harder routing |
| `deepseek-ai/deepseek-moe-16b-base` | 16.4B | 64 routed + 2 shared | DeepSeek | Mixed shared/routed |
| `databricks/dbrx-base` | 132B | 16 (top-4) | DBRX | Fine-grained MoE |

You can also point at any GGUF / MLX / LM Studio model via the `vllm`
backend — see [§ vLLM](#vllm-mode) below.

---

## Run locally (no Docker)

```bash
cd moe-breakdown
pip install -e ".[hf]"

# CPU-only -- tiny model
PYTHONPATH=src python3 scripts/run_breakdown.py \
    --backend transformers \
    --model yujiepan/phi-moe-tiny-random \
    --passes 3 \
    --out runs/yujiepan-cpu

# GPU -- recommended default (PrimeIntellect qwen3-moe-tiny, fits any single GPU)
PYTHONPATH=src python3 scripts/run_breakdown.py \
    --backend transformers \
    --model PrimeIntellect/qwen3-moe-tiny \
    --tokens 32 --passes 3 \
    --out runs/qwen3-moe-tiny

# Larger MoE -- Mixtral 8x7B needs ~100 GB VRAM (multi-GPU box)
PYTHONPATH=src python3 scripts/run_breakdown.py \
    --backend transformers \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tokens 32 --passes 3 \
    --out runs/mixtral-gpu
```

Results land in `runs/<your-out-dir>/` with the 4-artifact bundle
(`breakdown.png`, `breakdown.csv`, `breakdown.json`, `events.jsonl`).

---

## Run in Docker

### Build once

```bash
docker build -t moe-breakdown .
```

### Quick demo (no GPU required)

```bash
docker run --rm moe-breakdown \
    --backend synthetic \
    --out /runs/demo
```

### Profile a real MoE model (GPU required)

```bash
# Pick one of these:

# 0. PrimeIntellect/qwen3-moe-tiny -- RECOMMENDED DEFAULT. 670M params,
#    fits any single GPU, has 16 experts (top-4) so the MoE pattern is
#    clearly visible in the breakdown.
docker run --rm --gpus all -v $PWD/runs:/runs moe-breakdown \
    --backend transformers \
    --model PrimeIntellect/qwen3-moe-tiny \
    --tokens 32 --passes 3 \
    --out /runs/qwen3-moe-tiny

# 1. Mixtral 8x7B -- needs ~100 GB VRAM with bf16
docker run --rm --gpus all -v $PWD/runs:/runs moe-breakdown \
    --backend transformers \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tokens 32 --passes 3 \
    --out /runs/mixtral-8x7b

# 2. Qwen MoE 2.7B -- ~6 GB VRAM, fits on most GPUs
docker run --rm --gpus all -v $PWD/runs:/runs moe-breakdown \
    --backend transformers \
    --model Qwen/Qwen1.5-MoE-A2.7B \
    --tokens 32 --passes 3 \
    --out /runs/qwen-moe-2.7b

# 3. DeepSeek-MoE 16B -- needs ~32 GB VRAM
docker run --rm --gpus all -v $PWD/runs:/runs moe-breakdown \
    --backend transformers \
    --model deepseek-ai/deepseek-moe-16b-base \
    --tokens 32 --passes 3 \
    --out /runs/deepseek-moe-16b

# 4. DBRX base -- 132B, needs 4+ GPUs
docker run --rm --gpus all -v $PWD/runs:/runs moe-breakdown \
    --backend transformers \
    --model databricks/dbrx-base \
    --tokens 32 --passes 3 \
    --out /runs/dbrx
```

### Quantized models (4-bit / 8-bit)

Use `examples/run_quantized_qwen.py` (built into the image):

```bash
docker run --rm --gpus all -v $PWD/runs:/runs moe-breakdown \
    python3 /opt/moe-breakdown/examples/run_quantized_qwen.py \
    --model PrimeIntellect/qwen3-moe-tiny --bits 4 \
    --out /runs/qwen3-moe-tiny-4bit

# Or for the larger Qwen MoE:
docker run --rm --gpus all -v $PWD/runs:/runs moe-breakdown \
    python3 /opt/moe-breakdown/examples/run_quantized_qwen.py \
    --model Qwen/Qwen1.5-MoE-A2.7B --bits 4 \
    --out /runs/qwen-moe-2.7b-4bit
```

### Use docker-compose

```bash
# Edit the model path in docker-compose.yml first, then:
docker compose run --rm mixtral-8x7b       # /runs/mixtral-8x7b/
docker compose run --rm qwen-moe           # /runs/qwen-moe/
docker compose run --rm deepseek-moe       # /runs/deepseek-moe/
docker compose run --rm demo               # no GPU, synthetic baseline
```

### Profile a running vLLM server

```bash
# On the host: start vLLM
vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tensor-parallel-size 2 --max-model-len 8192

# In a separate terminal / container:
docker run --rm --network host -v $PWD/runs:/runs moe-breakdown \
    --backend vllm \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --base-url http://localhost:8000 \
    --passes 10 \
    --out /runs/vllm-mixtral
```

### Profile your own (custom) model

```bash
docker run --rm --gpus all -v $PWD/runs:/runs moe-breakdown \
    python3 /opt/moe-breakdown/examples/run_my_own_model.py \
    --out /runs/custom-model
```

Edit `examples/run_my_own_model.py` to swap in your model class — see the
inline comments.

### Important Docker flags

| Flag | Why |
|---|---|
| `--gpus all` | Required for real model profiling |
| `--network host` | Required for vLLM-mode (so it can reach your vLLM server) |
| `-v $PWD/runs:/runs` | So artifacts land in your local `runs/` directory |
| `--rm` | Don't accumulate stopped containers |

---

## Run on Slurm

### Single-GPU job (most common)

```bash
# Edit MODEL in the script or pass via env:
MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1" \
TOKENS=64 PASSES=3 \
sbatch examples/run_slurm.sbatch
```

The default `examples/run_slurm.sbatch` requests:

- 1 node, 2 GPUs, 8 CPUs, 64 GB RAM, 30 min walltime
- module loads `cuda/12.3` + `python/3.11`
- writes artifacts to `runs/<SLURM_JOB_ID>-<TIMESTAMP>/`

To check progress while it runs:

```bash
squeue --me                    # job status
sacct -j <JOBID> --format=State,Elapsed,MaxRSS,ReqGRES
tail -f moe-breakdown-<JOBID>.out
```

To fetch results:

```bash
scp $HOSTNAME:moe-breakdown/runs/<JOBID>-<TIMESTAMP>/breakdown.png .
```

### PACE-ICE (Georgia Tech)

See [RUNNING-ICE.md](RUNNING-ICE.md) for the PACE-ICE-specific section
(setup, modules, QoS, GPU-type selection, VPN, storage symlinks).

The PACE-ICE-tuned Slurm script is at `examples/run_slurm_pace_ice.sbatch`.

### Multi-GPU / multi-node

For multi-node MoE profiling, wrap the model in `torch.distributed` and
gather per-rank events into rank 0:

```python
# examples/run_distributed.py  (template -- not yet in the image)
import torch.distributed as dist
dist.init_process_group(backend="nccl")
rank = dist.get_rank()

with torch.profiler.profile(...) as prof:
    model(*inputs)

events = [ev_to_dict(e) for e in prof.events()]
gathered = [None] * dist.get_world_size()
dist.all_gather_object(gathered, events)
all_events = sum(gathered, [])  # flatten

if rank == 0:
    breakdown = categorize_dicts(all_events)
    render_chart(breakdown, ...)
```

Each rank profiles its own slice (which is what you'd want anyway — the
experts that *this rank* hosts are the ones whose network time matters).
Then `all_gather_object` pulls everyone's events into rank 0 for the
combined breakdown chart.

### Override Slurm script defaults

```bash
# Bigger model, more passes, longer time:
sbatch --export=ALL,MODEL="deepseek-ai/deepseek-moe-16b-base",PASSES=5 \
       --time=02:00:00 --mem=128G \
       examples/run_slurm.sbatch

# Different partition (your cluster may have GPU nodes on `gpu-h100`):
sbatch --partition=gpu-h100 examples/run_slurm.sbatch

# Use scratch for the artifact bundle (large clusters often have $SCRATCH):
RUNS_DIR=$SCRATCH/moe-breakdown sbatch examples/run_slurm.sbatch
```

### Run topology analysis on Slurm

Same as locally — just pass the `--topology` flag:

```bash
sbatch --export=ALL,MODEL="Qwen/Qwen1.5-MoE-A2.7B",TOPOLOGY=1 \
       examples/run_slurm.sbatch
```

But you'd need to edit the Slurm script to forward `TOPOLOGY=1` to the
moe-breakdown CLI.  Or just run the topology analysis locally afterwards:

```bash
# On the cluster: profile and save the routing pattern
python3 scripts/run_breakdown.py --backend transformers \
    --model Qwen/Qwen1.5-MoE-A2.7B --out runs/qwen-routing --topology \
    --num-experts 60 --num-racks 4 --gpus-per-rack 8
```

---

## Troubleshooting

### "no CUDA GPU detected"

You're running on a CPU-only box.  Options:

* Use the `tiny` or `synthetic` backend (works on CPU).
* Use `--hybrid` to layer a synthetic GPU projection on the real CPU run.
* Get a GPU box (vast.ai, RunPod, Lambda Labs all rent H100s for $2/hr).

### "OOM (out of memory)"

The model is too big for your GPU.  Either:

* Use a quantized version (`--bits 4` via `run_quantized_qwen.py`).
* Use a smaller `--tokens` (shorter sequence = less memory).
* Use a smaller model.

### "401 Unauthorized" from HuggingFace

Set a token:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
# or pass --hf-token in docker:
docker run -e HF_TOKEN=$HF_TOKEN ...
```

### "model is gated" (Llama, etc.)

You need to accept the license on huggingface.co first, then make sure
your HF token has access.  Then `huggingface-cli login` before running.

### The transfer matrix heatmap is mostly empty

Your expert count is high but tokens per batch is low.  Increase
`--num-experts` or scale up the batch.  Edit the synthetic defaults in
`scripts/run_breakdown.py` (look for `T = 65536`) to make a more
realistic projection.

### The breakdown only shows CPU buckets

You're on CPU hardware.  See "no CUDA GPU detected" above.

### Results look the same across runs

Likely the framework cached something.  Try:

```bash
rm -rf ~/.cache/huggingface ~/.cache/torch
PYTHONPATH=src python3 scripts/run_breakdown.py ...
```

### "synthetic" tag in events.jsonl -- how to filter

```bash
# Real CPU events only:
jq 'select(.synthetic == false)' runs/<name>/events.jsonl > real.jsonl

# Projected GPU events only:
jq 'select(.synthetic == true)' runs/<name>/events.jsonl > projected.jsonl
```

This is how you tell apart measured data from the hybrid-mode
projection.  When you're on a real GPU box the synthetic tag is always
`false` (everything is measured).
