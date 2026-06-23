# Running moe-breakdown on PACE-ICE (Georgia Tech)

> Specific notes for [PACE-ICE](https://docs.pace.gatech.edu/ice_cluster/),
> the Georgia Tech instructional Slurm cluster.
>
> For generic Docker / Slurm instructions see [RUNNING.md](RUNNING.md).

### PACE-ICE (Georgia Tech) — specific notes

PACE-ICE works but needs a few tweaks because of its tight home-directory
quota (5–15 GB), specific module names, and GPU-type-granular partitions.

**First-time setup (do once after `ssh <gt-username>@login-ice.pace.gatech.edu`):**

```bash
# Move caches to scratch so the framework + HF models don't blow your quota
mkdir -p ~/scratch/.cache ~/scratch/.conda ~/scratch/hf_cache
ln -sfn ~/scratch/.cache ~/.cache
ln -sfn ~/scratch/.conda ~/.conda

# Clone moe-breakdown into scratch (not home)
cd ~/scratch
git clone <your-fork-url> moe-breakdown
cd moe-breakdown

# Create the venv in scratch
python3 -m venv ~/scratch/moe-breakdown-venv
~/scratch/moe-breakdown-venv/bin/pip install -e ".[hf]"
```

**Submit a job:**

```bash
# Default -- tiny in-tree smoke test, no GPU needed
sbatch examples/run_slurm_pace_ice.sbatch

# RECOMMENDED: PrimeIntellect/qwen3-moe-tiny (670M params, 16 experts
# top-4, fits any single GPU).  This is the default model for the
# framework and matches configs/qwen3-moe-tiny.yaml.
MODEL="PrimeIntellect/qwen3-moe-tiny" sbatch examples/run_slurm_pace_ice.sbatch

# Quantized 4-bit version (fits in <2 GB VRAM, even more flexible):
MODEL="PrimeIntellect/qwen3-moe-tiny" BITS=4 sbatch examples/run_slurm_pace_ice.sbatch

# Larger Qwen MoE (14.3B, ~6 GB VRAM):
MODEL="Qwen/Qwen1.5-MoE-A2.7B" sbatch examples/run_slurm_pace_ice.sbatch

# Mixtral 8x7B (needs ~100 GB VRAM, only on multi-GPU ICE nodes):
MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1" TOKENS=16 PASSES=2 \
    sbatch examples/run_slurm_pace_ice.sbatch
```

**Key things that differ from the generic Slurm script:**

| Thing | Generic script | PACE-ICE |
|---|---|---|
| Module name | `cuda/12.3 python/3.11` | `gcc/12.3.0 python/3.11 cuda/12.4` (verify with `module avail cuda`) |
| Partition | `--partition=gpu` | `--qos=coc-ice` or `--qos=pace-ice` (use `coc-grade` if you're a TA) |
| GPU request | `--gpus-per-node=2` | `--gres=gpu:a100:1` (or `v100`, `rtx6000`, `h100` — check `sinfo -o "%P %G"`) |
| Storage | `$SCRATCH/moe-breakdown/...` | `~/scratch/moe-breakdown-runs/...` |
| Time limit | `--time=00:30:00` | `--time=02:00:00` (ICE max) |
| HF cache | `$HF_HUB_CACHE` | `~/scratch/hf_cache_<jobid>/` |
| VPN | (assumed off-campus direct SSH) | **REQUIRED**: connect GlobalProtect first |

**Verifying module / partition names on your login:**

```bash
module avail cuda              # what's installed
module avail python            # which Python versions
sinfo -o "%P %G %D"            # partitions and GPU types
squeue -p coc-ice -t RUNNING   # how busy is the queue
```

**Multi-GPU runs (2+ GPUs)**

For real inter-GPU traffic measurements (AllToAll between experts on
different GPUs), you need an actually-distributed model.  The
`transformers` backend profiles a single forward pass — for true
multi-GPU you use the `vllm` backend with tensor parallelism.

```bash
# Set VLLM_TP=2 to launch vLLM with TP=2 in the same job, then profile it.
# The script automatically starts vLLM, waits for it to come up, profiles,
# and tears it down.

# On the Slurm header, request 2 GPUs of the same type:
#SBATCH --gres=gpu:rtx6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# At submit time, set VLLM_TP to match --gres count:
MODEL="PrimeIntellect/qwen3-moe-tiny" VLLM_TP=2 \
    sbatch examples/run_slurm_pace_ice.sbatch
```

The framework will then show:
* Real `gpu_compute` events (vLLM dispatches per-GPU)
* Real `network` events (AllToAll collectives for expert routing)
* Real `gpu_idle_gap` (per-GPU launch latency, sync between GPUs)
* A breakdown that actually answers "where did my multi-GPU time go"

If you'd rather run vLLM as a persistent service and profile from a
separate job, see the "Persistent vLLM server" section below.

**Resource requirements scale up for 2-GPU:**

| Resource | 1 GPU | 2 GPUs |
|---|---|---|
| `--gres` | `gpu:rtx6000:1` | `gpu:rtx6000:2` |
| `--cpus-per-task` | 8 | 16 |
| `--mem` | 16 GB | 64 GB |
| `--time` | 30 min | 1 h |

For 4+ GPUs (Mixtral 8x7B TP=4, DBRX TP=8, etc.) add ~32 GB RAM and
~8 cores per additional GPU.

**What to request in `#SBATCH` directives**

Recommended starting point for `PrimeIntellect/qwen3-moe-tiny` (670M params):

```bash
#SBATCH --qos=coc-ice                  # 'pace-ice' or 'coc-grade' if TA
#SBATCH --gres=gpu:rtx6000:1           # 1 GPU is plenty; swap to a100/v100/h100 if free
#SBATCH --cpus-per-task=8              # 4 minimum, 8 comfortable
#SBATCH --mem=16G                      # 16 GB for 670M-7B models
#SBATCH --time=01:00:00                # 1 h plenty; max on ICE is 2 h
```

Scaled for other models:

| Model | VRAM | Cores | Mem | Time |
|---|---|---|---|---|
| `PrimeIntellect/qwen3-moe-tiny` (670M) | ~2 GB | 4-8 | 16 GB | 30 min |
| `Qwen/Qwen1.5-MoE-A2.7B` (14.3B) | ~6 GB | 8 | 32 GB | 1 h |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` bf16 | ~100 GB | 16 | 64 GB | 2 h |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` 4-bit | ~30 GB | 16 | 64 GB | 2 h |
| `databricks/dbrx-base` (132B, 4-bit) | ~260 GB | 32 | 128 GB | 2 h, multi-GPU |

Check what's available before submitting:

```bash
sinfo -p coc-ice -o "%P %G %D %t"
sinfo -t idle -o "%N %G %C %m"           # idle nodes + GPU type + cores + RAM
squeue -p coc-ice -t RUNNING -o "%u %T %M"
```

A line like `ice-gpu-01 rtx6000:2 8 64000` means: 2 RTX 6000s, 8 cores, 64 GB RAM.

**Fetching results back to your laptop:**

```bash
scp <gt-username>@login-ice.pace.gatech.edu:~/scratch/moe-breakdown-runs/<jobid>-<timestamp>/breakdown.png .
```

### Using Open OnDemand (web UI)

If you'd rather use the browser instead of plain SSH, log into
https://ondemand-ice.pace.gatech.edu and pick an Interactive App:

| App | When to use it |
|---|---|
| **Coder** | Best if you want to edit the framework files / YAML configs |
| **VS Code** | Same as Coder but with VS Code's UI |
| **Jupyter** | Best for exploratory runs (view charts inline) |

**Do NOT use "VLLM + Jupyter"** — that's for serving LLMs, not for
profiling them. (Unless you specifically want to use the framework's
`vllm` backend to profile a running VLLM server.)

From any of the three, click **Launch**, configure (2h, 4-8 cores, 16+ GB,
1 GPU of whatever type is free — RTX 6000 is a safe default), then once
the session starts click **Connect** and open a Terminal. From there run
the same setup + sbatch steps described above.

**Typical flow:**

1. Click **Coder** (or Jupyter / VS Code)
2. Configure: 2 hours, 8 cores, 32 GB RAM, 1 GPU (RTX 6000)
3. Click **Launch** → wait → click **Connect**
4. Open Terminal (`Terminal → New Terminal`)
5. Run the one-time setup commands (storage symlinks, clone, venv)
6. `cd ~/scratch/moe-breakdown && sbatch examples/run_slurm_pace_ice.sbatch`
   - Default: tiny in-tree MoE smoke test (no model download, ~2 seconds)
   - Add `MODEL=...` to profile a real model, e.g.:
     ```bash
     MODEL="PrimeIntellect/qwen3-moe-tiny" sbatch examples/run_slurm_pace_ice.sbatch
     ```
7. Monitor with `squeue --me` and `tail -f moe-breakdown-*.out`
8. Open `~/scratch/moe-breakdown-runs/<jobid>-*/breakdown.png` in the file browser


**Common ICE gotchas:**

* "Disk quota exceeded" — your HF cache or venv is in home.  Run the
  symlink commands above.
* "GPU not available" — ICE has a limited pool of each GPU type.
  Try a different `--gres=gpu:<type>:1` or wait until the queue is empty.
* "Connection refused" — you forgot the VPN.  Connect GlobalProtect first.
* Module not found — `module avail cuda` to see what's actually installed;
  edit `examples/run_slurm_pace_ice.sbatch` accordingly.



## File reference

* `examples/run_slurm_pace_ice.sbatch` — the PACE-ICE-tuned Slurm script
  (handles storage symlinks, module loading, QoS, GPU-type selection).

## Quick checklist before your first job

- [ ] Connect to GT VPN (GlobalProtect) before SSH
- [ ] SSH into login-ice.pace.gatech.edu
- [ ] Run `mkdir -p ~/scratch/.cache ~/scratch/.conda && ln -sfn ~/scratch/.{.cache,.conda} ~`
- [ ] Clone the repo into `~/scratch/`, not `~/`
- [ ] Create the venv in `~/scratch/`, not `~/`
- [ ] Pick your GPU type: `sinfo -o "%P %G %D"`
- [ ] Pick your modules: `module avail cuda python`
- [ ] Submit: `sbatch examples/run_slurm_pace_ice.sbatch`
- [ ] Watch: `squeue --me`
- [ ] Fetch: `scp <gt-username>@login-ice.pace.gatech.edu:~/scratch/moe-breakdown-runs/<jobid>-*/breakdown.png .`
