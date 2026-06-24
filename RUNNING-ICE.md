# Running fused-moe-kernel-study on Georgia Tech ICE

This is the **pinned first-run guide** for your current setup.

It assumes:

- repo path: `~/scratch/moe-breakdown`
- temporary heavy-weight runtime state lives in `$TMPDIR`
- modules:
  - `gcc/12.3.0`
  - `python/3.11`
  - `cuda/12.1.1`
- benchmark target: **Qwen3-30B-A3B initial direct-kernel validation**

The goal here is not to give every possible option. The goal is to give you a small number of exact commands that should work first.

There is also a pinned helper script now:

```bash
bash scripts/ice_qwen3_first_run.sh dry-run
bash scripts/ice_qwen3_first_run.sh submit
```

and, inside a GPU allocation:

```bash
bash scripts/ice_qwen3_first_run.sh setup
bash scripts/ice_qwen3_first_run.sh profile-none
bash scripts/ice_qwen3_first_run.sh profile-nsys
bash scripts/ice_qwen3_first_run.sh profile-ncu
```

---

## 0. Current recommended layout

You already moved the repo to scratch:

```bash
mkdir -p ~/scratch
rsync -a /tmp/moe-breakdown/ ~/scratch/moe-breakdown/
cd ~/scratch/moe-breakdown
```

That is the correct place for the repo.

### Why not run from `/tmp/moe-breakdown`?

Because for Slurm jobs `/tmp` is usually:

- node-local
- ephemeral
- not shared between login and compute nodes

So:

- repo stays in `~/scratch/moe-breakdown`
- env/cache/HF/triton cache go to `$TMPDIR` during actual compute jobs

---

## 1. Pinned defaults used by the scripts

These are the defaults I pinned into the current workflow:

- repo/workdir: `~/scratch/moe-breakdown`
- uv env in jobs: `$TMPDIR/moe-breakdown-venv`
- cache dirs in jobs:
  - `$TMPDIR/uv-cache`
  - `$TMPDIR/hf-cache`
  - `$TMPDIR/triton-cache`
  - `$TMPDIR/torchinductor-cache`
- modules:
  - `gcc/12.3.0`
  - `python/3.11`
  - `cuda/12.1.1`

---

## 2. Quick dry-run on the login node

This does **not** require `torch` anymore.

```bash
cd ~/scratch/moe-breakdown
module purge
module load gcc/12.3.0 python/3.11 cuda/12.1.1
bash scripts/submit_qwen3_30b_a3b_initial.sh dry-run
```

If you copied the default Qwen config and want to use your local copy:

```bash
cp configs/study.qwen3_30b_a3b.initial.yaml configs/study.qwen3_30b_a3b.local.yaml
CONFIG_PATH=$PWD/configs/study.qwen3_30b_a3b.local.yaml \
  bash scripts/submit_qwen3_30b_a3b_initial.sh dry-run
```

---

## 3. Submit the initial Qwen3-30B-A3B study

Default config:

```bash
cd ~/scratch/moe-breakdown
module purge
module load gcc/12.3.0 python/3.11 cuda/12.1.1
export VLLM_SPEC='vllm'
bash scripts/submit_qwen3_30b_a3b_initial.sh submit
```

With your local config copy:

```bash
cd ~/scratch/moe-breakdown
module purge
module load gcc/12.3.0 python/3.11 cuda/12.1.1
export VLLM_SPEC='vllm'
CONFIG_PATH=$PWD/configs/study.qwen3_30b_a3b.local.yaml \
  bash scripts/submit_qwen3_30b_a3b_initial.sh submit
```

---

## 4. Watch the jobs

```bash
squeue --me
```

Look for logs:

```bash
ls runs/qwen3-30b-a3b-initial/slurm-logs
```

Tail one:

```bash
tail -f runs/qwen3-30b-a3b-initial/slurm-logs/<jobname>-<jobid>.out
```

---

## 5. Aggregate initial-study results

```bash
cd ~/scratch/moe-breakdown
python3 scripts/aggregate_results.py \
  --study-root runs/qwen3-30b-a3b-initial
```

---

# Exact environment bootstrap

I added a script for the fixed ICE setup.

## 6. Use the pinned setup script inside an allocation

Script:

```bash
scripts/setup_ice_env.sh
```

Default behavior:

- loads
  - `gcc/12.3.0`
  - `python/3.11`
  - `cuda/12.1.1`
- installs `uv` to `~/.local/bin` if missing
- creates env at
  - `$TMPDIR/moe-breakdown-venv` if `$TMPDIR` exists
  - otherwise falls back to scratch
- uses caches in `$TMPDIR` when available
- installs local package + `vllm`

You normally run it **inside a GPU allocation**, not on the login node.

---

# Interactive validation workflow

Use this if you want to test the direct-kernel path before submitting more jobs.

## 7. Request an interactive allocation

The active two-condition Qwen profile script uses at most 4 GPUs right now.

```bash
salloc --qos=coc-ice --gres=gpu:a100:4 --cpus-per-task=8 --mem=96G --time=02:00:00
```

Once the allocation starts:

```bash
cd ~/scratch/moe-breakdown
export VLLM_SPEC='vllm'
bash scripts/setup_ice_env.sh
```

That should leave you in an activated env.

If you need to reactivate manually after it finishes:

```bash
source "$TMPDIR/moe-breakdown-venv/bin/activate"
```

---

## 8. Validate the two-condition Qwen profile script without profiler overhead

Run this first:

```bash
cd ~/scratch/moe-breakdown
export PROFILE_TOOL=none
bash scripts/profile_qwen3_30b_a3b_two_conditions.sh
```

That validates:

- env bootstrap works
- direct kernel path works
- distributed launch works
- output files are created

---

## 9. Run the two-condition Qwen script with NSYS

```bash
cd ~/scratch/moe-breakdown
export PROFILE_TOOL=nsys
export PROFILE_RANK=0
bash scripts/profile_qwen3_30b_a3b_two_conditions.sh
```

---

## 10. Run the two-condition Qwen script with NCU

```bash
cd ~/scratch/moe-breakdown
export PROFILE_TOOL=ncu
export PROFILE_RANK=0
bash scripts/profile_qwen3_30b_a3b_two_conditions.sh
```

---

# Batch profiling workflow

If you do not want to use `salloc`, use the profiling batch script.

## 11. Submit the two-condition profile job with no profiler

```bash
cd ~/scratch/moe-breakdown
sbatch --export=ALL,PROFILE_TOOL=none slurm/profile_qwen3_30b_a3b_two_conditions.sbatch
```

## 12. Submit the two-condition profile job with NSYS

```bash
cd ~/scratch/moe-breakdown
sbatch --export=ALL,PROFILE_TOOL=nsys slurm/profile_qwen3_30b_a3b_two_conditions.sbatch
```

## 13. Submit the two-condition profile job with NCU

```bash
cd ~/scratch/moe-breakdown
sbatch --export=ALL,PROFILE_TOOL=ncu slurm/profile_qwen3_30b_a3b_two_conditions.sbatch
```

---

# Exact direct single-condition commands

These are useful when you want to debug one point manually.

## 14. One condition, no profiler

Inside an active allocation with the env set up:

```bash
cd ~/scratch/moe-breakdown
source "$TMPDIR/moe-breakdown-venv/bin/activate"

torchrun --nproc-per-node=4 python3 scripts/run_one_condition.py \
  --shape-name qwen3_30b_a3b_moe \
  --hidden-size 2048 \
  --intermediate-size 768 \
  --num-experts 128 \
  --topk 8 \
  --dtype bfloat16 \
  --activation silu \
  --all2all-backend deepep_low_latency \
  --tp-size 1 \
  --ep-size 4 \
  --num-tokens 1024 \
  --alpha 4.0 \
  --warmup-iters 10 \
  --measure-iters 40 \
  --collect-buckets \
  --bucket-profile-iters 3 \
  --bucket-full-events \
  --json-out runs/manual-qwen-condition.json
```

## 15. One condition with NSYS

```bash
cd ~/scratch/moe-breakdown
source "$TMPDIR/moe-breakdown-venv/bin/activate"
export PROFILE_RANK=0
export NSYS_OUT_BASE=$PWD/nsys/qwen-ep4-a4-t1024
export NSYS_TRACE='cuda,nvtx,osrt'
export NSYS_EXTRA_ARGS='--capture-range=nvtx'

torchrun --nproc-per-node=4 bash scripts/wrap_nsys.sh python3 scripts/run_one_condition.py \
  --shape-name qwen3_30b_a3b_moe \
  --hidden-size 2048 \
  --intermediate-size 768 \
  --num-experts 128 \
  --topk 8 \
  --dtype bfloat16 \
  --activation silu \
  --all2all-backend deepep_low_latency \
  --tp-size 1 \
  --ep-size 4 \
  --num-tokens 1024 \
  --alpha 4.0 \
  --warmup-iters 10 \
  --measure-iters 40 \
  --collect-buckets \
  --bucket-profile-iters 3 \
  --bucket-full-events
```

## 16. One condition with NCU

```bash
cd ~/scratch/moe-breakdown
source "$TMPDIR/moe-breakdown-venv/bin/activate"
export PROFILE_RANK=0
export NCU_OUT_BASE=$PWD/ncu/qwen-ep4-a4-t1024
export NCU_SET=full
export NCU_EXTRA_ARGS='--nvtx --nvtx-include moe_kernel_timed'

torchrun --nproc-per-node=4 bash scripts/wrap_ncu.sh python3 scripts/run_one_condition.py \
  --shape-name qwen3_30b_a3b_moe \
  --hidden-size 2048 \
  --intermediate-size 768 \
  --num-experts 128 \
  --topk 8 \
  --dtype bfloat16 \
  --activation silu \
  --all2all-backend deepep_low_latency \
  --tp-size 1 \
  --ep-size 4 \
  --num-tokens 1024 \
  --alpha 4.0 \
  --warmup-iters 10 \
  --measure-iters 40 \
  --collect-buckets \
  --bucket-profile-iters 3 \
  --bucket-full-events
```

---

# Interconnect inspection

Inside an allocation:

```bash
cd ~/scratch/moe-breakdown
source "$TMPDIR/moe-breakdown-venv/bin/activate"
python3 scripts/dump_topology.py
cat topology_snapshot.json
```

This captures:

- `nvidia-smi -L`
- `nvidia-smi topo -m`
- `nvidia-smi nvlink --status`
- PCIe / Mellanox devices
- `ibv_devinfo` if available

---

# Active initial Qwen profile conditions

The current profile script runs exactly these first two conditions:

1. `tp=1, ep=1, tokens=512, alpha=1.0`
2. `tp=1, ep=4, tokens=1024, alpha=4.0`

That is deliberate: validate the code path first.

After that, open:

```bash
scripts/profile_qwen3_30b_a3b_two_conditions.sh
```

and uncomment the additional conditions already listed there.

---

# If you uncomment 8-GPU conditions later

Then update:

```bash
slurm/profile_qwen3_30b_a3b_two_conditions.sbatch
```

from:

```bash
#SBATCH --gres=gpu:a100:4
```

to:

```bash
#SBATCH --gres=gpu:a100:8
```

---

# Minimal first-run checklist

Run these exact commands in order.

## Login node: dry-run + submit

```bash
cd ~/scratch/moe-breakdown
module purge
module load gcc/12.3.0 python/3.11 cuda/12.1.1
cp configs/study.qwen3_30b_a3b.initial.yaml configs/study.qwen3_30b_a3b.local.yaml
export VLLM_SPEC='vllm'
CONFIG_PATH=$PWD/configs/study.qwen3_30b_a3b.local.yaml bash scripts/submit_qwen3_30b_a3b_initial.sh dry-run
CONFIG_PATH=$PWD/configs/study.qwen3_30b_a3b.local.yaml bash scripts/submit_qwen3_30b_a3b_initial.sh submit
```

## Interactive validation after that

```bash
salloc --qos=coc-ice --gres=gpu:a100:4 --cpus-per-task=8 --mem=96G --time=02:00:00
cd ~/scratch/moe-breakdown
export VLLM_SPEC='vllm'
bash scripts/setup_ice_env.sh
export PROFILE_TOOL=none
bash scripts/profile_qwen3_30b_a3b_two_conditions.sh
```

If that works, repeat with:

```bash
export PROFILE_TOOL=nsys
bash scripts/profile_qwen3_30b_a3b_two_conditions.sh
```

and then:

```bash
export PROFILE_TOOL=ncu
bash scripts/profile_qwen3_30b_a3b_two_conditions.sh
```
