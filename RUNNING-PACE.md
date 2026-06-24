# Running fused-moe-kernel-study on Georgia Tech PACE (A100)

This suite is intended to run from scratch storage with a `uv`-managed env.

## Suggested layout

```bash
mkdir -p ~/scratch
cd ~/scratch
git clone <your-repo-url> fused-moe-kernel-study
cd fused-moe-kernel-study
cp configs/study.pace.a100.yaml configs/study.pace.local.yaml
```

Edit:

- `slurm.workdir`
- `slurm.uv_env_dir`
- `kernel_shapes`
- `parallel_points`

## Recommended module stack

The default PACE-oriented config uses:

```yaml
modules: [gcc/12.3.0, python/3.11, cuda/12.4]
```

If your cluster image differs, change the YAML.

## uv bootstrap

The sbatch launcher will automatically create and reuse the env at:

```yaml
slurm:
  uv_env_dir: /home/$USER/scratch/fused-moe-kernel-study-venv
```

It uses:

```bash
uv venv <env>
uv pip install -e .
uv pip install "$VLLM_SPEC"
```

Default:

```bash
export VLLM_SPEC=vllm
```

If you need a particular build:

```bash
export VLLM_SPEC='vllm==0.10.2'
```

or point to a local wheel / editable checkout.

## Qwen3-30B-A3B initial smoke study

There is a ready-made initial config and launcher:

```bash
bash scripts/submit_qwen3_30b_a3b_initial.sh dry-run
bash scripts/submit_qwen3_30b_a3b_initial.sh submit
```

## Dry-run job generation

```bash
python3 scripts/submit_slurm_study.py \
  --config configs/study.pace.local.yaml \
  --dry-run
```

## Submit

```bash
export VLLM_SPEC=vllm
python3 scripts/submit_slurm_study.py \
  --config configs/study.pace.local.yaml
```

## Output

```text
runs/<study_name>/
├── slurm-logs/
├── tp1-ep1/
├── tp1-ep2/
├── tp1-ep4/
└── ...
```

## Aggregate all points

```bash
python3 scripts/aggregate_results.py \
  --study-root runs/<study_name>
```

## NCU / NSYS on one condition

For profiler runs, do **not** profile the full sweep first. Pick one representative point and run `scripts/run_one_condition.py` under the wrapper.

### NCU example

```bash
export PROFILE_RANK=0
export NCU_OUT_BASE=$PWD/ncu/deepep-ep4-tok1024-alpha4
export NCU_SET=full
export NCU_EXTRA_ARGS='--nvtx --nvtx-include moe_kernel_timed'

srun --ntasks=4 --gpus-per-task=1 \
  bash scripts/wrap_ncu.sh \
  python3 scripts/run_one_condition.py \
    --shape-name mixtral_like \
    --hidden-size 4096 \
    --intermediate-size 14336 \
    --num-experts 8 \
    --topk 2 \
    --tp-size 1 \
    --ep-size 4 \
    --num-tokens 1024 \
    --alpha 4.0 \
    --all2all-backend deepep_low_latency
```

### NSYS example

```bash
export PROFILE_RANK=0
export NSYS_OUT_BASE=$PWD/nsys/deepep-ep4-tok1024-alpha4
export NSYS_TRACE='cuda,nvtx,osrt'

srun --ntasks=4 --gpus-per-task=1 \
  bash scripts/wrap_nsys.sh \
  python3 scripts/run_one_condition.py \
    --shape-name mixtral_like \
    --hidden-size 4096 \
    --intermediate-size 14336 \
    --num-experts 8 \
    --topk 2 \
    --tp-size 1 \
    --ep-size 4 \
    --num-tokens 1024 \
    --alpha 4.0 \
    --all2all-backend deepep_low_latency
```

The benchmark emits NVTX ranges:

- `moe_kernel_timed`
- `moe_kernel_bucket_profile`
- `moe_kernel_warmup`

so you can filter on the timed region.

## Interconnect inspection

Inside an interactive A100 job, run:

```bash
python3 scripts/dump_topology.py
cat topology_snapshot.json
```

This captures:

- `nvidia-smi -L`
- `nvidia-smi topo -m`
- `nvidia-smi nvlink --status`
- PCIe / Mellanox devices
- `ibv_devinfo` if available
