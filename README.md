# fused-moe-kernel-study

A direct **MoE kernel sweep** benchmark suite for the characterization study you described.

For the concrete Georgia Tech ICE command-by-command workflow, see:

- `RUNNING-ICE.md`

## Scope

This suite measures **only** the fused MoE kernel path.

It does **not** use:

- tokenizer
- vLLM serving pipeline
- request scheduler
- attention
- KV cache
- full-model forward passes

It **does** use the vLLM fused-MoE kernel library as the implementation under test.
That is an important distinction:

- **not** vLLM serving
- **yes** direct invocation of the fused MoE kernel implementation

Each timed region is a CUDA-event bracket around the kernel call itself.

## Sweep knobs

The suite is built around the three knobs from your study:

1. routing imbalance ratio `alpha`
2. number of tokens
3. TP / EP degree

across **two representative kernel shapes**.

## High-level design

A different `(tp, ep)` point implies a different distributed layout, so TP/EP is handled as **one separate distributed job per point**.

Inside each job, the suite sweeps:

- token count
- routing imbalance `alpha`
- kernel shape

That gives you a clean study structure:

- outer sweep: TP/EP job grid
- inner sweep: tokens × alpha × shape

## Python API

The package root now exposes the benchmark API directly:

```python
from fused_moe_kernel_study import (
    MoEKernelConfig,
    KernelMeasurement,
    measure_moe_kernel_latency,
    run_sweep,
)
```

## Georgia Tech PACE notes

This repo now includes a PACE-oriented config:

- `configs/study.pace.a100.yaml`

and a `uv` bootstrap flow inside the Slurm launch path.

The expectation is that you run this from scratch storage, e.g.:

- repo: `~/scratch/moe-breakdown`
- env: `~/scratch/moe-breakdown-venv`

The example config assumes:

- `qos: coc-ice`
- `cuda/12.4`
- `python/3.11`
- `gcc/12.3.0`
- A100 nodes requested with `--gres=gpu:a100:N`

If you are using PACE nodes with A100s, the cards are typically **40 GB or 80 GB**, not 60 GB. The benchmark itself does not assume either capacity, except that larger token grids and larger EP points are easier on 80 GB cards.

## File layout

```text
fused-moe-kernel-study/
├── configs/
│   ├── study.example.yaml
│   └── study.pace.a100.yaml
├── scripts/
│   ├── bootstrap_uv_env.sh
│   ├── run_direct_moe_sweep.py
│   ├── submit_slurm_study.py
│   ├── aggregate_results.py
│   └── dump_topology.py
├── slurm/
│   └── run_direct_moe_sweep.sbatch
└── src/fused_moe_kernel_study/
    ├── config.py
    ├── distributed.py
    ├── reporting.py
    ├── routing.py
    ├── runner.py
    └── vllm_adapter.py
```

## What gets measured

For each condition, the suite constructs:

- synthetic `hidden_states`
- synthetic `w1` / `w2` expert weights
- synthetic `topk_ids`
- synthetic `topk_weights`
- per-rank `expert_map`

Then it invokes the fused MoE kernel directly and records two kinds of measurements:

### 1. Primary latency measurement

CUDA-event timing around the kernel call itself:

- warmup iterations
- measured iterations
- per-rank timing vectors in milliseconds
- max-rank median latency
- max-rank mean latency
- mean latency across ranks
- observed routing imbalance from realized assignments

### 2. Optional bucket profiling

If `collect_buckets: true`, the suite also runs a short `torch.profiler` pass for each condition and categorizes events into buckets similar to the earlier `moe-breakdown` framework:

- `cpu_python`
- `cpu_native`
- `gpu_compute`
- `gpu_memory`
- `gpu_idle_gap`
- `gpu_idle_sync`
- `network`
- `mem_transfer`
- `allocator`

These bucket profiles are written separately from the main latency measurements.

## Inter-GPU communication behavior

Yes: when `ep > 1` and the selected all-to-all backend is an EP-capable backend such as `deepep_low_latency`, the direct kernel path is expected to dispatch tokens to experts hosted on other ranks and then combine the results back.

That means the benchmark is intended to exercise the same **expert-parallel communication path inside the fused MoE kernel backend itself**, without the rest of the serving stack.

Important nuance:

- `ep = 1` -> no cross-rank expert dispatch, so the `network` bucket should be near zero or absent
- `ep > 1` -> the backend may issue NCCL / all-to-all style communication internally during prepare/finalize

## Routing model

The gating network is bypassed entirely.

Instead, routing is generated directly.
A simple hot-expert distribution is used:

- `alpha = 1.0` -> balanced / uniform
- `alpha > 1.0` -> the first `hot_expert_count` experts are proportionally hotter

The suite records both:

- `alpha_requested`
- `alpha_observed`

so you can see how the sampled routing matched the intended imbalance.

## uv-based environment setup

The Slurm path supports **automatic `uv` environment bootstrap**.

If `uv_env_dir` is set in the YAML config, the sbatch script will:

1. load requested modules
2. create the env with `uv venv` if needed
3. install the local package with `uv pip install -e .`
4. optionally install a vLLM spec if `VLLM_SPEC` is set in the environment

The helper script is:

```bash
scripts/bootstrap_uv_env.sh
```

## Qwen3-30B-A3B initial scripts

I added an initial smoke-study config and profile scripts for Qwen3-30B-A3B:

- `configs/study.qwen3_30b_a3b.initial.yaml`
- `scripts/submit_qwen3_30b_a3b_initial.sh`
- `scripts/profile_qwen3_30b_a3b_two_conditions.sh`
- `slurm/profile_qwen3_30b_a3b_two_conditions.sbatch`

## PACE quick start

### 1. Put the repo in scratch

```bash
cd ~/scratch
git clone <your-repo-url> moe-breakdown
cd moe-breakdown
```

### 2. Edit the PACE config

Start from:

```bash
cp configs/study.pace.a100.yaml configs/study.pace.local.yaml
```

Then update at least:

- `kernel_shapes`
- `parallel_points`
- `workdir`
- `uv_env_dir`
- `time`
- `mem`

### 3. Dry-run submission

```bash
python3 scripts/submit_slurm_study.py \
  --config configs/study.pace.local.yaml \
  --dry-run
```

### 4. Submit the study

```bash
export VLLM_SPEC='vllm'
python3 scripts/submit_slurm_study.py \
  --config configs/study.pace.local.yaml
```

If you need a particular version, set for example:

```bash
export VLLM_SPEC='vllm==0.10.2'
```

or a source checkout / wheel path appropriate for your environment.

## Local / interactive run

For a fixed point such as `tp=1, ep=4`:

```bash
cd moe-breakdown
PYTHONPATH=src torchrun --nproc-per-node=4 \
  scripts/run_direct_moe_sweep.py \
  --config configs/study.example.yaml \
  --tp-size 1 \
  --ep-size 4
```

## Aggregate results

```bash
python3 scripts/aggregate_results.py \
  --study-root runs/fused-moe-characterization-pace-a100
```

## Output layout

For a point like `tp=1, ep=4`, outputs go to:

```text
runs/<study_name>/tp1-ep4/
├── hardware.json
├── results.csv
├── results.jsonl
├── results_summary.json
├── study_config.json
├── per_rank/
│   ├── mixtral_like-full_factorial-tp1-ep4-tok128-alpha1.000.json
│   └── ...
└── bucket_profiles/
    ├── mixtral_like-full_factorial-tp1-ep4-tok128-alpha1.000.json
    └── ...
```

`results.csv` includes both the main latency summary and, when bucket profiling is enabled, flattened bucket summaries such as:

- `bucket_max_rank_network_ms`
- `bucket_max_rank_gpu_compute_ms`
- `bucket_mean_rank_mem_transfer_ms`
- etc.

## NCU / NSYS wrappers

I added lightweight wrappers so you can profile **one specific condition** without changing the benchmark logic.

### Single-condition runner

Use:

```bash
scripts/run_one_condition.py
```

That script runs exactly one shape / token / alpha / TP / EP point and writes one JSON result.

### NCU wrapper

Use:

```bash
scripts/wrap_ncu.sh
```

Example under `srun` for a 4-rank EP job:

```bash
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
      --all2all-backend deepep_low_latency \
      --collect-buckets \
      --bucket-profile-iters 3 \
      --bucket-full-events
```

Useful env vars:

```bash
export PROFILE_RANK=0
export NCU_OUT_BASE=$PWD/ncu/mixtral-like-ep4
export NCU_SET=full
export NCU_EXTRA_ARGS='--nvtx --nvtx-include moe_kernel_timed'
```

### NSYS wrapper

Use:

```bash
scripts/wrap_nsys.sh
```

Example:

```bash
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

Useful env vars:

```bash
export PROFILE_RANK=0
export NSYS_OUT_BASE=$PWD/nsys/mixtral-like-ep4
export NSYS_TRACE='cuda,nvtx,osrt'
export NSYS_EXTRA_ARGS='--capture-range=nvtx'
```

### Why wrappers instead of embedding NCU/NSYS in Python?

Because `ncu` and `nsys` are external launchers. The wrapper approach lets:

- all distributed ranks still participate
- only one selected rank be profiled
- the exact same Python benchmark code run under both normal and profiled modes

The benchmark now emits NVTX ranges such as:

- `moe_kernel_timed`
- `moe_kernel_bucket_profile`
- `moe_kernel_warmup`

so you can use NVTX filters in NCU / NSYS.

## Topology probe

To inspect the node interconnect first:

```bash
python3 scripts/dump_topology.py
```

This writes:

- `nvidia-smi topo -m`
- NVLink status
- PCI device listing
- InfiniBand device info when present

## Notes on vLLM compatibility

vLLM's fused-MoE internals move around between versions.

The compatibility layer is:

- `src/fused_moe_kernel_study/vllm_adapter.py`

That file already tries multiple constructor paths for:

- `FusedMoEKernel` / `FusedMoEModularKernel`
- `TritonExperts` / `TritonOrDeepGemmExperts`
- `BatchedTritonExperts` / `BatchedTritonOrDeepGemmExperts`
- different `FusedMoEConfig` signatures

If your local vLLM checkout differs, that is the main file to patch.

## Caveat

This workspace does not have your real PACE GPU/vLLM environment, so I could not execute the real kernel path end-to-end here.

What I did build is the benchmark suite structure, sweep logic, Slurm orchestration, `uv` bootstrap path, PACE-oriented config, and version-tolerant vLLM adapter layer so you can take it onto the cluster and finish the last mile against your exact installed vLLM build.
