# moe-breakdown

Execution-time breakdown for mixture-of-experts (MoE) models.

> **See [RUNNING.md](RUNNING.md) for generic Docker / Slurm
> instructions.**  For PACE-ICE (Georgia Tech) specifically, see
> [RUNNING-ICE.md](RUNNING-ICE.md).  This README focuses on what the
> framework does and why.

Given any MoE forward pass — Mixtral, Qwen-MoE, DeepSeek-MoE, DBRX, OLMoE,
custom, or a quantized checkpoint — `moe-breakdown` runs the model under
`torch.profiler`, categorizes every Kineto event into one of **nine buckets**
that explain where wall-clock time actually goes, and produces a chart +
JSON + CSV report bundle.

```
   bucket              %     time(ms)   count
   -------------------------------------------------------
   cpu_native         99.5%     14.863       32   ##################################################  (CPU — Native)
   allocator           0.4%      0.057        1     (Allocator)
   gpu_idle_sync       0.1%      0.020        1     (GPU Idle — sync)
```

A single `breakdown.png` chart is also generated, with three panels:

* **(a)** Time-share by bucket (horizontal bar, sorted by %)
* **(b)** Absolute time per bucket (stacked bar)
* **(c)** Top events per bucket (which kernels dominate each bucket)

## Buckets

| Bucket | What it captures |
|---|---|
| `cpu_python` | Python interpreter, dispatcher, autograd traversal |
| `cpu_native` | Native CPU ops (aten on CPU thread, tokenize, sample, data prep) |
| `gpu_compute` | Compute-bound GPU kernels (matmul, gemm, conv, attention) |
| `gpu_memory` | Memory-bound GPU kernels (norm, softmax, MoE dispatch) |
| `gpu_idle_gap` | Wall-clock gap between consecutive GPU kernels (CPU dispatch latency) |
| `gpu_idle_sync` | Explicit GPU sync stalls (`cudaStreamSynchronize`, `.item()`, `.cpu()`) |
| `network` | Collective communication (NCCL AllToAll / AllReduce / AllGather) |
| `mem_transfer` | DMA copies (H2D / D2H / D2D / memset) |
| `allocator` | CUDA caching allocator work (`cudaMalloc`, `cudaFree`, `aten::empty`) |

The categorisation is heuristic but explicit — every rule is a regex in
[`src/moe_breakdown/categorize.py`](src/moe_breakdown/categorize.py) and
you can read or extend it.  Unit tests in
[`tests/test_categorize.py`](tests/test_categorize.py) verify each bucket.

## Quick start

### 1. The tiny in-tree model — runs anywhere, no GPU needed

A 3.7 M-parameter MoE (3 experts × ~1 M each, top-1 routing) defined in
[`src/moe_breakdown/models/tiny_moe.py`](src/moe_breakdown/models/tiny_moe.py):

```bash
cd moe-breakdown
PYTHONPATH=src python3 scripts/run_breakdown.py --backend tiny --out runs/tiny-moe-cpu
# or, with a config file:
PYTHONPATH=src python3 scripts/run_breakdown.py \
    --backend tiny --config configs/tiny.yaml --passes 20
```

Useful when:
* You want to verify the framework end-to-end without a checkpoint download.
* You're developing new categoriser rules and want a known-input to test against.
* You're teaching / demoing — this is what I ran for the demo above.

### 2. Any HuggingFace MoE model — needs a GPU

```bash
# local
PYTHONPATH=src python3 scripts/run_breakdown.py \
    --backend transformers \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tokens 32 --passes 3 --out runs/mixtral

# docker
docker build -t moe-breakdown .
docker run --rm --gpus all -v $PWD/runs:/runs moe-breakdown \
    --backend transformers \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tokens 32 --passes 3 --out /runs/mixtral

# docker-compose
docker compose run --rm mixtral-8x7b
docker compose run --rm qwen-moe
docker compose run --rm deepseek-moe
```

### 2c. Expert placement / topology analysis

For a MoE model deployed across multiple GPUs, the dominant systems
question is: **which experts live on which GPU?**  AllToAll dispatch
time scales with the number of cross-rack transfers, so a good
placement can save a lot of network time.

```bash
PYTHONPATH=src python3 scripts/run_breakdown.py \
    --backend synthetic \
    --topology \
    --num-experts 100 \
    --num-racks 2 --gpus-per-rack 8 \
    --out runs/100-experts-topology
```

Output:

```
   total transfer TIME :  687.19 ms
   total transfer DATA :    2.00 GB  (2,147,480,256 bytes)

   strategy         intra(ms)  inter(ms)  total(ms)
   round-robin        37.10     343.59     380.69
   greedy             36.68     346.93     383.61
   cluster            79.77       2.21      81.98     <-- 4.6x faster
```

Five extra artifacts per topology run, all in `runs/<name>/`:

* `transfer_matrix_time.png`   - N x N heatmap of transfer **time** (us)
* `transfer_matrix_bytes.png`  - N x N heatmap of transfer **data volume** (bytes)
* `topology.png`               - experts laid out on GPUs, edges coloured by
                                intra-rack (green) vs inter-rack (red) traffic
* `placement_comparison.png`   - horizontal bars comparing each strategy
* `placement.json`             - explicit expert -> GPU mapping per strategy

Both heatmaps use the same N x N layout so you can compare them
side-by-side: a placement that reduces *time* but not *bytes* is
suspicious (link already saturated); a placement that reduces *bytes*
but not *time* is suspicious (per-message overhead dominates).

### 2b. Hybrid mode — populate all 9 buckets even on CPU

When you run on CPU the GPU-specific buckets (`gpu_compute`, `gpu_memory`,
`network`, etc.) will be empty because the hardware doesn't exercise those
paths.  Add `--hybrid` to layer a *synthetic GPU projection* on top of the
real CPU run, calibrated to your model's actual architecture.  The chart
then shows all 9 buckets and you can see where the time *would* go on GPU.

```bash
PYTHONPATH=src python3 scripts/run_breakdown.py \
    --backend transformers \
    --model yujiepan/phi-moe-tiny-random \
    --passes 3 \
    --hybrid \
    --out runs/yujiepan-hybrid
```

Output:
```
   bucket              %     time(ms)   count
   cpu_python           1.0%      1.500        7     (CPU — Python)
   cpu_native          17.0%     26.013       93     (CPU — Native)
   gpu_compute         56.2%     85.970       30     (GPU Compute)
   gpu_memory           3.9%      5.953        5     (GPU Memory-bound)
   gpu_idle_gap         0.6%      0.864        2     (GPU Idle — gap)
   gpu_idle_sync        4.8%      7.316        2     (GPU Idle — sync)
   network              8.9%     13.640        4     (Network)
   mem_transfer         4.1%      6.327        3     (Mem Transfer (DMA))
   allocator            3.5%      5.356        7     (Allocator)
```

Every event in `events.jsonl` is tagged `"synthetic": true|false` so you
can filter out the projection when you only want real CPU data.

> **Note:** the GPU-side numbers in hybrid mode are a calibrated
> *projection*, not measured GPU runs.  They are based on the model's
> expert count, FFN size, and routing pattern, with plausible time
> fractions.  For ground-truth GPU numbers, run on a real GPU box.

### 3. A running vLLM server

```bash
docker run --rm --network host moe-breakdown \
    --backend vllm --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --base-url http://localhost:8000 --passes 10 --out /runs/vllm
```

### 4. Quantized Qwen (4-bit / 8-bit)

```bash
pip install -e ".[hf]" bitsandbytes accelerate

python3 examples/run_quantized_qwen.py \
    --model Qwen/Qwen1.5-MoE-A2.7B --bits 4
```

### 5. Your own model

See [`examples/run_my_own_model.py`](examples/run_my_own_model.py).  Five-line template:

```python
model = MyModel(...).eval()
with torch.no_grad():
    with torch.profiler.profile(activities=[...], acc_events=True) as prof:
        model(input)

from moe_breakdown import categorize, render_chart
render_chart(categorize(prof), title="My Model", out_path="breakdown.png")
```

### 6. Slurm cluster

```bash
sbatch examples/run_slurm.sbatch

# Override the model via env var:
sbatch --export=ALL,MODEL="Qwen/Qwen1.5-MoE-A2.7B",TOKENS=64,PASSES=3 \
    examples/run_slurm.sbatch
```

The Slurm script handles module loading, scratch directories, and HF cache
isolation per job.  Outputs land in `$SCRATCH/moe-breakdown/$SLURM_JOB_ID/`.

For multi-node MoE profiling, the framework is rank-agnostic — each rank
profiles its own slice, and you gather results with `torch.distributed.gather`
into rank 0.  Wrap it as needed; the categorizer doesn't care.

## Reproducibility

Every run writes a self-contained artifact bundle:

```
runs/<name>/
├── breakdown.png      # the chart
├── breakdown.json     # full structured report
├── breakdown.csv      # bucket-level table (one row per bucket)
└── events.jsonl       # every categorized event, one per line
```

The CSV is designed to be diff-friendly: compare `breakdown.csv` from two
runs of the same model with different batch sizes / TP sizes / sequence
lengths to see exactly where the extra time went.

## Backends summary

| Backend | What it profiles | Use case |
|---|---|---|
| `tiny` | A 3.7 M-parameter in-tree MoE under `torch.profiler` | Local smoke test, demo, regression tests |
| `transformers` | Any HuggingFace MoE model under `torch.profiler` | Local benchmarking, ablations |
| `vllm` | A running vLLM server (in-process or remote) | Production serving diagnosis |
| `synthetic` | A synthetic MoE-shaped trace with configurable fractions | CI / testing the categorizer itself |

### Environment knobs

```bash
# Enable per-instance event capture (slower but enables gpu_idle_gap detection)
MOE_BREAKDOWN_FULL_EVENTS=1 moe-breakdown --backend transformers --model ...
```

## How the categorisation works

Each event has `(name, category, device)`.  We lower-case the name and
match against nine compiled regex tables, in priority order:

1. **Sync** — `cudastreamsynchronize`, `aten::item`, `aten::cpu` → `gpu_idle_sync`
2. **Network** — `cat == "communication"` or matches `*all_to_all*`, `*all_reduce*` → `network`
3. **Allocator** — `cudamalloc`, `caching_allocator`, `aten::empty` → `allocator`
4. **Memcpy** — `cat in {memcpy, memset, gpu_memcpy}` or `*memcpy*` → `mem_transfer`
5. **CUDA runtime** — `cat == "cuda_runtime"` → `cpu_native`
6. **CPU** — `device == "cpu"` → `cpu_native` (or `cpu_python` if name matches Python/dispatcher patterns)
7. **GPU kernel** — match against compute-bound (gemm, attention, conv…) or memory-bound (norm, softmax, MoE dispatch…) patterns; unknown kernel defaults to `gpu_compute`

GPU **idle gaps** are not an event — they are the wall-clock gap between
consecutive GPU events on the same stream.  Detected when events have
`start_us`/`end_us` fields.  Set `MOE_BREAKDOWN_FULL_EVENTS=1` for the
transformers / tiny backends to get them.

## Tests

```bash
python tests/test_categorize.py
```

14 tests covering every bucket plus the gap detector.

## File layout

```
moe-breakdown/
├── src/moe_breakdown/
│   ├── categorize.py          # the 9-bucket regex rules + gap detector
│   ├── chart.py               # 3-panel chart
│   ├── report.py              # JSON + CSV writer
│   ├── models/
│   │   └── tiny_moe.py        # the in-tree 3.7 M-param MoE
│   └── backends/
│       ├── tiny.py            # tiny backend
│       ├── transformers.py    # HF transformers backend
│       ├── vllm.py            # vLLM backend (remote or in-process)
│       └── synthetic.py       # synthetic events
├── configs/
│   ├── tiny.yaml
│   ├── mixtral-8x7b.yaml
│   ├── qwen-moe.yaml
│   └── deepseek-moe.yaml
├── scripts/run_breakdown.py   # CLI entrypoint
├── examples/
│   ├── run_slurm.sbatch       # Slurm batch script
│   ├── run_quantized_qwen.py  # quantized Qwen template
│   └── run_my_own_model.py    # 5-line "drop in your model" template
├── tests/test_categorize.py
├── runs/                      # all run output goes here (gitignored)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## License

MIT
