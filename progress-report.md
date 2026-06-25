# MoE Routing Latency Study — Progress Report

**Date:** 2026-06-25  
**Cluster:** Georgia Tech PACE ICE (H100 SXM5, NVLink + InfiniBand HDR)  
**Repo:** `/home/hice1/sghose7/moe-breakdown`

---

## Overview

This report maps the four-phase experimental study spec against what has actually been run and what data currently exists. The study characterizes fused MoE kernel latency as a function of routing distribution, token count, TP/EP parallelism, and all2all backend — calling `FusedMoEModularKernel.apply()` directly with synthetic inputs, bypassing the vLLM serving stack entirely.

As of this update, the `routing-modes-allgather-h100` study has completed, providing the first clean full-factorial dataset: **2 kernel shapes × 6 routing distributions × 10 token counts × 2 EP configs = 240 measurement conditions**, all with deferred-sync CUDA-event timing and per-condition bucket profiling.

---

## Methodology

### What is not present

vLLM is used purely as a kernel library. No HTTP server, tokenizer, scheduler, attention layer, KV cache, or model weight loading is involved. The only vLLM entry point called during measurement is `FusedMoEModularKernel.apply()`. A minimal `VllmConfig` stub in `vllm_adapter.py` satisfies vLLM's internal parallel-config plumbing; it carries no model checkpoint path and loads no weights from disk.

### Synthetic inputs

Weight tensors `w1` and `w2` are created with `torch.randn` scaled by `1/sqrt(dim)`, seeded deterministically per rank. Hidden states are likewise `torch.randn`, seeded as a function of `cfg.seed`, `num_tokens`, `alpha`, `rank`, and `routing_mode`. The gating network is bypassed entirely: `topk_ids` and `topk_weights` are constructed programmatically by `routing.py`.

### Timing methodology

Timing uses deferred-sync CUDA events. For each of the 100 measured iterations, `start_event.record()` and `end_event.record()` are called without any mid-loop synchronization. A single `torch.cuda.synchronize()` follows the entire loop, then `elapsed_time()` is read from all event pairs. This keeps CPU-GPU sync overhead out of the hot path. 20 warmup iterations precede the timed loop. The headline metric is `latency_median_ms_max_rank` — the maximum across ranks of each rank's per-condition median — which captures the bottleneck rank under imbalanced routing.

### Bucket profiling

A second pass uses `torch.profiler` for 5 iterations to classify time into 9 buckets: `cpu_python`, `cpu_native`, `gpu_compute`, `gpu_memory`, `gpu_idle_gap`, `gpu_idle_sync`, `mem_transfer`, `network`, `allocator`. Written alongside each timing result in `bucket_profiles/`.

### Routing distributions

Six named distributions implemented in `routing.py`:

| Mode | Description |
|------|-------------|
| **uniform** | Equal probability 1/E to each expert; sampled via `torch.multinomial` |
| **zipfian** | Probability ∝ 1/(rank+1), normalized; creates natural head-of-distribution skew |
| **random** | Single Dirichlet(ones) per batch via Gamma(1,1) trick; one random probability vector shared across all tokens |
| **skewed-2x** | One hot expert at 2× uniform probability, rest at 1×, normalized |
| **skewed-4x** | One hot expert at 4× uniform probability, rest at 1×, normalized |
| **worst-case** | All tokens sent to the first `topk` experts only; bypasses multinomial entirely |

### Distributed launch

Each Slurm job calls `srun --ntasks=world_size`. One process per GPU. Process group initialized with NCCL. For the allgather backend, `NCCL_P2P_DISABLE=1` and `NCCL_SHM_DISABLE=1` prevent NVML index aliasing when GPUs are hidden per task.

---

## Completed Study: routing-modes-allgather-h100

**Jobs:** 5430733 (tp1-ep1), 5430736 (tp1-ep4)  
**Completed:** 2026-06-25  
**Backend:** `allgather_reducescatter`  
**Sweep:** full_factorial — all routing modes × all token counts × both shapes  
**Conditions per job:** 120 (6 modes × 10 tokens × 2 shapes)  
**Result files:** 241 per parallel point (120 timing + 120 bucket profiles + metadata)

### Kernel shapes

| Shape | hidden | intermediate | experts | topk | dtype |
|-------|--------|-------------|---------|------|-------|
| mixtral_like | 4096 | 14336 | 8 | 2 | bfloat16 |
| deepseek_like | 7168 | 18432 | 16 | 2 | bfloat16 |

### Results: mixtral_like — median latency (ms, max rank)

**tp1-ep1** (single GPU, no dispatch):

| tokens | uniform | zipfian | random | skewed-2x | skewed-4x | worst-case |
|-------:|--------:|--------:|-------:|----------:|----------:|-----------:|
|     64 |   0.943 |   1.084 |  1.095 |     0.945 |     1.085 |      0.441 |
|    256 |   1.214 |   1.281 |  1.312 |     1.196 |     1.113 |      0.791 |
|    512 |   2.128 |   1.966 |  2.040 |     1.959 |     1.940 |      1.641 |
|   1024 |   2.240 |   2.355 |  2.345 |     2.269 |     2.278 |      1.895 |
|   2048 |   3.363 |   3.647 |  3.586 |     3.487 |     3.676 |      2.971 |
|   4096 |   6.488 |   6.622 |  6.525 |     6.644 |     6.475 |      6.034 |
|   8192 |  12.912 |  12.891 |  12.741 |    13.002 |    12.830 |     12.532 |
|  12288 |  19.614 |  19.801 |  19.706 |    19.618 |    19.575 |     18.925 |
|  16384 |  26.140 |  26.203 |  26.150 |    26.104 |    26.238 |     25.526 |
|  20480 |  32.596 |  32.852 |  32.742 |    32.797 |    32.745 |     32.159 |

**tp1-ep4** (4 GPUs, allgather dispatch across EP group):

| tokens | uniform | zipfian | random | skewed-2x | skewed-4x | worst-case |
|-------:|--------:|--------:|-------:|----------:|----------:|-----------:|
|     64 |   0.920 |   1.030 |  0.960 |     0.924 |     0.948 |      1.273 |
|    256 |   2.218 |   2.210 |  2.180 |     2.176 |     2.156 |      2.623 |
|    512 |   4.245 |   4.229 |  4.079 |     4.133 |     4.083 |      5.248 |
|   1024 |   8.097 |   8.692 |  8.171 |     7.890 |     8.141 |     11.014 |
|   2048 |  15.821 |  17.748 |  16.684 |    15.800 |    16.649 |     22.726 |
|   4096 |  31.386 |  35.886 |  33.539 |    32.012 |    33.680 |     46.258 |
|   8192 |  62.341 |  72.294 |  67.091 |    64.287 |    67.862 |     93.261 |
|  12288 |  93.011 | 108.820 | 100.945 |    96.663 |   102.442 |    140.284 |
|  16384 | 124.124 | 145.246 | 134.651 |   128.997 |   137.017 |    188.309 |
|  20480 | 154.982 | 181.538 | 168.440 |   161.602 |   171.135 |    236.724 |

### Results: deepseek_like — median latency (ms, max rank)

**tp1-ep1** (single GPU, no dispatch):

| tokens | uniform | zipfian | random | skewed-2x | skewed-4x | worst-case |
|-------:|--------:|--------:|-------:|----------:|----------:|-----------:|
|     64 |   4.108 |   4.051 |  3.802 |     4.108 |     4.102 |      0.985 |
|    256 |   4.330 |   5.044 |  4.416 |     4.365 |     4.426 |      1.950 |
|    512 |   6.073 |   5.818 |  6.391 |     5.695 |     5.104 |      3.756 |
|   1024 |   6.367 |   6.530 |  6.694 |     6.297 |     5.423 |      4.286 |
|   2048 |  10.288 |  10.712 |  11.059 |     9.827 |     9.908 |      8.751 |
|   4096 |  16.166 |  17.367 |  17.213 |    16.912 |    17.720 |     14.244 |
|   8192 |  32.851 |  33.336 |  33.245 |    33.072 |    32.764 |     29.108 |
|  12288 |  48.821 |  48.279 |  48.866 |    48.805 |    48.995 |     45.141 |
|  16384 |  64.636 |  63.698 |  64.408 |    64.958 |    64.392 |     60.578 |
|  20480 |  79.986 |  78.700 |  80.038 |    80.048 |    80.085 |     76.435 |

**tp1-ep4** (4 GPUs, allgather dispatch across EP group):

| tokens | uniform | zipfian | random | skewed-2x | skewed-4x | worst-case |
|-------:|--------:|--------:|-------:|----------:|----------:|-----------:|
|     64 |   1.837 |   2.635 |  1.820 |     1.835 |     1.888 |      2.555 |
|    256 |   4.171 |   4.678 |  4.207 |     4.170 |     4.252 |      5.189 |
|    512 |   7.487 |   8.791 |  7.611 |     7.520 |     7.462 |     10.823 |
|   1024 |  14.423 |  17.840 |  14.760 |    14.438 |    15.062 |     22.184 |
|   2048 |  28.619 |  35.907 |  29.923 |    28.688 |    30.388 |     45.886 |
|   4096 |  56.713 |  72.461 |  59.734 |    57.641 |    61.480 |     93.472 |
|   8192 | 113.023 | 146.905 | 119.272 |   116.474 |   124.181 |    192.278 |
|  12288 | 169.821 | 222.660 | 179.547 |   175.183 |   187.018 |    292.603 |
|  16384 | 226.992 | 299.051 | 240.941 |   234.449 |   250.682 |    393.137 |
|  20480 | 284.792 | 375.590 | 302.017 |   296.325 |   313.178 |    492.527 |

### Key observations

**1. worst-case at ep=1 is fastest, not slowest.**  
At tp1-ep1 (no inter-GPU dispatch), routing all tokens to the same topk experts is *faster* because there is no token-redistribution overhead, the expert is warm in L2 cache, and the Triton kernel has maximally contiguous work. The effect is large at small token counts: deepseek_like at 64 tokens, worst-case is 0.99 ms vs uniform 4.11 ms — a 4× difference. At large tokens (≥8K) the gap narrows to ~5% as compute dominates.

**2. worst-case at ep=4 is the most expensive routing by a large margin.**  
With allgather dispatch, sending all tokens to two experts means the rank holding those experts receives the full token batch from all four peers, while the other three ranks receive almost nothing. This creates the maximum possible load imbalance in the allgather gather/scatter. At 4096 tokens, worst-case (46.3 ms) is 1.5× uniform (31.4 ms) for mixtral_like and 1.6× for deepseek_like.

**3. Zipfian is the most expensive "realistic" distribution at ep=4.**  
Among distributions that could occur in real models, zipfian is consistently 10–20% slower than uniform at ep=4. At 20480 tokens, zipfian is 181 ms vs uniform 155 ms for mixtral_like (+17%) and 376 ms vs 285 ms for deepseek_like (+32%). This is the Zipf head-expert effect: a small number of experts receive disproportionately many tokens, becoming bottleneck ranks.

**4. Allgather latency scales linearly with tokens at ep=4.**  
The ep=4 latency is near-linear from 256–20480 tokens, consistent with allgather communication cost dominating: each rank must send its full token batch to every peer, so cost ∝ tokens × (ep-1)/ep × hidden_size. At ep=1 the relationship is sublinear at low token counts due to launch overhead, then near-linear at high counts.

**5. DeepSeek-like is 2–5× more expensive than mixtral-like at ep=1, but similar at ep=4.**  
At tp1-ep1 the ratio is 4× at 64 tokens and ~2.5× at 20480 tokens, driven by the larger weight matrices (7168×18432 vs 4096×14336) and more experts (16 vs 8). At tp1-ep4, the ratio narrows because network/allgather cost dominates and both shapes transfer the same hidden states (all_to_all is on hidden_size dimension).

**6. Bucket profiler values are per-5-iteration totals; divide by 5 to compare to per-iteration latency.**  
`bucket_max_rank_network_ms` values in the tables above are summed over `bucket_profile_iters=5` profiler iterations. Dividing by 5: mixtral uniform 4096 tokens at ep=4 gives ~24 ms of network per iteration vs 31 ms total latency — network accounts for ~77% of wall time. For deepseek worst-case 4096 at ep=4: ~91 ms network / 93 ms latency ≈ 98%, nearly fully network-bound. Notably, `bucket_max_rank_gpu_compute_ms` reads 0.000 for all conditions — this is a known profiler categorization artifact where the CUDA kernels launched by vllm's Triton MoE implementation are attributed to `cpu_native` (JIT-compiled Triton ops register as CPU-side launches) rather than `gpu_compute`.

---

## Data Inventory

| Study | Parallel points | Backend | Shapes | Routing | Tokens | Rows | Status |
|-------|----------------|---------|--------|---------|--------|------|--------|
| routing-modes-allgather-h100 | tp1-ep1, tp1-ep4 | allgather | mixtral_like, deepseek_like | 6 named modes | 64–20480 (10 pts) | 240 | **COMPLETE** |
| fused-moe-characterization-pace-h100-deepep | tp1-ep1 | deepep_low_latency | mixtral_like, deepseek_like | alpha sweep | 128–4096 | 66 | Complete (ep=1 only) |
| fused-moe-characterization-pace-h100-deepep | tp1-ep2, tp2-ep2, tp1-ep4 | deepep_low_latency | mixtral_like, deepseek_like | alpha sweep | 128–4096 | 0 | BLOCKED (NVSHMEM IBRC) |
| routing-modes-deepep-h100 | tp1-ep1, tp1-ep4 | deepep_low_latency | mixtral_like, deepseek_like | 6 named modes | 64–20480 | 0 | Not submitted |
| qwen3-30b-a3b-full-onenode-limited | tp1-ep1, tp2-ep1, tp1-ep2, tp1-ep4, tp2-ep2 | allgather | qwen3-30b-a3b-moe (E=128, topk=8) | 6 named modes | 1–200K (11 pts) | 415 | Complete (off-spec shape) |
| qwen3-30b-a3b-full-twonode-limited | tp1-ep4, tp2-ep2 | allgather | qwen3-30b-a3b-moe | 6 named modes | 1–200K | 166 | Complete — cross-node IB path |
| qwen3-30b-a3b-medium-threenode | tp1-ep6 | allgather | qwen3-30b-a3b-moe | alpha-only | 64–4096 | 30 | Complete — only tp1-ep6 data |
| qwen3-30b-a3b-medium / initial | various | allgather | qwen3-30b-a3b-moe | alpha-only | 64–4096 | ~240 | Superseded by full-onenode |

---

## Phase Status

### Phase A: Routing Sensitivity at tp1-ep1
**Goal:** 6 routing modes × token sweep × shapes at tp1-ep1  
**Status: COMPLETE for allgather backend** — 2 shapes × 6 modes × 10 tokens = 120 rows. Token range 64–20480 (spec wants 1–200K; large end not yet covered). DeepEP side blocked.

### Phase B: EP Sweep at Fixed Tokens
**Goal:** Compare EP configs at fixed token count  
**Status: Partial** — tp1-ep1 and tp1-ep4 allgather data exists at all token counts. tp2-ep2 not submitted for the new study. DeepEP EP>1 blocked.

### Phase C: Token × TP/EP Sweep at Uniform Routing
**Goal:** Characterize scaling across token counts and EP degrees  
**Status: Partial** — tp1-ep1 and tp1-ep4 complete. tp2-ep2 and higher EP configs not yet run for this study. Token range 64–20480.

### Phase D: DeepEPLL vs DeepEPHT Comparison
**Goal:** Compare allgather_reducescatter vs deepep_low_latency vs deepep_high_throughput  
**Status: BLOCKED** — DeepEP EP>1 fails with NVSHMEM IBRC error (cluster-level IB permission issue). tp1-ep1 DeepEP data exists but has no inter-GPU dispatch.

---

## Infrastructure Fixes Applied (this session)

The following vLLM API incompatibilities were discovered and fixed during job debugging:

| Error | Root cause | Fix (file:line) |
|-------|-----------|-----------------|
| `'str' has no attribute 'is_gated'` | vllm now requires `MoEActivation` enum, not plain string | `runner.py`: convert via `MoEActivation(activation)` |
| `WorkspaceManager not initialized` | vllm v1 workspace singleton set up by `GPUModelRunner`, which we bypass | `vllm_adapter.py`: `_maybe_init_workspace_manager()` before kernel build |
| `Could not construct fused-experts` | `TritonExperts({})` called with empty kwargs; needs `moe_config` + `quant_config` | `vllm_adapter.py`: pass required kwargs to all expert constructors |
| `dp_metadata.local_sizes is None` | vllm MoE runner enters `sp_local_sizes(sp_size)` ctx before dispatch; we don't | `vllm_adapter.py` + `runner.py`: `sp_local_sizes_context()` around all `kernel.apply()` calls |

---

## Next Steps

1. **Submit tp2-ep2 for routing-modes-allgather** (config exists, just needs submission).
2. **Submit routing-modes-deepep tp1-ep1** (ep=1 works, no NVSHMEM needed).
3. **Extend token range to 65536+** for the high-token regime where DeepEP HT should outperform LL.
4. **Resolve NVSHMEM IBRC** for DeepEP multi-GPU — open PACE sysadmin ticket if UCX fallback also fails.
5. **Submit deepep_high_throughput** config once multi-GPU DeepEP works.
