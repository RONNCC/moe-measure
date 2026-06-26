# Experiment Catalog

## expt1 — Fused MoE Kernel Latency Characterization (Allgather + DeepEP)

**Goal:** Characterize fused MoE kernel latency as a function of routing distribution, token count, TP/EP parallelism, and all2all backend by calling `FusedMoEModularKernel.apply()` directly with synthetic inputs, bypassing the vLLM serving stack.

**Hardware:** PACE ICE H100 SXM5 (NVLink + InfiniBand HDR), single-node jobs

**Backends measured:**
- `allgather_reducescatter` — standard NCCL allgather dispatch (tp1-ep1, tp1-ep4, tp2-ep2)
- `deepep_low_latency` — DeepEP one-sided RDMA dispatch (tp1-ep1 only; multi-GPU blocked by NVSHMEM IBRC on PACE ICE)

**Routing distributions:** uniform, zipfian, random, skewed-2x, skewed-4x, worst-case

**Token range:** 64–65536

**Key finding:** Even over NVSwitch (best-case intra-node interconnect), allgather dispatch dominates latency at realistic token counts (>4096). At 65536 tokens, tp1-ep4 is ~16× slower than tp1-ep1.

**Results:** `expt1/all_runs.zip` — 33 result CSVs across all studies

**Code:** `expt1/src/fused_moe_kernel_study/`

**Configs:** `expt1/configs/`

**Cluster storage:** `/storage/ice1/0/2/sghose7/moe-breakdown-runs/`
