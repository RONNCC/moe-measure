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

---

## expt2 — NCCL Transport Condition Degradation Study

**Goal:** Quantify how much fused-MoE latency depends on fast intra-node GPU P2P communication (NVSwitch/NVLink) vs. PCIe fallback, by intentionally degrading NCCL transport and measuring latency across decode-like and prefill-like token regimes.

**Hardware:** PACE ICE H100 SXM5, single-node, 4 GPUs/node

**Shape:** Qwen3-30B-A3B (hidden=2048, inter=768, E=128, topk=8, bf16)

**Transport conditions (3 NCCL configs):**
- `nvlink_default` — NVLink/NVSwitch at full bandwidth (~900 GB/s), baseline
- `no_nvls_no_p2p` — `NCCL_NVLS_ENABLE=0 NCCL_P2P_DISABLE=1` (PCIe fallback)
- `no_nvls_no_p2p_1ch` — above + `NCCL_MAX_NCHANNELS=1` (single-channel PCIe)

**Parallel points:** tp1-ep1 (control, no inter-GPU comm), tp1-ep2, tp1-ep4, tp2-ep2

**Token range:** 1–8192 (11 values spanning decode and prefill regimes)

**Routing:** uniform (fixed for isolation)

**Total jobs:** 12 SLURM jobs (3 transport × 4 parallel points)

**Key question:** Does the slowdown ratio peak in the moderate-token regime (where allgather volume is significant but GEMM hasn't gone compute-bound) and return to ~1.0 at large tokens?

**Results:** `expt2/all_runs.zip` (after jobs complete)

**Code:** `expt1/src/fused_moe_kernel_study/` (shared library; `transport_condition` field added)

**Configs:** `expt2/configs/study.transport-conditions.qwen3.yaml`

**Cluster storage:** `/storage/ice1/0/2/sghose7/moe-breakdown-runs/expt2/`
