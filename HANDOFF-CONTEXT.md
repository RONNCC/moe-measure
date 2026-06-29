# Data Handoff Context — expt2 + expt2.5

This document gives a complete briefing for combining expt2 and expt2.5 into a report.

---

## Hardware

**Both expt2 and expt2.5 ran on the same hardware:**
- PACE ICE H200 SXM5, single node, 4 GPUs
- Nodes: atl1-1-03-017, atl1-1-03-018
- (expt2 was originally submitted for H100 nodes, but all H100 nodes had broken
  multi-GPU NCCL transport, so it was rerouted to H200. The H200 fix is in git.)

Direct latency comparisons across expt2 and expt2.5 are valid — same hardware.

---

## Data Files

| File | Rows | Description |
|------|------|-------------|
| `expt2/all_runs.zip` | 132 | Transport ablation baseline study |
| `expt2.5/all_runs.zip` | 1120 | Extended transport + routing study |

Both zips contain `results.csv` files with **identical schemas** (56 columns).
They can be concatenated directly with `pd.read_csv`.

---

## Schema Key Columns

| Column | Meaning |
|--------|---------|
| `study_name` | Which study (e.g. `transport-conditions-qwen3`, `transport-extended-qwen3`, `routing-sweep-qwen3`) |
| `transport_condition` | NCCL config name (see below) |
| `routing_mode` | `uniform`, `zipfian`, `random`, `skewed-2x`, `skewed-4x`, `worst-case` |
| `tp`, `ep` | Tensor parallelism / Expert parallelism degree |
| `tokens` | Number of tokens in the batch |
| `latency_median_ms_max_rank` | **Primary latency metric** — median over 100 iterations, worst rank |
| `latency_p95_ms_max_rank` | P95 latency (worst rank) |
| `bucket_max_rank_network_ms` | Time spent in network (NCCL allgather/reduce-scatter) |
| `bucket_max_rank_gpu_compute_ms` | Time spent in GPU compute (expert GEMMs) |
| `allgather_recv_bytes` | Bytes received in allgather (use with network_ms for achieved BW) |
| `alpha_observed` | Actual routing imbalance ratio achieved |

Achieved bandwidth (GB/s) = `allgather_recv_bytes / (bucket_max_rank_network_ms * 1e-3 * 1e9)`

---

## What Each Study Covers

### expt2 — `transport-conditions-qwen3` (132 rows)
- **Purpose:** Initial transport ablation; does PCIe fallback hurt at decode vs prefill scale?
- Transports: `nvlink_default`, `no_nvls_no_p2p`, `no_nvls_no_p2p_1ch`
- Routing: `uniform` only
- Tokens: 1, 2, 4, 8, 16, 32, 64, 128, 512, 2048, 8192 (11 values, decode→prefill)
- Parallel points: tp1-ep1, tp1-ep2, tp1-ep4, tp2-ep2

### expt2.5 Study A — `transport-extended-qwen3` (448 rows)
- **Purpose:** Extended transport ablation with finer NCCL knob resolution + large tokens
- Transports: `nvlink_default`, `nvls_off`, `p2p_off`, `no_nvls_no_p2p`,
  `no_nvls_no_p2p_8ch`, `no_nvls_no_p2p_4ch`, `no_nvls_no_p2p_2ch`, `no_nvls_no_p2p_1ch`
- Routing: `uniform` only
- Tokens: 1–65536 (14 values; adds 16384, 32768, 65536 beyond expt2)
- Parallel points: same 4

### expt2.5 Study B — `routing-sweep-qwen3` (672 rows)
- **Purpose:** Does routing imbalance interact with transport degradation?
- Transports: `nvlink_default`, `no_nvls_no_p2p` (two extremes only)
- Routing: `uniform`, `zipfian`, `random`, `skewed-2x`, `skewed-4x`, `worst-case`
- Tokens: 1–65536 (14 values)
- Parallel points: same 4

---

## Transport Condition Glossary

| Condition | NCCL env vars | What it tests |
|-----------|--------------|--------------|
| `nvlink_default` | (none) | NVLink/NVSwitch at full speed — best case |
| `nvls_off` | `NCCL_NVLS_ENABLE=0` | NVLink-SHARP (NVLS) disabled, P2P-IPC still on |
| `p2p_off` | `NCCL_P2P_DISABLE=1` | P2P-IPC disabled, NVLS still on |
| `no_nvls_no_p2p` | both above | Full PCIe fallback — worst case |
| `no_nvls_no_p2p_8ch` | above + `NCCL_MAX_NCHANNELS=8` | PCIe with 8 channels |
| `no_nvls_no_p2p_4ch` | above + `NCCL_MAX_NCHANNELS=4` | PCIe with 4 channels |
| `no_nvls_no_p2p_2ch` | above + `NCCL_MAX_NCHANNELS=2` | PCIe with 2 channels |
| `no_nvls_no_p2p_1ch` | above + `NCCL_MAX_NCHANNELS=1` | PCIe with 1 channel — bandwidth floor |

**Note:** `no_nvls_no_p2p` (no channel constraint) appears in expt2, expt2.5-A, and
expt2.5-B — it is the common denominator across all three studies.

---

## Overlapping Conditions

expt2 and expt2.5-A share:
- Same shape, hardware, parallel points, routing (uniform)
- Both have `nvlink_default` and `no_nvls_no_p2p` (and `no_nvls_no_p2p_1ch`)
- Token overlap: 1–8192 (all 11 expt2 tokens appear in expt2.5's 14-token set)

expt2.5-A and expt2.5-B share:
- Same shape, hardware, parallel points
- Both have `nvlink_default` and `no_nvls_no_p2p`
- Same 14-token set; expt2.5-B adds 5 non-uniform routing modes

---

## Model Shape (all studies)

Qwen3-30B-A3B MoE layer:
- `hidden_size`: 2048
- `intermediate_size`: 768
- `num_experts`: 128
- `topk`: 8
- `dtype`: bfloat16
- `backend`: allgather_reducescatter (standard NCCL, not DeepEP)

---

## Known Data Quality Notes

1. **No missing data:** All 1252 rows (132 + 1120) have valid latency values.

2. **tp1-ep1 is the control:** At ep=1, there is no inter-GPU expert dispatch, so
   `bucket_max_rank_network_ms` ≈ 0 regardless of transport condition. Use this to
   isolate pure compute cost vs. communication cost.

3. **Routing imbalance at tp1-ep1 vs ep>1:** With ep=1 all tokens stay on-device,
   so routing mode should not affect latency at tp1-ep1. Cross-checking this is a
   useful sanity check on the data.

4. **The `no_nvls_no_p2p_1ch` in expt2 vs expt2.5:** These are the same condition
   run in two separate experiments. They can be compared for reproducibility.
