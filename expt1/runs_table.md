# Run Inventory — routing-modes-allgather-h100

All results on PACE ICE H100, stored under `/storage/ice1/0/2/sghose7/moe-breakdown-runs/`.

## Output path bug (pre-fix)

Jobs submitted before the `OUT_DIR_BASE→OUT_DIR` fix all wrote to the same flat
directory `{study_name}/tp{tp}-ep{ep}/`, so successive runs for the same parallel
point accumulated (and partially overwrote) each other's results. The fix adds:

```bash
if [[ -z "$OUT_DIR" && -n "${OUT_DIR_BASE:-}" ]]; then
  OUT_DIR="${OUT_DIR_BASE}/${SLURM_JOB_ID}_tp${TP_SIZE}-ep${EP_SIZE}"
fi
```

Jobs submitted after this fix write to `{study_name}/{job_id}_tp{tp}-ep{ep}/`.

---

## routing-modes-allgather-h100

| Job ID | Parallel | Tokens | Routing modes | Config | Output dir | Status | Notes |
|--------|----------|--------|---------------|--------|------------|--------|-------|
| 5430690 | tp1-ep1 | 64–4096 | all 6 | routing-modes.allgather (old) | tp1-ep1/ (flat) | FAILED | Old timing code (pre-fix), failed with vllm API error |
| 5430691 | tp1-ep4 | 64–4096 | all 6 | routing-modes.allgather (old) | tp1-ep4/ (flat) | FAILED | Old timing code, failed |
| 5430708 | tp1-ep1 | 64–4096 | all 6 | routing-modes.allgather | tp1-ep1/ (flat) | FAILED | MoEActivation str bug |
| 5430709 | tp1-ep4 | 64–4096 | all 6 | routing-modes.allgather | tp1-ep4/ (flat) | FAILED | MoEActivation str bug |
| 5430716 | tp1-ep1 | 64–4096 | all 6 | routing-modes.allgather | tp1-ep1/ (flat) | FAILED | WorkspaceManager not init |
| 5430717 | tp1-ep4 | 64–4096 | all 6 | routing-modes.allgather | tp1-ep4/ (flat) | FAILED | WorkspaceManager not init |
| 5430719 | tp1-ep1 | 64–4096 | all 6 | routing-modes.allgather | tp1-ep1/ (flat) | FAILED | TritonExperts empty kwargs |
| 5430720 | tp1-ep4 | 64–4096 | all 6 | routing-modes.allgather | tp1-ep4/ (flat) | FAILED | TritonExperts empty kwargs |
| 5430722 | tp1-ep1 | 64–4096 | all 6 | routing-modes.allgather | tp1-ep1/ (flat) | FAILED | sp_local_sizes missing |
| 5430723 | tp1-ep4 | 64–4096 | all 6 | routing-modes.allgather | tp1-ep4/ (flat) | FAILED | sp_local_sizes missing |
| **5430733** | **tp1-ep1** | **64–20480** | **all 6** | routing-modes.allgather | tp1-ep1/ (flat) | **COMPLETED** | First clean run, 120 conditions |
| 5430734 | tp1-ep4 | 64–20480 | all 6 | routing-modes.allgather | tp1-ep4/ (flat) | FAILED | sp_local_sizes ctx generator bug |
| **5430736** | **tp1-ep4** | **64–20480** | **all 6** | routing-modes.allgather | tp1-ep4/ (flat) | **COMPLETED** | First clean ep4 run, 120 conditions |
| 5430759 | tp1-ep1 | 32768–131072 | all 6 | routing-modes.allgather.tok-ext | tp1-ep1/ (flat) | CANCELLED | Cancelled — would OOM at 131072 tokens; partial results (~23 rows) written before cancel |
| 5430760 | tp1-ep4 | 32768–131072 | all 6 | routing-modes.allgather.tok-ext | tp1-ep4/ (flat) | FAILED | OOM at 131072 + sp_local_sizes generator bug |
| **5430769** | **tp1-ep1** | **32768–65536** | **all 6** | routing-modes.allgather.tok-ext | tp1-ep1/ (flat) | **COMPLETED** | 24 conditions, high-token extension |
| **5430770** | **tp1-ep4** | **32768–65536** | **all 6** | routing-modes.allgather.tok-ext | tp1-ep4/ (flat) | **COMPLETED** | 24 conditions, high-token extension |
| **5430783** | **tp2-ep2** | **64–65536** | **all 6** | routing-modes.allgather | tp2-ep2/ (flat) | **COMPLETED** | 144 conditions, full token range |

### Current state of tp1-ep1 flat directory (pre-clean-rerun)
167 rows: 120 (5430733, tokens 64–20480) + 24 (5430769, tokens 32768–65536) + ~23 partial rows from cancelled 5430759 (tokens include 131072). **Contains 131072-token rows that should be excluded.**

### Current state of tp1-ep4 flat directory (pre-clean-rerun)
146 rows: 120 (5430736, tokens 64–20480) + 24 (5430770, tokens 32768–65536) + 2 extra. Clean — no 131072 contamination.

---

## routing-modes-deepep-h100

| Job ID | Parallel | Tokens | Routing modes | Config | Output dir | Status | Notes |
|--------|----------|--------|---------------|--------|------------|--------|-------|
| **5430771** | **tp1-ep1** | **64–65536** | **all 6** | routing-modes.deepep | tp1-ep1/ (flat) | **COMPLETED** | 156 conditions (13 tokens × 6 modes × 2 shapes) |

---

## fused-moe-characterization-pace-h100-deepep

| Job ID | Parallel | Tokens | Routing | Output dir | Status | Notes |
|--------|----------|--------|---------|------------|--------|-------|
| 5430528 | tp1-ep1 | 128–4096 | alpha sweep | tp1-ep1/ | COMPLETED | 66 rows, old timing code |
| 5430654 | tp1-ep4 | 128–4096 | alpha sweep | tp1-ep4/ | FAILED | NVSHMEM IBRC |
| 5430667 | tp1-ep2 | 128–4096 | alpha sweep | tp1-ep2/ | FAILED | NVSHMEM IBRC |
| 5430669 | tp2-ep2 | 128–4096 | alpha sweep | tp2-ep2/ | FAILED | NVSHMEM IBRC |
| 5430670 | tp2-ep4 | 128–4096 | alpha sweep | tp2-ep4/ | CANCELLED | Infeasible (8 GPUs) |

---

## Planned (Task 2 — clean reruns with fixed output path)

| Study | Parallel | Tokens | Output dir pattern | Why |
|-------|----------|--------|--------------------|-----|
| routing-modes-allgather-h100 | tp1-ep1 | 64–65536 | {job_id}_tp1-ep1/ | Clean rerun; removes 131072 contamination |
| routing-modes-allgather-h100 | tp1-ep4 | 64–65536 | {job_id}_tp1-ep4/ | Belt-and-suspenders clean copy |

---

## Planned (Task 3 — NCU profiling)

| Condition | Parallel | Tokens | Purpose |
|-----------|----------|--------|---------|
| mixtral_like / uniform | tp1-ep1 | 4096 | Baseline compute: DRAM BW, SM occupancy |
| mixtral_like / uniform | tp1-ep4 | 4096 | Allgather overhead vs compute |
| deepseek_like / zipfian | tp1-ep4 | 4096 | Most expensive realistic routing |
| deepseek_like / worst-case | tp1-ep4 | 4096 | Load imbalance mechanism |
| mixtral_like / worst-case | tp1-ep1 | 4096 | Cache-warm fast path |
