# MoE Kernel Benchmark Experiment Audit

**Audit date:** 2026-06-24
**Auditor:** automated cross-reference of local configs + git history
**SSH note:** The PACE ICE login node (`login-ice.pace.gatech.edu`) was unreachable during
this audit session, so result file counts, `squeue --me` output, and the
`routing_mode` field check in live JSON files could not be gathered. Fields
marked [CLUSTER-UNVERIFIED] require a follow-up check on the cluster. All
config-level and code-level findings are fully verified from local sources.

---

## 1. Timing Code History

The deferred-sync fix is in the **working tree only** — it has never been committed.

| State | Pattern | Status |
|-------|---------|--------|
| Committed (HEAD = `71d63f0`) | `end_event.synchronize()` inside measure loop + `timings_ms.append()` per iter | OLD / BUGGY |
| Working tree (uncommitted diff) | `event_pairs` list, single `torch.cuda.synchronize()` after loop | NEW / FIXED |

**Consequence:** Every job submitted from the committed code — including all runs in
`/storage/ice1/0/2/sghose7/moe-breakdown-runs/` — used the OLD timing code.
The per-iteration `end_event.synchronize()` stalls the CPU after every kernel
call, inflating measured latency by the CPU-GPU round-trip overhead (typically
5–50 µs per iteration on H100, more under load). Relative comparisons within the
same study are still internally consistent, but absolute latency numbers are
systematically biased upward.

**The timing fix also bundles two other uncommitted changes:**
1. `routing_mode` field added to result JSON (previously absent from `run_condition` output).
2. `routing_mode` passed to `make_routing_batch` (previously the routing_mode arg existed in routing.py but was not wired through from `run_condition`).
3. `hidden_gen` seed now includes `hash(cond.routing_mode)` so different routing modes get different input tensors.

This means: any routing-modes study run from committed code produced results
where **all routing modes used identical input hidden states** (same seed, no
routing_mode term), and the `routing_mode` field is absent from result JSONs.

---

## 2. Config Inventory

Six config files exist locally. Two are new (untracked, never submitted):

| File | Status |
|------|--------|
| `study.example.yaml` | Reference only — not submitted |
| `study.pace.a100.yaml` | Committed (8856fcb, 2026-06-24 11:39) |
| `study.pace.h100.deepep.yaml` | **Untracked** (new, not yet submitted) |
| `study.qwen3_30b_a3b.initial.yaml` | Committed (8856fcb, 2026-06-24 11:39) |
| `study.routing-modes.allgather.yaml` | **Untracked** (new, not yet submitted) |
| `study.routing-modes.deepep.yaml` | **Untracked** (new, not yet submitted) |

---

## 3. Study-by-Study Audit

### Study A: `fused-moe-characterization-pace-h100-deepep`
**Config file:** `configs/study.pace.h100.deepep.yaml`

| Dimension | Config value | Spec requirement | Verdict |
|-----------|-------------|-----------------|---------|
| Backend | `deepep_low_latency` | deepep_low_latency OR deepep_high_throughput | OK for Phase D LL side |
| Shapes | mixtral_like (4096/14336/E8/topk2/bf16), deepseek_like (7168/18432/E16/topk2/bf16) | Same | OK |
| Token range | 128–4096 (6 values) | 1–200000 (11 values) | INCOMPLETE — missing tokens 1,4,16,64 (small) and 16384,65536,131072,200000 (large) |
| Routing parameterization | Alpha sweep: alphas=[1.0,2.0,4.0,8.0] | Named routing modes for Phase A | WRONG for Phase A; Phase B/C/D use alpha-style, so this config is a Phase B/C/D instrument, not Phase A |
| Parallel points | tp1-ep1, tp1-ep2, tp2-ep2, tp1-ep4 (max world=4) | Max 4 GPUs single node | OK — all within constraint |
| EP/TP coverage vs spec | Missing tp2-ep1, tp1-ep8, tp4-ep2 | Phase B wants uniform+skewed-4x across all 7 EP configs | INCOMPLETE — 3 EP configs absent; tp1-ep8 and tp4-ep2 exceed 4-GPU limit anyway (BLOCKED by hardware) |
| Sweep modes | one_at_a_time + full_factorial | Both needed | OK |
| Expected conditions | 264 total (66 per parallel point × 4 points) | — | — |
| Results on cluster | [CLUSTER-UNVERIFIED] | — | — |
| Timing code | OLD (buggy) — all submitted jobs predate the working-tree fix | NEW required | NEEDS-RERUN |
| `routing_mode` field in JSON | Absent (not wired in committed runner.py) | Required for analysis | NEEDS-RERUN |

**Verdict: NEEDS-RERUN**
- The timing fix and `routing_mode` wiring must be committed before resubmission.
- Token range must be extended to cover at least 1–200K per spec.
- The missing EP configs (tp1-ep8, tp4-ep2 = 8 GPUs) are hardware-blocked on PACE ICE H100; confirm whether multi-node is permissible or those points must be dropped from scope.

---

### Study B: `routing-modes-allgather-h100`
**Config file:** `configs/study.routing-modes.allgather.yaml`

| Dimension | Config value | Spec requirement | Verdict |
|-----------|-------------|-----------------|---------|
| Backend | `allgather_reducescatter` | NCCL baseline (allgather) | OK |
| Shapes | mixtral_like, deepseek_like — correct dimensions | Same | OK |
| Token range | 64–4096 (6 values) | 1–200000 | INCOMPLETE — missing tokens 1,4,16 (small) and 16384–200000 (large) |
| Routing parameterization | Named modes: [uniform, zipfian, random, skewed-2x, skewed-4x, worst-case] | Same 6 named modes (Phase A) | OK |
| `alphas` field | Present as [1.0] (fallback) | N/A when routing_modes is set | OK — routing_modes takes precedence |
| Parallel points | tp1-ep1, tp1-ep4 (max world=4) | Phase A: tp1-ep1 required; tp1-ep4 is extra | OK — within constraint |
| Sweep modes | full_factorial only | Phase A uses full factorial | OK |
| gpus_per_node in slurm | 8 | PACE ICE H100 max 4 per node | PROBLEM — requests 8 GPUs for a 4-GPU job; this overallocates resources. The submit script computes `min(world_size, gpus_per_node)` so it will request `min(4, 8)=4` GPUs correctly. OK in practice, but the config value is misleading. |
| Expected conditions | 144 total (72 per parallel point × 2 points) | — | — |
| Results on cluster | [CLUSTER-UNVERIFIED] — config is untracked, likely never submitted | — | — |
| Timing code | Config is untracked; if submitted it would have used OLD committed runner.py | NEW required | NOT YET SUBMITTED (untracked) |
| `routing_mode` field in JSON | Would be absent with current committed runner.py | Required | Must commit fix before submitting |

**Verdict: NOT YET SUBMITTED — commit timing fix + routing wiring first, then expand token range, then submit**

---

### Study C: `routing-modes-deepep-h100`
**Config file:** `configs/study.routing-modes.deepep.yaml`

| Dimension | Config value | Spec requirement | Verdict |
|-----------|-------------|-----------------|---------|
| Backend | `deepep_low_latency` | deepep_low_latency for Phase A DeepEP side | OK |
| Shapes | mixtral_like, deepseek_like | Same | OK |
| Token range | 64–4096 (6 values) | 1–200000 | INCOMPLETE — same gaps as Study B |
| Routing parameterization | Named modes: [uniform, zipfian, random, skewed-2x, skewed-4x, worst-case] | Phase A named modes | OK |
| Parallel points | tp1-ep1, tp1-ep4 (max world=4) | Within constraint | OK |
| Sweep modes | full_factorial only | OK for Phase A | OK |
| deepep_wheel path | Hardcoded `/storage/ice1/0/2/sghose7/deepep-build/workspace/dist/deep_ep-*.whl` | Needs pre-built wheel | OK — matches build path |
| Expected conditions | 144 total | — | — |
| Results on cluster | [CLUSTER-UNVERIFIED] — config is untracked, likely never submitted | — | — |
| Timing code | NOT YET SUBMITTED — but would use OLD code if submitted now | NEW required | Must commit fix first |
| `routing_mode` field in JSON | Would be absent with committed runner.py | Required | Must commit fix before submitting |

**Verdict: NOT YET SUBMITTED — same blockers as Study B; deepep_wheel must exist before submitting**

---

### Study D: `fused-moe-characterization-pace-a100`
**Config file:** `configs/study.pace.a100.yaml`

| Dimension | Config value | Spec requirement | Verdict |
|-----------|-------------|-----------------|---------|
| Backend | `deepep_low_latency` | Spec targets H100; A100 study is off-spec hardware | WRONG HARDWARE for main spec — A100 is not H100 |
| Shapes | mixtral_like, deepseek_like | Same | OK |
| Token range | 128–4096 | 1–200000 | INCOMPLETE |
| Routing parameterization | Alpha sweep: alphas=[1.0,2.0,4.0,8.0] | Named modes for Phase A | WRONG for Phase A |
| Parallel points | tp1-ep1, tp1-ep2, tp1-ep4, **tp1-ep8**, tp2-ep2, **tp2-ep4** | Max 4 GPUs on PACE ICE H100 | WARNING — tp1-ep8 and tp2-ep4 request 8 GPUs. A100 nodes may allow 8 GPUs/node; verify. These points are valid for A100 but not for H100 spec compliance. |
| Divisibility | All shapes divisible by tp=1 or tp=2, E>=ep for all cases | OK | OK |
| GPU constraint | gpus_per_node=8, A100 hardware | Hardware-dependent — not the primary H100 study | ACCEPTABLE on A100 |
| Results on cluster | [CLUSTER-UNVERIFIED] | — | — |
| Timing code | Committed with OLD runner.py (8856fcb same commit as config) | NEW required | NEEDS-RERUN |
| `routing_mode` field | Absent in committed code | N/A for alpha sweep (routing_mode="alpha" default) | OK — alpha sweep doesn't need named routing_mode; field defaults to "alpha" |

**Verdict: NEEDS-RERUN (timing fix) + WRONG HARDWARE for primary spec**
- This is supplementary data on A100, not the H100 spec target.
- If A100 data is intentionally wanted: commit timing fix and rerun.
- tp1-ep8 and tp2-ep4 (8-GPU points) may be feasible on A100 but need to be removed if transferred to H100.

---

### Study E: `qwen3-30b-a3b-initial`
**Config file:** `configs/study.qwen3_30b_a3b.initial.yaml`

| Dimension | Config value | Spec requirement | Verdict |
|-----------|-------------|-----------------|---------|
| Backend | `deepep_low_latency` | Not in primary spec (different model shape) | OUT OF SPEC SCOPE — Qwen3 shape, not Mixtral/DeepSeek |
| Shapes | qwen3_30b_a3b (hidden=2048, inter=768, E=128, topk=8, bf16) | Spec uses mixtral/deepseek shapes | DIFFERENT SHAPE — this is an exploratory study, not part of the 4-phase spec |
| Token range | 128–1024 (3 values) | 1–200000 | Very incomplete |
| Routing parameterization | Alpha sweep: [1.0, 4.0] | N/A | Exploratory only |
| Parallel points | tp1-ep1, tp1-ep4 | Within 4-GPU limit | OK |
| E=128, ep=4 | 128/4=32 local experts | OK | OK |
| Results on cluster | [CLUSTER-UNVERIFIED] | — | — |
| Timing code | OLD (committed same as A100 study) | NEW required | NEEDS-RERUN if results are wanted |

**Verdict: OUT OF SPEC SCOPE — exploratory study; timing is still OLD/buggy if results exist**

---

### Study F: `fused-moe-characterization` (example)
**Config file:** `configs/study.example.yaml`

| Dimension | Config value | Notes |
|-----------|-------------|-------|
| Shapes | shape_a (E=16, non-standard) and shape_b (different inter) | Wrong shapes for spec |
| Backend | `deepep_low_latency` | Generic example |
| Status | Reference template only — never submitted | Not applicable |

**Verdict: REFERENCE ONLY — ignore**

---

## 4. Cross-Cutting Issues

### 4.1 Missing spec phases not covered by any config

| Phase | Spec requirement | Coverage |
|-------|-----------------|---------|
| Phase A: routing sensitivity at tp1-ep1 | All 6 routing modes × 11 token values × 2 shapes × 2 backends | Partially in routing-modes configs (6 modes, but only 6 token values, only 2 of 11) — NOT YET SUBMITTED |
| Phase B: EP sweep at fixed tokens=128 | uniform+skewed-4x × all 7 EP configs × 2 shapes | No dedicated config exists. Only 4 of 7 EP points are feasible (<=4 GPUs). tp1-ep8 and tp4-ep2 are hardware-blocked. |
| Phase C: tokens × TP/EP at uniform routing | uniform × 11 token values × all EP configs | No dedicated config exists; h100-deepep config covers 6 of 11 token values with alpha-sweep, not named routing mode |
| Phase D: DeepEPLL vs DeepEPHT | Both backends compared | deepep_high_throughput backend absent from all configs |

### 4.2 `gpus_per_node: 8` in H100 routing-modes configs
Both `study.routing-modes.allgather.yaml` and `study.routing-modes.deepep.yaml` declare
`gpus_per_node: 8` in the Slurm block. The submit script uses
`min(world_size, gpus_per_node)` when building the `--gres` line, so the actual
request for tp1-ep4 (world=4) will be `--gres=gpu:h100:4`, which is correct.
However the `gpus_per_node: 8` value will cause PACE ICE to allocate an
8-GPU-capable node (which may be harder to schedule) instead of requesting
exactly the 4 needed. Recommend changing to `gpus_per_node: 4` in these configs.

### 4.3 DeepEP backend `deepep_high_throughput` missing
The spec requires comparing `deepep_low_latency` vs `deepep_high_throughput` (Phase D).
No config with `all2all_backend: deepep_high_throughput` exists. A new config is needed.

### 4.4 Token range gap
Every config stops at tokens=4096. The spec extends to 200,000. The large-token
regime (16384–200000) is entirely unmeasured. At high token counts, DeepEP
high-throughput mode should dominate low-latency mode — this is a key comparison
point in Phase D and Phase C.

### 4.5 No Phase B config (EP sweep at fixed tokens=128)
The spec requires a dedicated study sweeping all EP configs at tokens=128 with
uniform and skewed-4x routing. The existing h100-deepep config comes closest
(alpha sweep, 4 EP points) but does not use named routing modes and is missing
tp2-ep1 entirely. tp1-ep8 and tp4-ep2 are hardware-blocked (>4 GPUs, single node).
A Phase B config needs to clarify whether multi-node EP is in scope.

---

## 5. Prioritized Action List

### Priority 1 — Must do before any new submissions

**1a. Commit the timing fix and routing wiring.**
The working tree has the correct deferred-sync timing code, `routing_mode` JSON field,
routing seed fix, and routing dispatch. These changes must be committed and synced
to the cluster (`git push` + `git pull` on PACE) before any job is resubmitted.
Files to commit: `src/fused_moe_kernel_study/runner.py`, `src/fused_moe_kernel_study/config.py`, `src/fused_moe_kernel_study/routing.py`, all three new config files.

**1b. Verify the DeepEP wheel is built.**
`study.pace.h100.deepep.yaml` and `study.routing-modes.deepep.yaml` reference
`/storage/ice1/0/2/sghose7/deepep-build/workspace/dist/deep_ep-*.whl`.
Confirm `slurm/build_deepep.sbatch` has been run and the wheel exists before
submitting any DeepEP study.

---

### Priority 2 — Token range expansion (required for spec compliance)

**2a. Extend token lists in all H100 study configs.**
Add small tokens (1, 4, 16) and large tokens (16384, 65536, 131072, 200000) to
`study.routing-modes.allgather.yaml`, `study.routing-modes.deepep.yaml`, and
`study.pace.h100.deepep.yaml`. Note that very large token counts will significantly
increase job runtime — adjust `--time` accordingly and test one condition interactively first.

---

### Priority 3 — Phase A: routing sensitivity (first complete spec phase)

**3a. Submit `routing-modes-allgather-h100`** (after 1a + 2a).
This is the NCCL baseline for Phase A. Two jobs: tp1-ep1 and tp1-ep4.
Fix `gpus_per_node: 8` -> `gpus_per_node: 4` in the config first.

**3b. Submit `routing-modes-deepep-h100`** (after 1a + 1b + 2a).
DeepEP side of Phase A. Same two parallel points.
Fix `gpus_per_node: 8` -> `gpus_per_node: 4` first.

---

### Priority 4 — Phase D: backend comparison

**4a. Create `study.routing-modes.deepep-ht-h100.yaml`** with
`all2all_backend: deepep_high_throughput` and otherwise identical to
`study.routing-modes.deepep.yaml`. Submit after 1a + 1b + 2a.

**4b. Rerun `fused-moe-characterization-pace-h100-deepep`** with fixed timing,
extended token range, and confirm EP coverage. This provides Phase C data.

---

### Priority 5 — Phase B: EP sweep at fixed tokens=128

**5a. Clarify hardware scope.** tp1-ep8 (8 GPUs) and tp4-ep2 (8 GPUs) are
hardware-blocked on single-node PACE ICE H100. Decide: drop them, or provision
multi-node jobs. If multi-node: set `min_nodes: 2` and `gpus_per_node: 4` in the
Slurm config section.

**5b. Create a dedicated Phase B config** sweeping [uniform, skewed-4x] × fixed
tokens=128 × all feasible EP points. The existing h100-deepep config is not
appropriate because it uses an alpha sweep rather than named routing modes.

---

### Priority 6 — A100 and Qwen3 studies (secondary)

**6a. Rerun `fused-moe-characterization-pace-a100`** with timing fix if A100
baseline data is wanted. These are off-spec hardware but useful for comparison.

**6b. Rerun `qwen3-30b-a3b-initial`** with timing fix if Qwen3 shape data is wanted.

---

## 6. Summary Table

| Study | Backend | Shapes | Token range | Routing param | Parallel points | GPU constraint | Timing | Verdict |
|-------|---------|--------|-------------|---------------|-----------------|----------------|--------|---------|
| fused-moe-characterization-pace-h100-deepep | deepep_low_latency | mixtral+deepseek (correct) | 128–4096 (too narrow) | Alpha sweep (wrong for Phase A) | tp1-ep1/2/4, tp2-ep2 (OK) | All <=4 GPUs: PASS | OLD/BUGGY | NEEDS-RERUN |
| routing-modes-allgather-h100 | allgather_reducescatter | mixtral+deepseek (correct) | 64–4096 (too narrow) | Named modes (correct for Phase A) | tp1-ep1, tp1-ep4 (OK) | All <=4 GPUs: PASS | NOT SUBMITTED YET | BLOCKED (commit fix first) |
| routing-modes-deepep-h100 | deepep_low_latency | mixtral+deepseek (correct) | 64–4096 (too narrow) | Named modes (correct for Phase A) | tp1-ep1, tp1-ep4 (OK) | All <=4 GPUs: PASS | NOT SUBMITTED YET | BLOCKED (commit fix + wheel first) |
| fused-moe-characterization-pace-a100 | deepep_low_latency | mixtral+deepseek (correct) | 128–4096 (too narrow) | Alpha sweep | tp1-ep1/2/4/8, tp2-ep2/4 (8-GPU pts feasible on A100) | tp1-ep8, tp2-ep4 exceed H100 limit (OK on A100) | OLD/BUGGY | NEEDS-RERUN (secondary priority) |
| qwen3-30b-a3b-initial | deepep_low_latency | Qwen3 shape (off-spec) | 128–1024 (very narrow) | Alpha sweep | tp1-ep1, tp1-ep4 (OK) | All <=4 GPUs: PASS | OLD/BUGGY | OUT OF SCOPE / NEEDS-RERUN if wanted |
| deepep_high_throughput (any) | — | — | — | — | — | — | — | MISSING — no config exists |
| Phase B EP sweep (dedicated) | — | — | — | — | — | — | — | MISSING — no config exists |
