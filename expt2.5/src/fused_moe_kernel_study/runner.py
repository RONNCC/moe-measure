from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from .buckets import breakdown_to_rows, categorize_dicts, profiler_to_event_dicts
from .config import BenchmarkCondition, StudyConfig
from .distributed import DistributedEnv
from .reporting import append_jsonl, ensure_dir, write_csv, write_json
from .routing import make_routing_batch
from .vllm_adapter import KernelArtifacts, build_kernel_artifacts, sp_local_sizes_context, vllm_config_context


def dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype={name!r}")
    return mapping[name]


def next_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def rank_geometry(rank: int, tp_size: int) -> tuple[int, int]:
    ep_rank = rank // tp_size
    tp_rank = rank % tp_size
    return ep_rank, tp_rank


def local_num_experts(num_experts: int, ep_size: int, ep_rank: int) -> int:
    base = num_experts // ep_size
    remainder = num_experts % ep_size
    return base + 1 if ep_rank < remainder else base


def local_expert_start(num_experts: int, ep_size: int, ep_rank: int) -> int:
    base = num_experts // ep_size
    remainder = num_experts % ep_size
    return ep_rank * base + min(ep_rank, remainder)


def make_expert_map(num_experts: int, ep_size: int, ep_rank: int, device: torch.device) -> torch.Tensor | None:
    if ep_size == 1:
        return None
    start = local_expert_start(num_experts, ep_size, ep_rank)
    count = local_num_experts(num_experts, ep_size, ep_rank)
    expert_map = torch.full((num_experts,), fill_value=-1, dtype=torch.int32, device=device)
    expert_map[start : start + count] = torch.arange(count, dtype=torch.int32, device=device)
    return expert_map


def make_local_weights(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    tp_size: int,
    ep_size: int,
    tp_rank: int,
    ep_rank: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    inter_per_tp = intermediate_size // tp_size
    num_local = local_num_experts(num_experts, ep_size, ep_rank)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + ep_rank * 10_000 + tp_rank)
    w1 = torch.randn(
        (num_local, 2 * inter_per_tp, hidden_size),
        generator=gen,
        dtype=dtype,
        device=device,
    ) / math.sqrt(hidden_size)
    w2 = torch.randn(
        (num_local, hidden_size, inter_per_tp),
        generator=gen,
        dtype=dtype,
        device=device,
    ) / math.sqrt(inter_per_tp)
    return w1.contiguous(), w2.contiguous()


def collect_hardware_snapshot() -> dict[str, Any]:
    commands = {
        "nvidia_smi_L": ["nvidia-smi", "-L"],
        "nvidia_smi_topo": ["nvidia-smi", "topo", "-m"],
        "nvidia_smi_query": [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version,pci.bus_id",
            "--format=csv,noheader",
        ],
        "ibv_devinfo": ["ibv_devinfo"],
    }
    out: dict[str, Any] = {"env": {k: os.environ[k] for k in sorted(os.environ) if k.startswith(("SLURM_", "CUDA", "NCCL", "VLLM", "MASTER_", "WORLD_SIZE", "RANK", "LOCAL_RANK"))}}
    for name, cmd in commands.items():
        try:
            out[name] = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        except Exception as exc:
            out[name] = f"UNAVAILABLE: {exc}"
    return out


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    idx = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return float(xs[idx])


def summarize_timings_ms(timings_ms: list[float]) -> dict[str, float]:
    return {
        "mean_ms": float(statistics.mean(timings_ms)),
        "median_ms": float(statistics.median(timings_ms)),
        "min_ms": float(min(timings_ms)),
        "max_ms": float(max(timings_ms)),
        "p05_ms": _percentile(timings_ms, 0.05),
        "p95_ms": _percentile(timings_ms, 0.95),
        "p99_ms": _percentile(timings_ms, 0.99),
        "std_ms": float(statistics.stdev(timings_ms)) if len(timings_ms) > 1 else 0.0,
    }


def profile_bucket_breakdown(
    *,
    cfg: StudyConfig,
    cond: BenchmarkCondition,
    env: DistributedEnv,
    artifacts: KernelArtifacts,
    kernel: Any,
    mk_kwargs: dict[str, Any],
    num_tokens_across_dp: torch.Tensor,
) -> dict[str, Any] | None:
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    from vllm.forward_context import set_forward_context

    dist.barrier()
    with vllm_config_context(artifacts.vllm_config):
        with torch.no_grad():
            with torch.profiler.profile(
                activities=activities,
                record_shapes=False,
                with_stack=False,
                acc_events=True,
            ) as prof:
                for _ in range(cfg.bucket_profile_iters):
                    with set_forward_context(
                        None,
                        artifacts.vllm_config,
                        num_tokens=cond.tokens,
                        num_tokens_across_dp=num_tokens_across_dp,
                    ):
                        with sp_local_sizes_context():
                            torch.cuda.nvtx.range_push("moe_kernel_bucket_profile")
                            _ = kernel.apply(**mk_kwargs)
                            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()

    raw_events = profiler_to_event_dicts(prof, full_events=cfg.bucket_full_events)
    breakdown = categorize_dicts(raw_events)
    local = {
        "rank": env.rank,
        "ep_rank": env.rank // cond.parallel.tp,
        "tp_rank": env.rank % cond.parallel.tp,
        "summary": breakdown.as_dict(),
        "rows": breakdown_to_rows(breakdown),
    }
    gathered: list[dict[str, Any] | None] = [None for _ in range(env.world_size)]
    dist.all_gather_object(gathered, local)
    if env.rank != 0:
        return None

    per_rank = [x for x in gathered if x is not None]
    max_rank_bucket_us = {bucket: max(float(rank_payload["summary"]["per_bucket_us"].get(bucket, 0.0)) for rank_payload in per_rank) for bucket in breakdown.as_dict()["per_bucket_us"].keys()}
    mean_rank_bucket_us = {bucket: float(statistics.mean(float(rank_payload["summary"]["per_bucket_us"].get(bucket, 0.0)) for rank_payload in per_rank)) for bucket in breakdown.as_dict()["per_bucket_us"].keys()}

    return {
        "condition": {
            "shape": cond.shape.name,
            "mode": cond.mode,
            "routing_mode": cond.routing_mode,
            "tp": cond.parallel.tp,
            "ep": cond.parallel.ep,
            "tokens": cond.tokens,
            "alpha": cond.alpha,
        },
        "profile_iters": cfg.bucket_profile_iters,
        "per_rank": per_rank,
        "max_rank_bucket_us": max_rank_bucket_us,
        "mean_rank_bucket_us": mean_rank_bucket_us,
    }


def _kernel_call_kwargs(
    *,
    kernel: Any,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    global_num_experts: int,
    apply_router_weight_on_input: bool,
    activation: str = "silu",
) -> dict[str, Any]:
    kwargs = {
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "expert_map": expert_map,
        "global_num_experts": global_num_experts,
        "apply_router_weight_on_input": apply_router_weight_on_input,
    }
    sig = None
    for attr in ("apply", "forward", "__call__"):
        if hasattr(kernel, attr):
            try:
                sig = getattr(kernel, attr)
                break
            except Exception:
                pass
    if sig is not None:
        import inspect

        params = inspect.signature(sig).parameters
        if "activation" in params:
            try:
                from vllm.model_executor.layers.fused_moe.activation import MoEActivation
                kwargs["activation"] = MoEActivation(activation)
            except (ImportError, ValueError):
                kwargs["activation"] = activation
        if "inplace" in params:
            kwargs["inplace"] = False
    return kwargs


# Hardware constants for H100 SXM5
H100_TFLOPS_BF16 = 989.0          # BF16 TFLOPS (sparsity off)
H100_HBM_BW_TBYPS = 3.35          # HBM bandwidth in TB/s
H100_FLOPS_PER_BYTE = H100_TFLOPS_BF16 / H100_HBM_BW_TBYPS  # ~295.2


def _derived_columns(summary: dict[str, Any]) -> dict[str, Any]:
    """Compute derived analytical columns from a condition summary dict."""
    tokens: int = int(summary["tokens"])
    hidden_size: int = int(summary["hidden_size"])
    intermediate_size: int = int(summary["intermediate_size"])
    num_experts: int = int(summary["num_experts"])
    topk: int = int(summary["topk"])
    ep: int = int(summary["ep"])
    tp: int = int(summary["tp"])

    bytes_per_element = 2  # bfloat16

    # Expert weight bytes held by this GPU's local experts
    num_local_experts = num_experts // ep
    w1_bytes = num_local_experts * 2 * intermediate_size * hidden_size * bytes_per_element // tp
    w2_bytes = num_local_experts * intermediate_size * hidden_size * bytes_per_element // tp
    expert_weight_bytes_per_gpu = w1_bytes + w2_bytes

    # Theoretical dispatch bytes (allgather of input hidden states across EP ranks)
    dispatch_bytes_theoretical = tokens * hidden_size * bytes_per_element * (ep - 1) / ep

    # Expert GEMM FLOPs (across all locally assigned tokens, averaged)
    tokens_per_expert_avg = tokens * topk / num_experts
    flops_per_expert = 2 * tokens_per_expert_avg * (2 * intermediate_size * hidden_size)
    expert_GEMM_flops = flops_per_expert * num_local_experts

    # Arithmetic intensity (FLOP/byte) — roofline x-axis
    if expert_weight_bytes_per_gpu > 0:
        expert_GEMM_AMI = expert_GEMM_flops / expert_weight_bytes_per_gpu
    else:
        expert_GEMM_AMI = float("inf")

    # Roofline prediction: compute-bound if AMI >= H100 ops-per-byte ridge point
    compute_bound_predicted = int(expert_GEMM_AMI >= H100_FLOPS_PER_BYTE)

    # Imbalance ratio (corrected)
    imbalance_ratio_alpha = float(summary["alpha_observed"])

    # Theoretical allgather network bytes (lower bound)
    allgather_send_bytes = tokens * hidden_size * bytes_per_element
    allgather_recv_bytes = tokens * hidden_size * bytes_per_element * (ep - 1)

    return {
        "expert_weight_bytes_per_gpu": expert_weight_bytes_per_gpu,
        "dispatch_bytes_theoretical": dispatch_bytes_theoretical,
        "expert_GEMM_flops": expert_GEMM_flops,
        "expert_GEMM_AMI": expert_GEMM_AMI,
        "compute_bound_predicted": compute_bound_predicted,
        "imbalance_ratio_alpha": imbalance_ratio_alpha,
        "allgather_send_bytes": allgather_send_bytes,
        "allgather_recv_bytes": allgather_recv_bytes,
    }


def run_condition(
    *,
    cfg: StudyConfig,
    cond: BenchmarkCondition,
    env: DistributedEnv,
    artifacts: KernelArtifacts,
    out_dir: Path,
) -> dict[str, Any] | None:
    device = torch.device("cuda", torch.cuda.current_device())
    ep_rank, tp_rank = rank_geometry(env.rank, cond.parallel.tp)
    dtype = dtype_from_name(cond.shape.dtype)

    hidden_gen = torch.Generator(device=device)
    hidden_gen.manual_seed(cfg.seed + cond.tokens + int(cond.alpha * 1000) + env.rank + hash(cond.routing_mode) % 10000)
    hidden_states = torch.randn(
        (cond.tokens, cond.shape.hidden_size),
        generator=hidden_gen,
        device=device,
        dtype=dtype,
    ) / math.sqrt(cond.shape.hidden_size)

    expert_map = make_expert_map(cond.shape.num_experts, cond.parallel.ep, ep_rank, device)
    w1, w2 = make_local_weights(
        hidden_size=cond.shape.hidden_size,
        intermediate_size=cond.shape.intermediate_size,
        num_experts=cond.shape.num_experts,
        tp_size=cond.parallel.tp,
        ep_size=cond.parallel.ep,
        tp_rank=tp_rank,
        ep_rank=ep_rank,
        dtype=dtype,
        device=device,
        seed=cfg.seed + 13,
    )

    routing = make_routing_batch(
        num_tokens=cond.tokens,
        num_experts=cond.shape.num_experts,
        topk=cond.shape.topk,
        alpha=cond.alpha,
        hot_expert_count=cfg.hot_expert_count,
        device=device,
        seed=cfg.seed + 17,
        weight_mode=cfg.routing_weight_mode,
        topk_index_dtype=artifacts.topk_index_dtype,
        routing_mode=cond.routing_mode,
    )

    num_tokens_across_dp = torch.tensor([cond.tokens] * cond.parallel.ep, device=device, dtype=torch.int32)
    from vllm.forward_context import set_forward_context

    kernel = artifacts.kernel
    mk_kwargs = _kernel_call_kwargs(
        kernel=kernel,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=routing.topk_weights,
        topk_ids=routing.topk_ids,
        expert_map=expert_map,
        global_num_experts=cond.shape.num_experts,
        apply_router_weight_on_input=cfg.apply_router_weight_on_input,
        activation=cond.shape.activation,
    )

    dist.barrier()
    with vllm_config_context(artifacts.vllm_config):
        with set_forward_context(
            None,
            artifacts.vllm_config,
            num_tokens=cond.tokens,
            num_tokens_across_dp=num_tokens_across_dp,
        ):
            for _ in range(cfg.warmup_iters):
                torch.cuda.nvtx.range_push("moe_kernel_warmup")
                with sp_local_sizes_context():
                    _ = kernel.apply(**mk_kwargs)
                torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    dist.barrier()

    # Deferred-sync timing: record all events without syncing inside the loop,
    # then synchronize once after all reps. This keeps the CPU-GPU sync out of
    # the hot path so each elapsed_time() reflects only kernel execution time.
    event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    with vllm_config_context(artifacts.vllm_config):
        for _ in range(cfg.measure_iters):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            with set_forward_context(
                None,
                artifacts.vllm_config,
                num_tokens=cond.tokens,
                num_tokens_across_dp=num_tokens_across_dp,
            ):
                with sp_local_sizes_context():
                    torch.cuda.nvtx.range_push("moe_kernel_timed")
                    start_event.record()
                    _ = kernel.apply(**mk_kwargs)
                    end_event.record()
                    torch.cuda.nvtx.range_pop()
            event_pairs.append((start_event, end_event))

    torch.cuda.synchronize()
    timings_ms = [float(s.elapsed_time(e)) for s, e in event_pairs]

    torch.cuda.synchronize()
    dist.barrier()

    bucket_profile = None
    if cfg.collect_buckets:
        bucket_profile = profile_bucket_breakdown(
            cfg=cfg,
            cond=cond,
            env=env,
            artifacts=artifacts,
            kernel=kernel,
            mk_kwargs=mk_kwargs,
            num_tokens_across_dp=num_tokens_across_dp,
        )

    local_summary = summarize_timings_ms(timings_ms)
    local_payload = {
        "rank": env.rank,
        "local_rank": env.local_rank,
        "ep_rank": ep_rank,
        "tp_rank": tp_rank,
        "timings_ms": timings_ms,
        **local_summary,
    }

    gathered: list[dict[str, Any] | None] = [None for _ in range(env.world_size)]
    dist.all_gather_object(gathered, local_payload)
    if env.rank != 0:
        return None

    per_rank = [item for item in gathered if item is not None]
    rank_medians = [float(item["median_ms"]) for item in per_rank]
    rank_means = [float(item["mean_ms"]) for item in per_rank]
    rank_p95 = [float(item["p95_ms"]) for item in per_rank]
    rank_p05 = [float(item["p05_ms"]) for item in per_rank]
    rank_stds = [
        float(statistics.stdev(item["timings_ms"])) if len(item["timings_ms"]) > 1 else 0.0
        for item in per_rank
    ]
    rank_p99s = [float(item.get("p99_ms", float("nan"))) for item in per_rank]
    rank_std_from_summary = [float(item.get("std_ms", 0.0)) for item in per_rank]

    summary = {
        "study_name": cfg.study_name,
        "shape": cond.shape.name,
        "mode": cond.mode,
        "routing_mode": cond.routing_mode,
        "tp": cond.parallel.tp,
        "ep": cond.parallel.ep,
        "world_size": cond.parallel.world_size,
        "tokens": cond.tokens,
        "alpha_requested": cond.alpha,
        "alpha_observed": routing.stats.observed_alpha,
        "backend": cfg.all2all_backend,
        "transport_condition": os.environ.get("TRANSPORT_CONDITION", "nvlink_default"),
        "dtype": cond.shape.dtype,
        "hidden_size": cond.shape.hidden_size,
        "intermediate_size": cond.shape.intermediate_size,
        "num_experts": cond.shape.num_experts,
        "topk": cond.shape.topk,
        "warmup_iters": cfg.warmup_iters,
        "measure_iters": cfg.measure_iters,
        "latency_median_ms_max_rank": max(rank_medians),
        "latency_mean_ms_max_rank": max(rank_means),
        "latency_median_ms_mean_across_ranks": float(statistics.mean(rank_medians)),
        "latency_mean_ms_mean_across_ranks": float(statistics.mean(rank_means)),
        "latency_p95_ms_max_rank": max(rank_p95),
        "latency_p05_ms_max_rank": max(rank_p05),
        "latency_std_ms_max_rank": max(rank_stds),
        "latency_p99_ms_max_rank": max(r for r in rank_p99s if not math.isnan(r)) if any(not math.isnan(r) for r in rank_p99s) else float("nan"),
        "latency_std_ms_mean_across_ranks": float(statistics.mean(rank_std_from_summary)),
        "per_rank": per_rank,
        "routing_counts": routing.stats.counts,
        "routing_probabilities": routing.stats.probabilities,
    }

    if cond.routing_mode != "alpha":
        stem = f"{cond.shape.name}-{cond.mode}-{cond.routing_mode}-tp{cond.parallel.tp}-ep{cond.parallel.ep}-tok{cond.tokens}"
    else:
        stem = f"{cond.shape.name}-{cond.mode}-tp{cond.parallel.tp}-ep{cond.parallel.ep}-tok{cond.tokens}-alpha{cond.alpha:.3f}"
    rank_dir = ensure_dir(out_dir / "per_rank")
    rank_path = rank_dir / f"{stem}.json"
    write_json(rank_path, summary)

    bucket_path = None
    if bucket_profile is not None:
        bucket_dir = ensure_dir(out_dir / "bucket_profiles")
        bucket_path = bucket_dir / f"{stem}.json"
        write_json(bucket_path, bucket_profile)

    row = {k: v for k, v in summary.items() if k not in {"per_rank", "routing_counts", "routing_probabilities"}}
    row.update(_derived_columns(summary))
    row["per_rank_path"] = str(rank_path)
    row["bucket_profile_path"] = str(bucket_path) if bucket_path is not None else ""
    if bucket_profile is not None:
        for bucket, value in bucket_profile["max_rank_bucket_us"].items():
            row[f"bucket_max_rank_{bucket}_ms"] = float(value) / 1000.0
        for bucket, value in bucket_profile["mean_rank_bucket_us"].items():
            row[f"bucket_mean_rank_{bucket}_ms"] = float(value) / 1000.0
    return row


def run_parallel_point(
    *,
    cfg: StudyConfig,
    parallel_tp: int,
    parallel_ep: int,
    env: DistributedEnv,
    out_dir: Path,
) -> None:
    out_dir = ensure_dir(out_dir)

    if env.rank == 0:
        write_json(out_dir / "hardware.json", collect_hardware_snapshot())
        write_json(out_dir / "study_config.json", {
            "study_config": asdict(cfg),
            "tp": parallel_tp,
            "ep": parallel_ep,
        })

    nccl_log_dir = None
    if cfg.all2all_backend not in ("none",):
        nccl_log_dir = ensure_dir(out_dir / "nccl_logs")
        nccl_log_file = str(nccl_log_dir / f"nccl_rank{env.rank}.log")
        os.environ.setdefault("NCCL_DEBUG", "INFO")
        os.environ.setdefault("NCCL_DEBUG_FILE", nccl_log_file)

    rows: list[dict[str, Any]] = []
    from .config import ParallelPoint, make_conditions

    parallel = ParallelPoint(tp=parallel_tp, ep=parallel_ep)
    conditions = make_conditions(cfg, parallel)
    dtype_name = cfg.kernel_shapes[0].dtype

    for shape in cfg.kernel_shapes:
        if shape.intermediate_size % parallel.tp != 0:
            raise ValueError(
                f"Shape {shape.name}: intermediate_size={shape.intermediate_size} must be divisible by tp={parallel.tp}."
            )
        if shape.num_experts < parallel.ep:
            raise ValueError(
                f"Shape {shape.name}: num_experts={shape.num_experts} must be >= ep={parallel.ep}."
            )

    artifacts_by_shape: dict[str, KernelArtifacts] = {}
    for shape in cfg.kernel_shapes:
        ep_rank, _ = rank_geometry(env.rank, parallel.tp)
        artifacts_by_shape[shape.name] = build_kernel_artifacts(
            shape_name=shape.name,
            hidden_size=shape.hidden_size,
            intermediate_size=shape.intermediate_size,
            num_experts=shape.num_experts,
            num_local_experts=local_num_experts(shape.num_experts, parallel.ep, ep_rank),
            topk=shape.topk,
            dtype=dtype_from_name(shape.dtype),
            max_num_tokens=next_power_of_two(cfg.max_tokens()),
            all2all_backend=cfg.all2all_backend,
            ep_size=parallel.ep,
        )

    if env.rank == 0:
        print(f"[study] {cfg.study_name} :: tp={parallel.tp} ep={parallel.ep} :: {len(conditions)} conditions")
        print(f"[study] backend={cfg.all2all_backend} dtype={dtype_name} out_dir={out_dir}")

    for cond in conditions:
        if env.rank == 0:
            print(
                f"[cond] shape={cond.shape.name} mode={cond.mode} routing_mode={cond.routing_mode} "
                f"tp={cond.parallel.tp} ep={cond.parallel.ep} tokens={cond.tokens} alpha={cond.alpha}"
            )
        row = run_condition(cfg=cfg, cond=cond, env=env, artifacts=artifacts_by_shape[cond.shape.name], out_dir=out_dir)
        if env.rank == 0 and row is not None:
            rows.append(row)
            append_jsonl(out_dir / "results.jsonl", row)

    if env.rank == 0:
        write_csv(out_dir / "results.csv", rows)
        write_json(out_dir / "results_summary.json", {"rows": rows, "num_rows": len(rows)})

    if env.rank == 0 and nccl_log_dir is not None:
        nccl_summary: dict[str, Any] = {"rank_files": []}
        for log_path in sorted(nccl_log_dir.glob("*.log")):
            bytes_total = 0
            op_counts: dict[str, int] = {}
            try:
                for line in log_path.read_text(errors="replace").splitlines():
                    # NCCL INFO lines look like: "NCCL INFO AllReduce: opCount ... count 1024 datatype ... nBytes 8192"
                    if "nBytes" in line:
                        parts = line.split("nBytes")
                        if len(parts) > 1:
                            try:
                                nb = int(parts[1].split()[1])
                                bytes_total += nb
                            except (IndexError, ValueError):
                                pass
                    for op in ("AllReduce", "AllGather", "ReduceScatter", "Send", "Recv", "Broadcast"):
                        if op in line:
                            op_counts[op] = op_counts.get(op, 0) + 1
            except Exception:
                pass
            nccl_summary["rank_files"].append({"file": str(log_path), "bytes_total": bytes_total, "op_counts": op_counts})
        write_json(out_dir / "nccl_summary.json", nccl_summary)
