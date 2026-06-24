from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from .config import KernelShape, ParallelPoint, StudyConfig, load_study_config
from .distributed import cleanup_vllm_distributed, init_vllm_distributed
from .runner import run_parallel_point


@dataclass(frozen=True)
class MoEKernelConfig:
    shape_name: str
    hidden_size: int
    intermediate_size: int
    num_experts: int
    topk: int = 2
    dtype: str = "bfloat16"
    activation: str = "silu"
    all2all_backend: str = "deepep_low_latency"
    tp_size: int = 1
    ep_size: int = 1
    num_tokens: int = 512
    alpha: float = 1.0
    warmup_iters: int = 20
    measure_iters: int = 100
    seed: int = 0
    hot_expert_count: int = 1
    apply_router_weight_on_input: bool = False
    routing_weight_mode: str = "uniform"
    collect_buckets: bool = True
    bucket_profile_iters: int = 5
    bucket_full_events: bool = True
    output_root: str = "runs"
    out_dir: str | None = None


@dataclass
class KernelMeasurement:
    shape_name: str
    tp_size: int
    ep_size: int
    num_tokens: int
    alpha: float
    latency_median_ms_max_rank: float
    latency_mean_ms_max_rank: float
    latency_median_ms_mean_across_ranks: float
    latency_mean_ms_mean_across_ranks: float
    results_dir: str
    per_rank_path: str = ""
    bucket_profile_path: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def _study_from_kernel_config(config: MoEKernelConfig) -> tuple[StudyConfig, Path]:
    shape = KernelShape(
        name=config.shape_name,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_experts=config.num_experts,
        topk=config.topk,
        dtype=config.dtype,
        activation=config.activation,
    )
    parallel = ParallelPoint(tp=config.tp_size, ep=config.ep_size)
    out_dir = Path(config.out_dir or Path(config.output_root) / "single_measurements" / f"{config.shape_name}-tp{config.tp_size}-ep{config.ep_size}-tok{config.num_tokens}-alpha{config.alpha:.3f}")
    study = StudyConfig(
        study_name="single-moe-kernel-measurement",
        all2all_backend=config.all2all_backend,
        kernel_shapes=[shape],
        parallel_points=[parallel],
        alphas=[config.alpha],
        tokens=[config.num_tokens],
        sweep_modes=["full_factorial"],
        warmup_iters=config.warmup_iters,
        measure_iters=config.measure_iters,
        seed=config.seed,
        apply_router_weight_on_input=config.apply_router_weight_on_input,
        hot_expert_count=config.hot_expert_count,
        output_root=config.output_root,
        routing_weight_mode=config.routing_weight_mode,
        collect_buckets=config.collect_buckets,
        bucket_profile_iters=config.bucket_profile_iters,
        bucket_full_events=config.bucket_full_events,
        baseline={"alpha": config.alpha, "tokens": config.num_tokens, "parallel": {"tp": config.tp_size, "ep": config.ep_size}},
    )
    return study, out_dir


def measure_moe_kernel_latency(config: MoEKernelConfig) -> KernelMeasurement:
    study, out_dir = _study_from_kernel_config(config)
    env = init_vllm_distributed(tp_size=config.tp_size, ep_size=config.ep_size, backend="nccl")
    payload: dict[str, Any] | None = None
    try:
        run_parallel_point(cfg=study, parallel_tp=config.tp_size, parallel_ep=config.ep_size, env=env, out_dir=out_dir)
        if env.rank == 0:
            import json

            summary = json.loads((out_dir / "results_summary.json").read_text())
            if not summary.get("rows"):
                raise RuntimeError(f"No rows written to {out_dir / 'results_summary.json'}")
            row = summary["rows"][0]
            payload = {
                "shape_name": config.shape_name,
                "tp_size": config.tp_size,
                "ep_size": config.ep_size,
                "num_tokens": config.num_tokens,
                "alpha": config.alpha,
                "latency_median_ms_max_rank": row["latency_median_ms_max_rank"],
                "latency_mean_ms_max_rank": row["latency_mean_ms_max_rank"],
                "latency_median_ms_mean_across_ranks": row["latency_median_ms_mean_across_ranks"],
                "latency_mean_ms_mean_across_ranks": row["latency_mean_ms_mean_across_ranks"],
                "results_dir": str(out_dir),
                "per_rank_path": row.get("per_rank_path", ""),
                "bucket_profile_path": row.get("bucket_profile_path", ""),
                "raw": row,
            }
        object_list = [payload]
        dist.broadcast_object_list(object_list, src=0)
        payload = object_list[0]
    finally:
        cleanup_vllm_distributed()
    if payload is None:
        raise RuntimeError("Failed to create measurement payload.")
    return KernelMeasurement(**payload)


def run_sweep(config: str | Path | StudyConfig, tp_size: int, ep_size: int, out_dir: str | Path | None = None) -> None:
    if isinstance(config, (str, Path)):
        study = load_study_config(config)
    else:
        study = config
    out_path = Path(out_dir or Path(study.output_root) / study.study_name / f"tp{tp_size}-ep{ep_size}")
    env = init_vllm_distributed(tp_size=tp_size, ep_size=ep_size, backend="nccl")
    try:
        run_parallel_point(cfg=study, parallel_tp=tp_size, parallel_ep=ep_size, env=env, out_dir=out_path)
    finally:
        cleanup_vllm_distributed()
