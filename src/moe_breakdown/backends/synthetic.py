"""
Synthetic MoE trace generator.

Used for:
  * End-to-end testing without needing a GPU or large checkpoint
  * Producing a reproducible demo chart in CPU-only environments
  * Sanity-checking the categorizer against a known-good breakdown

The generator emits a stream of events that mimic what torch.profiler +
Kineto produces for a small MoE forward pass, *including realistic per-event
start/end times* so that gpu_idle_gap detection works.

Distribution is plausible (not measured from a real model), but every event
is named so the categorizer reproduces the target percentages within jitter.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class SyntheticConfig:
    num_experts: int = 8
    tokens: int = 256
    hidden: int = 1024
    ffn_hidden: int = 4096
    top_k: int = 2
    seed: int = 0

    # Target fractions of total wall-clock time per bucket.  These are
    # the "ground truth" the categorizer should recover (within jitter).
    cpu_python_frac:    float = 0.04
    cpu_native_frac:    float = 0.03
    gpu_compute_frac:   float = 0.50
    gpu_memory_frac:    float = 0.10
    gpu_idle_gap_frac:  float = 0.08
    gpu_idle_sync_frac: float = 0.04
    network_frac:       float = 0.13
    mem_transfer_frac:  float = 0.05
    allocator_frac:     float = 0.03


def profile(cfg: Optional[SyntheticConfig] = None) -> list[dict]:
    """Return a synthetic list of profiler events for one MoE forward pass."""
    cfg = cfg or SyntheticConfig()
    rng = random.Random(cfg.seed)

    fractions = {
        "cpu_python":    cfg.cpu_python_frac,
        "cpu_native":    cfg.cpu_native_frac,
        "gpu_compute":   cfg.gpu_compute_frac,
        "gpu_memory":    cfg.gpu_memory_frac,
        "gpu_idle_gap":  cfg.gpu_idle_gap_frac,
        "gpu_idle_sync": cfg.gpu_idle_sync_frac,
        "network":       cfg.network_frac,
        "mem_transfer":  cfg.mem_transfer_frac,
        "allocator":     cfg.allocator_frac,
    }
    total_frac = sum(fractions.values())
    if abs(total_frac - 1.0) > 0.01:
        fractions = {k: v / total_frac for k, v in fractions.items()}

    # Total wall-clock time = ~150 ms (one small forward + a few decode steps).
    wall_clock_us = 150_000.0

    # Per-bucket total duration budgets.
    bucket_us = {k: wall_clock_us * frac for k, frac in fractions.items()}

    events: list[dict] = []
    cursor_us = 0.0
    stream = "stream0"

    def emit(
        name: str, cat: str, device: str, bucket: str,
        frac: float = 1.0, jitter: float = 0.15,
    ):
        """Emit one event consuming `frac` of the remaining bucket budget.

        The actual duration is bucket_us[bucket] * frac * (1 + jitter).
        This guarantees the per-bucket totals add up to the configured
        fractions (within jitter).
        """
        nonlocal cursor_us
        budget = bucket_us[bucket]
        if budget <= 0:
            return
        dur = max(0.0, budget * frac * (1.0 + rng.uniform(-jitter, jitter)))
        bucket_us[bucket] = max(0.0, budget - dur)
        start = cursor_us
        end = start + dur
        events.append({
            "name": name, "category": cat, "device": device,
            "duration_us": dur, "start_us": start, "end_us": end,
            "stream": stream,
        })
        cursor_us = end

    def idle_gap(frac: float):
        """Insert a CPU-only idle gap (no events emitted -> gap detector picks up)."""
        nonlocal cursor_us
        gap = wall_clock_us * frac
        cursor_us += gap

    # ---- 1. CPU Python: tokenize + dispatcher overhead ----------------- #
    for _ in range(4):
        emit("python_tokenize", "cpu_op", "cpu", "cpu_python", frac=0.20, jitter=0.3)
    emit("autograd::evaluate_function", "python", "cpu", "cpu_python", frac=0.20, jitter=0.3)

    # ---- 2. CPU Native: data prep -------------------------------------- #
    emit("aten::to", "cpu_op", "cpu", "cpu_native", frac=0.5, jitter=0.2)
    emit("aten::cat", "cpu_op", "cpu", "cpu_native", frac=0.5, jitter=0.2)

    # ---- 3. Allocator: KV cache + intermediate buffers ----------------- #
    emit("caching_allocator_alloc", "cpu_op", "cpu", "allocator", frac=0.5, jitter=0.2)
    emit("aten::empty", "cpu_op", "cpu", "allocator", frac=0.5, jitter=0.2)

    # ---- 4. H2D copy of input IDs ------------------------------------- #
    emit("cudaMemcpyAsync", "gpu_memcpy", "cuda", "mem_transfer", frac=0.5, jitter=0.2)

    # ---- 5. Embedding gather (memory-bound) ---------------------------- #
    emit("embedding_dense_kernel", "kernel", "cuda", "gpu_memory", frac=0.20, jitter=0.1)
    emit("rotary_embedding_kernel", "kernel", "cuda", "gpu_memory", frac=0.20, jitter=0.1)

    # ---- 6. Attention QKV + SDPA (compute) ---------------------------- #
    emit("gemm_kernel_qkv", "kernel", "cuda", "gpu_compute", frac=0.10, jitter=0.1)
    emit("scaled_dot_product_attention_flash_fwd", "kernel", "cuda", "gpu_compute", frac=0.20, jitter=0.05)
    emit("gemm_kernel_out_proj", "kernel", "cuda", "gpu_compute", frac=0.10, jitter=0.1)

    # ---- 7. MoE router + top-k (memory-bound) ------------------------- #
    emit("topk_kernel", "kernel", "cuda", "gpu_memory", frac=0.10, jitter=0.1)
    emit("moe_gather_kernel", "kernel", "cuda", "gpu_memory", frac=0.10, jitter=0.1)
    emit("softmax_kernel", "kernel", "cuda", "gpu_memory", frac=0.10, jitter=0.1)
    emit("moe_scatter_kernel", "kernel", "cuda", "gpu_memory", frac=0.10, jitter=0.1)

    # ---- 8. AllToAll dispatch (network) ------------------------------- #
    emit("nccl_all_to_all", "communication", "cuda", "network", frac=0.30, jitter=0.1)
    emit("nccl_all_to_all", "communication", "cuda", "network", frac=0.20, jitter=0.1)

    # ---- 9. Per-expert FFN (compute -- dominant) ---------------------- #
    for i in range(cfg.num_experts):
        emit(f"gemm_kernel_expert_{i}_w1", "kernel", "cuda", "gpu_compute",
             frac=0.05 / max(1, cfg.num_experts / 2), jitter=0.1)
        emit(f"silu_kernel", "kernel", "cuda", "gpu_memory",
             frac=0.04 / max(1, cfg.num_experts / 2), jitter=0.2)
        emit(f"gemm_kernel_expert_{i}_w2", "kernel", "cuda", "gpu_compute",
             frac=0.05 / max(1, cfg.num_experts / 2), jitter=0.1)

    # ---- 10. AllToAll combine (network) -------------------------------- #
    emit("nccl_all_to_all", "communication", "cuda", "network", frac=0.30, jitter=0.1)
    emit("nccl_all_reduce", "communication", "cuda", "network", frac=0.20, jitter=0.1)

    # ---- 11. Final norm + logits projection (memory + compute) -------- #
    emit("rmsnorm_kernel", "kernel", "cuda", "gpu_memory", frac=0.20, jitter=0.1)
    emit("gemm_kernel_lm_head", "kernel", "cuda", "gpu_compute", frac=1.0, jitter=0.05)

    # ---- 12. Sampling logits (CPU Python + sync) ---------------------- #
    emit("aten::argmax", "cpu_op", "cpu", "cpu_native", frac=1.0, jitter=0.2)
    emit("cudaStreamSynchronize", "cuda_runtime", "cuda", "gpu_idle_sync", frac=1.0, jitter=0.3)
    emit("python_sample_topk", "cpu_op", "cpu", "cpu_python", frac=1.0, jitter=0.3)

    # ---- 13. D2H copy of selected tokens ------------------------------ #
    emit("cudaMemcpyAsync", "gpu_memcpy", "cuda", "mem_transfer", frac=0.5, jitter=0.2)
    emit("aten::copy_", "memcpy", "cpu", "mem_transfer", frac=0.5, jitter=0.3)

    # ---- 14. CPU native: detokenize ----------------------------------- #
    emit("python_detokenize", "cpu_op", "cpu", "cpu_python", frac=1.0, jitter=0.3)

    # ---- 15. Insert explicit CPU-only idle gap (consumes gpu_idle_gap) -
    idle_gap(cfg.gpu_idle_gap_frac)

    # ---- 16. Allocator: free temporary buffers ------------------------ #
    emit("caching_allocator_free", "cpu_op", "cpu", "allocator", frac=1.0, jitter=0.3)

    return events



def calibrated_from_model_config(model_config, wall_clock_us: float = 150_000.0) -> "SyntheticConfig":
    """Build a SyntheticConfig that mirrors a real HF model's shape.

    Use this to produce a *projection* of what a real GPU run of this model
    would look like.  The projection is approximate -- it does not
    reproduce measured GPU times -- but it has the right *shape* (number
    of expert FFNs, attention calls, etc.) so the bucket percentages
    tell a believable story.

    Parameters
    ----------
    model_config : transformers.PretrainedConfig
        The config object returned by `AutoConfig.from_pretrained(...)`.
    wall_clock_us : float
        Total profiled time budget for the projection (microseconds).
    """
    num_experts = int(getattr(model_config, "num_local_experts", 8))
    top_k = int(getattr(model_config, "num_experts_per_tok", 2))
    hidden = int(getattr(model_config, "hidden_size", 256))
    ffn = int(getattr(model_config, "intermediate_size", 1024))
    seq_len = 256  # default projection seq len
    return SyntheticConfig(
        num_experts=num_experts,
        top_k=top_k,
        hidden=hidden,
        ffn_hidden=ffn,
        tokens=seq_len,
        # Plausible GPU breakdown: compute dominates, network ~10-15%,
        # allocator & sync a few %, the rest is memory-bound kernels.
        gpu_compute_frac=0.55,
        gpu_memory_frac=0.10,
        gpu_idle_gap_frac=0.08,
        gpu_idle_sync_frac=0.04,
        network_frac=0.13,
        mem_transfer_frac=0.05,
        allocator_frac=0.03,
        cpu_python_frac=0.01,
        cpu_native_frac=0.01,
    )
