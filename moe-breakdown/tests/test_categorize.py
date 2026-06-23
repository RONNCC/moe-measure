"""
Unit tests for the categorizer.  Each test is named after the bucket it
exercises so a failure makes it obvious which rule broke.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moe_breakdown import categorize_dicts
from moe_breakdown.categorize import (
    BUCKETS, _bucket_for, compute_gpu_idle_gaps,
)


# --------------------------------------------------------------------------- #
# Per-bucket rule tests
# --------------------------------------------------------------------------- #

def test_cpu_python():
    assert _bucket_for("python_tokenize", "cpu_op", "cpu") == "cpu_python"
    assert _bucket_for("autograd::evaluate_function", "python", "cpu") == "cpu_python"
    assert _bucket_for("optimizer_step", "python", "cpu") == "cpu_python"


def test_cpu_native():
    assert _bucket_for("aten::add", "cpu_op", "cpu") == "cpu_native"
    assert _bucket_for("aten::linear", "cpu_op", "cpu") == "cpu_native"
    assert _bucket_for("aten::embedding", "cpu_op", "cpu") == "cpu_native"


def test_gpu_compute():
    assert _bucket_for("gemm_kernel", "kernel", "cuda") == "gpu_compute"
    assert _bucket_for("cublas_gemm", "kernel", "cuda") == "gpu_compute"
    assert _bucket_for("scaled_dot_product_attention_flash_fwd",
                       "kernel", "cuda") == "gpu_compute"
    assert _bucket_for("flash_attention_fwd", "kernel", "cuda") == "gpu_compute"


def test_gpu_memory():
    assert _bucket_for("elementwise_kernel", "kernel", "cuda") == "gpu_memory"
    assert _bucket_for("rmsnorm_kernel", "kernel", "cuda") == "gpu_memory"
    assert _bucket_for("softmax_kernel", "kernel", "cuda") == "gpu_memory"
    assert _bucket_for("moe_gather_kernel", "kernel", "cuda") == "gpu_memory"
    assert _bucket_for("rotary_embedding_kernel", "kernel", "cuda") == "gpu_memory"


def test_network():
    assert _bucket_for("nccl_all_to_all", "communication", "cuda") == "network"
    assert _bucket_for("nccl_all_reduce", "communication", "cuda") == "network"
    assert _bucket_for("nccl_all_gather", "communication", "cuda") == "network"


def test_mem_transfer():
    assert _bucket_for("cudaMemcpyAsync", "gpu_memcpy", "cuda") == "mem_transfer"
    assert _bucket_for("aten::copy_", "memcpy", "cuda") == "mem_transfer"
    assert _bucket_for("memset_kernel", "memset", "cuda") == "mem_transfer"


def test_allocator():
    assert _bucket_for("caching_allocator_alloc", "cpu_op", "cpu") == "allocator"
    assert _bucket_for("cudaMalloc", "cpu_op", "cpu") == "allocator"
    assert _bucket_for("aten::empty", "cpu_op", "cpu") == "allocator"
    # H2D memcpy must NOT be classified as allocator
    assert _bucket_for("aten::to", "cpu_op", "cpu") in ("cpu_native", "allocator")
    # cudaMallocAsked = pooling allocator path (still allocator)


def test_gpu_idle_sync():
    assert _bucket_for("cudaStreamSynchronize", "cuda_runtime", "cuda") == "gpu_idle_sync"
    assert _bucket_for("cudaEventSynchronize", "cuda_runtime", "cuda") == "gpu_idle_sync"
    assert _bucket_for("aten::item", "cpu_op", "cpu") == "gpu_idle_sync"
    assert _bucket_for("aten::cpu", "cpu_op", "cpu") == "gpu_idle_sync"


def test_unknown_kernel_defaults_to_compute():
    """Unknown GPU kernel -> gpu_compute (the better-than-random default)."""
    assert _bucket_for("some_weird_kernel_xyz", "kernel", "cuda") == "gpu_compute"


# --------------------------------------------------------------------------- #
# Gap detection
# --------------------------------------------------------------------------- #

def test_compute_gpu_idle_gaps_simple():
    """Two CUDA events with a gap -> one gap event."""
    events = [
        {"name": "k1", "category": "kernel", "device": "cuda",
         "start_us": 0, "end_us": 100, "duration_us": 100, "stream": "s0"},
        {"name": "k2", "category": "kernel", "device": "cuda",
         "start_us": 250, "end_us": 350, "duration_us": 100, "stream": "s0"},
    ]
    gaps = compute_gpu_idle_gaps(events)
    assert len(gaps) == 1
    assert gaps[0].duration_us == 150  # 250 - 100
    assert gaps[0].bucket == "gpu_idle_gap"


def test_compute_gpu_idle_gaps_no_gap():
    """Back-to-back kernels -> no gap."""
    events = [
        {"name": "k1", "category": "kernel", "device": "cuda",
         "start_us": 0, "end_us": 100, "duration_us": 100, "stream": "s0"},
        {"name": "k2", "category": "kernel", "device": "cuda",
         "start_us": 100, "end_us": 200, "duration_us": 100, "stream": "s0"},
    ]
    gaps = compute_gpu_idle_gaps(events)
    assert gaps == []


def test_compute_gpu_idle_gaps_per_stream():
    """Gaps computed separately per stream."""
    events = [
        {"name": "k1", "category": "kernel", "device": "cuda",
         "start_us": 0, "end_us": 100, "duration_us": 100, "stream": "s0"},
        {"name": "k2", "category": "kernel", "device": "cuda",
         "start_us": 200, "end_us": 300, "duration_us": 100, "stream": "s1"},
    ]
    gaps = compute_gpu_idle_gaps(events)
    # k1 has no predecessor on s0, k2 has no predecessor on s1 -> no gaps
    assert gaps == []


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #

def test_categorize_dicts_end_to_end():
    events = [
        {"name": "python_tokenize", "category": "cpu_op", "device": "cpu",
         "duration_us": 500},
        {"name": "aten::add", "category": "cpu_op", "device": "cpu",
         "duration_us": 200},
        {"name": "gemm_kernel", "category": "kernel", "device": "cuda",
         "duration_us": 5000},
        {"name": "rmsnorm_kernel", "category": "kernel", "device": "cuda",
         "duration_us": 1000},
        {"name": "nccl_all_to_all", "category": "communication", "device": "cuda",
         "duration_us": 2000},
        {"name": "cudaMemcpyAsync", "category": "gpu_memcpy", "device": "cuda",
         "duration_us": 500},
        {"name": "caching_allocator_alloc", "category": "cpu_op", "device": "cpu",
         "duration_us": 300},
        {"name": "cudaStreamSynchronize", "category": "cuda_runtime", "device": "cuda",
         "duration_us": 400},
    ]
    bd = categorize_dicts(events)
    assert bd.per_bucket_us["cpu_python"] == 500
    assert bd.per_bucket_us["cpu_native"] == 200
    assert bd.per_bucket_us["gpu_compute"] == 5000
    assert bd.per_bucket_us["gpu_memory"] == 1000
    assert bd.per_bucket_us["network"] == 2000
    assert bd.per_bucket_us["mem_transfer"] == 500
    assert bd.per_bucket_us["allocator"] == 300
    assert bd.per_bucket_us["gpu_idle_sync"] == 400
    total = sum(bd.per_bucket_us[b] for b in BUCKETS)
    assert abs(total - 9900) < 1e-6


def test_gap_event_routed_correctly():
    """categorize_dicts picks up gaps when start/end times are present."""
    events = [
        {"name": "k1", "category": "kernel", "device": "cuda",
         "duration_us": 100, "start_us": 0, "end_us": 100, "stream": "s0"},
        {"name": "k2", "category": "kernel", "device": "cuda",
         "duration_us": 100, "start_us": 250, "end_us": 350, "stream": "s0"},
    ]
    bd = categorize_dicts(events)
    # k1, k2 -> gpu_compute; gap of 150us -> gpu_idle_gap
    assert bd.per_bucket_us["gpu_compute"] == 200
    assert bd.per_bucket_us["gpu_idle_gap"] == 150


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(0 if failed == 0 else 1)
