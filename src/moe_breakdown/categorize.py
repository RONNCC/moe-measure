"""
Categorization of PyTorch Kineto profiler events into execution-time buckets.

The buckets decompose one MoE forward pass into the places time actually
goes.  Each rule is a compiled regex matched against the event name and
the (category, device_type) pair, with the first match winning.

    cpu_python    - Python interpreter, dispatcher, autograd traversal
    cpu_native    - Native CPU ops (aten on CPU thread, tokenize, sample,
                    data prep)
    gpu_compute   - Compute-bound GPU kernels (matmul, gemm, conv, attention)
    gpu_memory    - Memory-bound GPU kernels (elementwise, RMSNorm, Softmax,
                    MoE dispatch, embedding gather/scatter)
    gpu_idle_gap  - Wall-clock gap between consecutive GPU kernels on the
                    same stream -- mostly CPU dispatch latency
    gpu_idle_sync - Explicit GPU sync stalls (cudaStreamSynchronize,
                    cudaEventSynchronize, .item(), .cpu())
    network       - Collective communication (NCCL AllToAll / AllReduce /
                    AllGather used in expert routing)
    mem_transfer  - DMA copies (H2D, D2H, D2D, memset)
    allocator     - CUDA caching allocator work (cudaMalloc, cudaFree,
                    StorageImpl::allocate, caching pool)

The categorizer accepts plain dicts so it can be driven from any backend
(real torch.profiler, vLLM HTTP traces, synthetic fixtures, JSONL replays).
For GPU idle-gap detection, pass events with `start_us` and `end_us` set.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

# --------------------------------------------------------------------------- #
# Bucket definitions
# --------------------------------------------------------------------------- #

BUCKETS = (
    "cpu_python",
    "cpu_native",
    "gpu_compute",
    "gpu_memory",
    "gpu_idle_gap",
    "gpu_idle_sync",
    "network",
    "mem_transfer",
    "allocator",
)

# Display labels and chart colors.  Chosen so adjacent buckets on the pie
# are visually distinguishable even in monochrome printouts.
BUCKET_STYLE: dict[str, dict[str, str]] = {
    "cpu_python":    {"label": "CPU — Python",       "color": "#5B8FF9"},
    "cpu_native":    {"label": "CPU — Native",       "color": "#1F4E9D"},
    "gpu_compute":   {"label": "GPU Compute",        "color": "#F6BD16"},
    "gpu_memory":    {"label": "GPU Memory-bound",   "color": "#E8684A"},
    "gpu_idle_gap":  {"label": "GPU Idle — gap",     "color": "#9AA0A6"},
    "gpu_idle_sync": {"label": "GPU Idle — sync",    "color": "#5F6368"},
    "network":       {"label": "Network",            "color": "#6DC354"},
    "mem_transfer":  {"label": "Mem Transfer (DMA)", "color": "#9270CA"},
    "allocator":     {"label": "Allocator",          "color": "#FF9D4D"},
}


# --------------------------------------------------------------------------- #
# Pattern tables
# --------------------------------------------------------------------------- #

# Python interpreter / framework overhead on CPU.  Patterns are matched
# against the lowercased name.
_PYTHON_PATTERNS = [
    r"^python", r"^pyeval", r"^pyobject", r"^dispatch_key",
    r"autograd", r"^grad", r"^optimizer",
    r"^profiler", r"^record_function",
] 
_PYTHON_RE = re.compile("|".join(_PYTHON_PATTERNS))

# Allocator work.  Patterns are matched against the lowercased name.
_ALLOC_PATTERNS = [
    r"cudamalloc", r"cudafree",                              # cudaMalloc / cudaFree
    r"caching_allocator", r"atencachingallocator",
    r"storageimpl", r"\baten::empty\b", r"\baten::zeros\b",
    r"\baten::ones\b", r"\baten::full\b",
    r"\baten::to\b.*\bdevice\b",                              # host->device move implies alloc
] 
_ALLOC_RE = re.compile("|".join(_ALLOC_PATTERNS))

# Compute-bound GPU kernels.
_COMPUTE_PATTERNS = [
    r"\bgemm\b", r"\bmatmul\b", r"\bmm\b", r"\bbmm\b",
    r"\bconv\b", r"convolution",
    r"\battention\b", r"scaled_dot_product", r"\bsdpa\b", r"\bflash\b",
    r"\bfwd\b", r"\bbackward\b", r"\bbwd\b",
    r"\bcutlass\b", r"\bcublas\b",
    r"fused.*attention",
]
_COMPUTE_RE = re.compile("|".join(_COMPUTE_PATTERNS))


# Custom "word boundary" that treats `_` as a separator (Python's `\b`
# doesn't, since `_` is a word character).  Patterns below use this helper.
def _sep(pattern: str) -> str:
    """Wrap a pattern so it matches only at underscore-or-punctuation boundaries."""
    return rf"(?<![a-z0-9_])" + pattern + rf"(?![a-z0-9_])"


# Memory-bound GPU kernels.
_MEMORY_PATTERNS = [
    r"elementwise", r"vectorized", _sep(r"vector"),
    r"copy_kernel", _sep(r"copy"), _sep(r"cast"), _sep(r"nonzero"),
    _sep(r"norm"), r"rmsnorm", r"layernorm", r"groupnorm",
    r"softmax", _sep(r"silu"), _sep(r"gelu"), _sep(r"relu"),
    _sep(r"sigmoid"), _sep(r"tanh"),
    _sep(r"index"), _sep(r"gather"), _sep(r"scatter"), _sep(r"embedding"),
    _sep(r"topk"), _sep(r"sort"), r"moe_", _sep(r"one_hot"),
    _sep(r"mask"),
    _sep(r"cumsum"), r"cat_", _sep(r"stack"), _sep(r"split"),
    _sep(r"reshape"), _sep(r"view"), _sep(r"permute"), _sep(r"transpose"),
    r"kv_cache", r"rotary", _sep(r"rope"),
]
_MEMORY_RE = re.compile("|".join(_MEMORY_PATTERNS))

# Collective communication.
_NETWORK_PATTERNS = [
    r"allreduce", r"all_reduce",
    r"allgather", r"all_gather",
    r"alltoall", r"all_to_all",
    r"reducescatter", r"reduce_scatter",
    r"\bbroadcast\b",
    r"\bnccl\b",
    r"^send$", r"^recv$",
]
_NETWORK_RE = re.compile("|".join(_NETWORK_PATTERNS))

# DMA / memory transfer.
_MEMXFER_PATTERNS = [
    r"memcpy", r"memset",
    r"h2d", r"d2h", r"d2d", r"pcie",
]
_MEMXFER_RE = re.compile("|".join(_MEMXFER_PATTERNS))

# Explicit GPU sync stalls.  Patterns are matched against the lowercased name.
_SYNC_PATTERNS = [
    r"streamsynchronize", r"eventsynchronize",
    r"cudadevicesynchronize", r"cudastreamwaitevent",
    r"\baten::item\b", r"\baten::cpu\b", r"\baten::numpy\b",
    r"\baten::tolist\b",
] 
_SYNC_RE = re.compile("|".join(_SYNC_PATTERNS))


def _bucket_for(name: str, cat: str, device: str) -> Optional[str]:
    """Return the bucket for one event, or None to skip it.

    Order matters: allocator/sync/network/memcpy are checked before the
    generic CPU/GPU dispatch rules.
    """
    n = (name or "").lower()

    # 1) Explicit sync events -- counted as GPU idle even though they
    #    fire on the CPU thread.
    if _SYNC_RE.search(n) or cat == "sync":
        return "gpu_idle_sync"

    # 2) Collective communication.
    if cat == "communication" or _NETWORK_RE.search(n):
        return "network"

    # 3) Allocator work (cudaMalloc/Free, caching pool, large empty/zeros).
    if _ALLOC_RE.search(n) and not _MEMXFER_RE.search(n):
        return "allocator"

    # 4) DMA copies.
    if cat in ("memcpy", "memset", "gpu_memcpy") or _MEMXFER_RE.search(n):
        return "mem_transfer"

    # 5) CUDA runtime API calls live on the CPU thread.
    if cat == "cuda_runtime":
        return "cpu_native"

    # 6) CPU ops: split Python framework from native ATen.
    if device == "cpu" or cat == "cpu_op":
        if _PYTHON_RE.search(n) or cat == "python":
            return "cpu_python"
        return "cpu_native"

    # 7) GPU kernel: pick compute vs memory bound by name pattern.
    if cat == "kernel" and device == "cuda":
        if _MEMORY_RE.search(n):
            return "gpu_memory"
        if _COMPUTE_RE.search(n):
            return "gpu_compute"
        # Unknown GPU kernel -> compute by default (MoE forward is
        # dominated by matmul-style kernels, so this is the
        # better-than-random default).
        return "gpu_compute"

    return None  # skip framework/profiler events


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

@dataclass
class CategorizedEvent:
    name: str
    bucket: str
    duration_us: float
    category: str = ""
    device: str = ""
    start_us: float = 0.0
    end_us: float = 0.0
    stream: str = ""
    synthetic: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class Breakdown:
    """Result of categorizing a profiler trace."""
    total_us: float
    per_bucket_us: dict[str, float]
    per_bucket_count: dict[str, int]
    events: list[CategorizedEvent]

    def percent(self, bucket: str) -> float:
        return 100.0 * self.per_bucket_us.get(bucket, 0.0) / self.total_us if self.total_us else 0.0

    def as_dict(self) -> dict:
        return {
            "total_us": self.total_us,
            "per_bucket_us": self.per_bucket_us,
            "per_bucket_pct": {b: round(self.percent(b), 3) for b in BUCKETS},
            "per_bucket_count": self.per_bucket_count,
            "n_events": len(self.events),
        }


# --------------------------------------------------------------------------- #
# Iteration over torch.profiler
# --------------------------------------------------------------------------- #

def _iter_kineto_events(profiler) -> Iterable:
    """Yield raw Kineto event objects from a torch.profiler profile."""
    try:
        return profiler.key_averages()
    except Exception:
        return []


def categorize(profiler) -> Breakdown:
    """Categorize events from a torch.profiler.profile context manager.

    This is the public entry point used by the transformers backend.
    """
    per_bucket_us: dict[str, float] = {b: 0.0 for b in BUCKETS}
    per_bucket_count: dict[str, int] = {b: 0 for b in BUCKETS}
    events: list[CategorizedEvent] = []
    total_us = 0.0

    for ev in _iter_kineto_events(profiler):
        name = getattr(ev, "key", None) or getattr(ev, "name", "")
        cat = str(getattr(ev, "category", "") or "")
        device = str(getattr(ev, "device_type", "") or "")
        dur = float(getattr(ev, "duration", 0.0) or 0.0)
        bucket = _bucket_for(name, cat, device)
        if bucket is None:
            continue
        per_bucket_us[bucket] += dur
        per_bucket_count[bucket] += 1
        total_us += dur
        events.append(CategorizedEvent(
            name=name, bucket=bucket, duration_us=dur,
            category=cat, device=device,
        ))

    return Breakdown(
        total_us=total_us,
        per_bucket_us=per_bucket_us,
        per_bucket_count=per_bucket_count,
        events=events,
    )


# --------------------------------------------------------------------------- #
# Gap detection
# --------------------------------------------------------------------------- #

def compute_gpu_idle_gaps(events: list[dict]) -> list[CategorizedEvent]:
    """Compute GPU idle gaps between consecutive GPU events on the same stream.

    Each input event should have `start_us` and `end_us` (in microseconds)
    and a `stream` identifier (or `device == "cuda"` is sufficient if only
    one stream is in use).  Returns synthetic CategorizedEvent entries
    with bucket="gpu_idle_gap" -- the caller adds them to the breakdown
    so the chart picks them up automatically.
    """
    # Keep only GPU-side timed events with valid start/end.
    cuda = [
        e for e in events
        if (e.get("device") == "cuda" or e.get("category") == "kernel")
        and e.get("end_us", 0) > 0 and e.get("start_us", 0) >= 0
    ]
    if not cuda:
        return []

    # Group by stream.
    by_stream: dict[str, list[dict]] = defaultdict(list)
    for e in cuda:
        by_stream[e.get("stream", "default")].append(e)

    gaps: list[CategorizedEvent] = []
    for stream, evs in by_stream.items():
        evs.sort(key=lambda x: x["start_us"])
        for prev, nxt in zip(evs, evs[1:]):
            gap_start = prev["end_us"]
            gap_end = nxt["start_us"]
            if gap_end > gap_start:
                gaps.append(CategorizedEvent(
                    name=f"idle_gap[{stream}]",
                    bucket="gpu_idle_gap",
                    duration_us=gap_end - gap_start,
                    category="idle_gap",
                    device="cuda",
                    start_us=gap_start,
                    end_us=gap_end,
                    stream=stream,
                ))
    return gaps


def categorize_dicts(events: list[dict]) -> Breakdown:
    """Same as categorize() but accepts plain dicts.

    Useful for synthetic traces, JSON-loaded traces, or testing.
    Each dict must have at least 'name'; 'category', 'device', 'duration_us'
    are optional.  If 'start_us' and 'end_us' are present, GPU idle gaps
    are computed automatically and added to the breakdown.
    """
    per_bucket_us: dict[str, float] = {b: 0.0 for b in BUCKETS}
    per_bucket_count: dict[str, int] = {b: 0 for b in BUCKETS}
    categorized: list[CategorizedEvent] = []
    total_us = 0.0

    for ev in events:
        name = ev.get("name", "")
        cat = ev.get("category", "")
        device = ev.get("device", "")
        dur = float(ev.get("duration_us", 0.0) or 0.0)
        bucket = _bucket_for(name, cat, device)
        if bucket is None:
            continue
        per_bucket_us[bucket] += dur
        per_bucket_count[bucket] += 1
        total_us += dur
        categorized.append(CategorizedEvent(
            name=name, bucket=bucket, duration_us=dur,
            category=cat, device=device,
            start_us=float(ev.get("start_us", 0.0) or 0.0),
            end_us=float(ev.get("end_us", 0.0) or 0.0),
            stream=ev.get("stream", ""),
            synthetic=bool(ev.get("synthetic", False)),
        ))

    # Add GPU idle gaps if start/end times are present.
    has_times = any(e.get("start_us") or e.get("end_us") for e in events)
    if has_times:
        gap_events = compute_gpu_idle_gaps(events)
        for g in gap_events:
            per_bucket_us["gpu_idle_gap"] += g.duration_us
            per_bucket_count["gpu_idle_gap"] += 1
            total_us += g.duration_us
            categorized.append(g)

    return Breakdown(
        total_us=total_us,
        per_bucket_us=per_bucket_us,
        per_bucket_count=per_bucket_count,
        events=categorized,
    )
