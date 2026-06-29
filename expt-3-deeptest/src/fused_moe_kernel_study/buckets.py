from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any


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


_BUCKET_LABELS = {
    "cpu_python": "CPU — Python",
    "cpu_native": "CPU — Native",
    "gpu_compute": "GPU Compute",
    "gpu_memory": "GPU Memory-bound",
    "gpu_idle_gap": "GPU Idle — gap",
    "gpu_idle_sync": "GPU Idle — sync",
    "network": "Network",
    "mem_transfer": "Mem Transfer",
    "allocator": "Allocator",
}


_PYTHON_RE = re.compile(r"^python|^pyeval|^pyobject|^dispatch_key|autograd|^grad|^optimizer|^profiler|^record_function")
_ALLOC_RE = re.compile(r"cudamalloc|cudafree|caching_allocator|atencachingallocator|storageimpl|\baten::empty\b|\baten::zeros\b|\baten::ones\b|\baten::full\b|\baten::to\b.*\bdevice\b")
_COMPUTE_RE = re.compile(r"\bgemm\b|\bmatmul\b|\bmm\b|\bbmm\b|\bconv\b|convolution|\battention\b|scaled_dot_product|\bsdpa\b|\bflash\b|\bfwd\b|\bbackward\b|\bbwd\b|\bcutlass\b|\bcublas\b|fused.*attention")
_MEMORY_RE = re.compile(r"elementwise|vectorized|copy_kernel|(?<![a-z0-9_])copy(?![a-z0-9_])|(?<![a-z0-9_])cast(?![a-z0-9_])|(?<![a-z0-9_])nonzero(?![a-z0-9_])|(?<![a-z0-9_])norm(?![a-z0-9_])|rmsnorm|layernorm|groupnorm|softmax|(?<![a-z0-9_])silu(?![a-z0-9_])|(?<![a-z0-9_])gelu(?![a-z0-9_])|(?<![a-z0-9_])relu(?![a-z0-9_])|(?<![a-z0-9_])sigmoid(?![a-z0-9_])|(?<![a-z0-9_])tanh(?![a-z0-9_])|(?<![a-z0-9_])index(?![a-z0-9_])|(?<![a-z0-9_])gather(?![a-z0-9_])|(?<![a-z0-9_])scatter(?![a-z0-9_])|(?<![a-z0-9_])embedding(?![a-z0-9_])|(?<![a-z0-9_])topk(?![a-z0-9_])|(?<![a-z0-9_])sort(?![a-z0-9_])|moe_|(?<![a-z0-9_])one_hot(?![a-z0-9_])|(?<![a-z0-9_])mask(?![a-z0-9_])|(?<![a-z0-9_])cumsum(?![a-z0-9_])|cat_|(?<![a-z0-9_])stack(?![a-z0-9_])|(?<![a-z0-9_])split(?![a-z0-9_])|(?<![a-z0-9_])reshape(?![a-z0-9_])|(?<![a-z0-9_])view(?![a-z0-9_])|(?<![a-z0-9_])permute(?![a-z0-9_])|(?<![a-z0-9_])transpose(?![a-z0-9_])|kv_cache|rotary|(?<![a-z0-9_])rope(?![a-z0-9_])")
_NETWORK_RE = re.compile(r"allreduce|all_reduce|allgather|all_gather|alltoall|all_to_all|reducescatter|reduce_scatter|\bbroadcast\b|\bnccl\b|^send$|^recv$")
_MEMXFER_RE = re.compile(r"memcpy|memset|h2d|d2h|d2d|pcie")
_SYNC_RE = re.compile(r"streamsynchronize|eventsynchronize|cudadevicesynchronize|cudastreamwaitevent|\baten::item\b|\baten::cpu\b|\baten::numpy\b|\baten::tolist\b")


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
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Breakdown:
    total_us: float
    per_bucket_us: dict[str, float]
    per_bucket_count: dict[str, int]
    events: list[CategorizedEvent]

    def percent(self, bucket: str) -> float:
        if self.total_us <= 0:
            return 0.0
        return 100.0 * self.per_bucket_us.get(bucket, 0.0) / self.total_us

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_us": self.total_us,
            "per_bucket_us": self.per_bucket_us,
            "per_bucket_count": self.per_bucket_count,
            "per_bucket_pct": {b: self.percent(b) for b in BUCKETS},
            "labels": _BUCKET_LABELS,
            "n_events": len(self.events),
        }


def _bucket_for(name: str, cat: str, device: str) -> str | None:
    n = (name or "").lower()
    cat = (cat or "").lower()
    device = (device or "").lower()

    if _SYNC_RE.search(n) or cat == "sync":
        return "gpu_idle_sync"
    if cat == "communication" or _NETWORK_RE.search(n):
        return "network"
    if _ALLOC_RE.search(n) and not _MEMXFER_RE.search(n):
        return "allocator"
    if cat in ("memcpy", "memset", "gpu_memcpy") or _MEMXFER_RE.search(n):
        return "mem_transfer"
    if cat == "cuda_runtime":
        return "cpu_native"
    if device == "cpu" or cat == "cpu_op":
        if _PYTHON_RE.search(n) or cat == "python":
            return "cpu_python"
        return "cpu_native"
    if cat == "kernel" and device == "cuda":
        if _MEMORY_RE.search(n):
            return "gpu_memory"
        if _COMPUTE_RE.search(n):
            return "gpu_compute"
        return "gpu_compute"
    return None


def compute_gpu_idle_gaps(events: list[dict[str, Any]]) -> list[CategorizedEvent]:
    cuda_events = [
        e for e in events
        if (e.get("device") == "cuda" or e.get("category") == "kernel")
        and e.get("end_us", 0) > 0
    ]
    by_stream: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in cuda_events:
        by_stream[e.get("stream", "default")].append(e)
    gaps: list[CategorizedEvent] = []
    for stream, evs in by_stream.items():
        evs.sort(key=lambda x: x["start_us"])
        for prev, nxt in zip(evs, evs[1:]):
            if nxt["start_us"] > prev["end_us"]:
                gaps.append(
                    CategorizedEvent(
                        name=f"idle_gap[{stream}]",
                        bucket="gpu_idle_gap",
                        duration_us=float(nxt["start_us"] - prev["end_us"]),
                        category="idle_gap",
                        device="cuda",
                        start_us=float(prev["end_us"]),
                        end_us=float(nxt["start_us"]),
                        stream=stream,
                    )
                )
    return gaps


def categorize_dicts(events: list[dict[str, Any]]) -> Breakdown:
    per_bucket_us = {b: 0.0 for b in BUCKETS}
    per_bucket_count = {b: 0 for b in BUCKETS}
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
        categorized_ev = CategorizedEvent(
            name=name,
            bucket=bucket,
            duration_us=dur,
            category=str(cat),
            device=str(device),
            start_us=float(ev.get("start_us", 0.0) or 0.0),
            end_us=float(ev.get("end_us", 0.0) or 0.0),
            stream=str(ev.get("stream", "") or ""),
        )
        categorized.append(categorized_ev)
        per_bucket_us[bucket] += dur
        per_bucket_count[bucket] += 1
        total_us += dur

    if any(e.get("start_us") or e.get("end_us") for e in events):
        for gap in compute_gpu_idle_gaps(events):
            categorized.append(gap)
            per_bucket_us[gap.bucket] += gap.duration_us
            per_bucket_count[gap.bucket] += 1
            total_us += gap.duration_us

    return Breakdown(total_us=total_us, per_bucket_us=per_bucket_us, per_bucket_count=per_bucket_count, events=categorized)


def profiler_to_event_dicts(profiler: Any, full_events: bool = True) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if full_events:
        for ev in profiler.events():
            name = getattr(ev, "name", "") or ""
            cat = str(getattr(ev, "cat", "") or "")
            device = "cuda" if cat in ("kernel", "gpu_memcpy", "communication") else "cpu"
            try:
                start_us = float(ev.time_range.start)
            except Exception:
                start_us = 0.0
            try:
                end_us = float(ev.time_range.end)
            except Exception:
                end_us = start_us
            dur = max(0.0, end_us - start_us)
            stream = ""
            try:
                stream = f"stream{getattr(ev, 'tid', 0) % 64}"
            except Exception:
                pass
            events.append({
                "name": name,
                "category": cat,
                "device": device,
                "duration_us": dur,
                "start_us": start_us,
                "end_us": end_us,
                "stream": stream,
            })
    else:
        for ev in profiler.key_averages():
            dur_us = float(
                getattr(ev, "self_cpu_time_total", 0.0)
                + getattr(ev, "self_device_time_total", 0.0)
            )
            if dur_us <= 0:
                continue
            dt = str(getattr(ev, "device_type", "") or "")
            device = "cuda" if "CUDA" in dt.upper() else "cpu"
            category = "kernel" if device == "cuda" else "cpu_op"
            events.append({
                "name": getattr(ev, "key", None) or getattr(ev, "name", ""),
                "category": category,
                "device": device,
                "duration_us": dur_us,
            })
    return events


def breakdown_to_rows(breakdown: Breakdown) -> list[dict[str, Any]]:
    rows = []
    for bucket in BUCKETS:
        rows.append(
            {
                "bucket": bucket,
                "label": _BUCKET_LABELS[bucket],
                "time_us": breakdown.per_bucket_us.get(bucket, 0.0),
                "time_ms": breakdown.per_bucket_us.get(bucket, 0.0) / 1000.0,
                "percent": breakdown.percent(bucket),
                "count": breakdown.per_bucket_count.get(bucket, 0),
            }
        )
    return rows


def breakdown_events_as_dicts(breakdown: Breakdown) -> list[dict[str, Any]]:
    return [asdict(ev) for ev in breakdown.events]
