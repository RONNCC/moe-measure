"""
Tiny backend -- profiles a tiny in-tree MoE model under torch.profiler.

This is the "does the framework work end-to-end on a real model, not just
on a synthetic trace" path.  Use it for:
  * Smoke-testing new backends / categorizer rules without a GPU or a
    big checkpoint download.
  * Reproducible demos (the synthetic traces are plausible but this is
    a *real* forward pass).
  * A baseline you can diff against a real model to see what changes
    when you swap in a quantized Qwen / Mixtral / DeepSeek.

To swap in your own model later, three options:

  1. Easiest -- use the `transformers` backend with --model <hf-id>.
     Already supported.
  2. Pass a user-supplied module by writing a thin backend (see
     `examples/run_my_own_model.py`).
  3. Replace the model class in `src/moe_breakdown/models/tiny_moe.py`
     with a quantized HuggingFace MoE -- the categorizer does not care.
"""

from __future__ import annotations

import os
from typing import Optional

import torch


def profile(
    n_passes: int = 5,
    warmup: int = 2,
    seed: int = 0,
    full_events: Optional[bool] = None,
    vocab_size: int = 1024,
    hidden_dim: int = 256,
    expert_hidden_dim: int = 2048,
    num_experts: int = 3,
    top_k: int = 1,
    seq_len: int = 32,
    batch_size: int = 1,
) -> list[dict]:
    """Run the tiny MoE model under torch.profiler and return event dicts."""
    from ..models.tiny_moe import TinyMoE, TinyMoEConfig, build

    cfg = TinyMoEConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        expert_hidden_dim=expert_hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
        seq_len=seq_len,
        batch_size=batch_size,
    )
    model = build(cfg)
    inputs = TinyMoE.make_inputs(cfg, seed=seed)

    # Pin to a single CPU thread so cpu_op events are not interleaved
    # across cores (makes the breakdown easier to read).
    torch.set_num_threads(1)

    # Warm-up (untimed) -- first forward can be much slower than steady-state.
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(inputs)

    if full_events is None:
        full_events = bool(int(os.environ.get("MOE_BREAKDOWN_FULL_EVENTS", "0")))

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    raw_events: list[dict] = []

    with torch.no_grad():
        # acc_events=True keeps events across multiple `step()` calls so
        # multi-pass profiling aggregates correctly.  This is the default
        # behaviour when using a scheduler; for plain `with profile(...)`
        # we have to opt in.
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            with_stack=False,
            acc_events=True,
        ) as prof:
            for _ in range(n_passes):
                _ = model(inputs)

    if full_events:
        for ev in prof.events():
            raw_events.append(_ev_to_dict(ev))
    else:
        # Use self_cpu_time_total (NOT `duration` -- that's 0 for aggregates).
        # self_cpu_time_total gives self-time excluding children, which is
        # what we want for "where did the time go" categorisation.
        for ev in prof.key_averages():
            dur_us = float(
                getattr(ev, "self_cpu_time_total", 0.0)
                + getattr(ev, "self_device_time_total", 0.0)
            )
            if dur_us <= 0:
                continue
            raw_events.append({
                "name": getattr(ev, "key", None) or getattr(ev, "name", ""),
                # Categorization heuristics read `category`.  For CPU-only
                # key-averages there's no Kineto category string, so we
                # tag cpu_op / kernel manually from device_type.
                "category": _infer_category(ev),
                "device": _infer_device(ev),
                "duration_us": dur_us,
                "count": int(getattr(ev, "count", 1)),
            })
    return raw_events


def _infer_category(ev) -> str:
    """Map a FunctionEventAvg to a Kineto-style category string.

    FunctionEventAvg does not carry a `cat` attribute, so we infer from
    device_type.  The categorizer's bucket rules read `category` first,
    then fall back to regex on the name -- so even a wrong category only
    matters when name patterns would otherwise have routed differently.
    """
    dev = _infer_device(ev)
    if dev == "cuda":
        return "kernel"
    # CPU: we don't know if it's aten::, Python, or runtime.  Default to
    # cpu_op (the safest -- categorizer will still split cpu_python vs
    # cpu_native by name pattern).
    return "cpu_op"


def _infer_device(ev) -> str:
    dt = getattr(ev, "device_type", None)
    if dt is None:
        return "cpu"
    s = str(dt)
    # DeviceType.CPU -> "cpu", DeviceType.CUDA -> "cuda"
    s = s.split(".")[-1].lower() if "." in s else s.lower()
    return "cuda" if "cuda" in s else "cpu"


def _ev_to_dict(ev) -> dict:
    """Convert a per-instance FunctionEvent to a dict (for full_events mode)."""
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
    try:
        stream = f"stream{getattr(ev, 'tid', 0) % 64}"
    except Exception:
        stream = ""
    return {
        "name": name, "category": cat, "device": device,
        "duration_us": dur,
        "start_us": start_us, "end_us": end_us,
        "stream": stream,
    }
