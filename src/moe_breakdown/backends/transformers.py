"""
Transformers backend -- profiles a HuggingFace MoE model (Mixtral, Qwen-MoE,
DeepSeek-MoE, DBRX, etc.) using torch.profiler.

By default we use profiler.key_averages(), which is fast and gives us
events aggregated by name.  Set MOE_BREAKDOWN_FULL_EVENTS=1 (or pass
full_events=True via YAML) to use profiler.events() instead -- this is
slower but gives per-instance start/end times so gpu_idle_gap detection
works against real GPU runs.
"""

from __future__ import annotations

import os
from typing import Optional


def profile(
    model_id: str,
    prompt: str = "The quick brown fox jumps over the lazy dog.",
    max_new_tokens: int = 32,
    n_passes: int = 3,
    warmup: int = 1,
    device: str = "auto",
    dtype: Optional[str] = None,
    trust_remote_code: bool = False,
    full_events: Optional[bool] = None,
) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Pick dtype
    if dtype is None:
        dtype = "bfloat16" if torch.cuda.is_available() else "float32"
    torch_dtype = getattr(torch, dtype)

    # Pick device
    if device == "auto":
        device_map = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_map = device

    if full_events is None:
        full_events = bool(int(os.environ.get("MOE_BREAKDOWN_FULL_EVENTS", "0")))

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    inputs = tok(prompt, return_tensors="pt").to(model.device)

    # Warm-up (not profiled)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tok.pad_token_id,
            )

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    raw_events: list[dict] = []

    with torch.no_grad():
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            with_stack=False,
        ) as prof:
            for _ in range(n_passes):
                _ = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=False, pad_token_id=tok.pad_token_id,
                )

    if full_events:
        # Per-instance events (slower, but provides start/end for gap det.)
        for ev in prof.events():
            name = getattr(ev, "name", "") or ""
            cat = str(getattr(ev, "cat", "") or "")  # Kineto category
            # device inferred from category
            device = "cuda" if cat in ("kernel", "gpu_memcpy", "communication") else "cpu"
            # Some Kineto events lack start/end on the CPU side.  Use what's
            # available.
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
                # torch.profiler exposes a stream id via the event's
                # 'tid' (CUDA stream id encoded in the trace)
                stream = f"stream{getattr(ev, 'tid', 0) % 64}"
            except Exception:
                pass
            raw_events.append({
                "name": name, "category": cat, "device": device,
                "duration_us": dur,
                "start_us": start_us, "end_us": end_us,
                "stream": stream,
            })
    else:
        # Aggregated events (fast, no start/end -- gap detection disabled)
        for ev in prof.key_averages():
            # `ev.duration` is 0 for FunctionEventAvg (it's the per-instance
            # duration that was zeroed out during aggregation).  Use the
            # self_*_time_total fields which carry the actual aggregated time.
            dur_us = float(
                getattr(ev, "self_cpu_time_total", 0.0)
                + getattr(ev, "self_device_time_total", 0.0)
            )
            if dur_us <= 0:
                continue
            # FunctionEventAvg doesn't carry a Kineto `category` directly.
            # We infer from device_type so the categorizer's rules still fire.
            dt = str(getattr(ev, "device_type", "") or "")
            device = "cuda" if "CUDA" in dt.upper() else "cpu"
            category = "kernel" if device == "cuda" else "cpu_op"
            raw_events.append({
                "name": getattr(ev, "key", None) or getattr(ev, "name", ""),
                "category": category,
                "device": device,
                "duration_us": dur_us,
            })
    return raw_events
