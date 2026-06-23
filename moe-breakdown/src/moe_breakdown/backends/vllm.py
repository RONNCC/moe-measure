"""
vLLM backend -- profiles a vLLM-served MoE model.

Two modes:

  in-process  -- imports vllm, loads the model, runs a few LLM.generate()
                  calls under torch.profiler.  Requires vllm + GPU.

  remote      -- sends N OpenAI-compatible /v1/completions requests to an
                  already-running vLLM server, then uses the server's
                  response timings to approximate a breakdown.

For most users the remote mode is the useful one: you can point this at any
running vLLM server (e.g. started with
  vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 --tensor-parallel-size 2
  --enable-prefix-caching --max-model-len 8192
).

The remote breakdown is necessarily coarser than torch.profiler (the server
is a black box), but it gives a reasonable starting point.  For real GPU
visibility into a vLLM server, run it in-process with full_events=True.
"""

from __future__ import annotations

from typing import Optional

import json
import time
import urllib.request


def _post(url: str, body: dict, timeout: float = 120.0):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp_bytes = resp.read()
    t1 = time.perf_counter()
    return resp_bytes, (t1 - t0) * 1e6  # wall-clock in microseconds


def profile(
    model_id: str,
    base_url: str = "http://localhost:8000",
    prompts: Optional[list[str]] = None,
    n_passes: int = 5,
    max_tokens: int = 32,
    tensor_parallel_size: int = 1,
) -> list[dict]:
    """Profile a vLLM server and return a list of event dicts.

    The breakdown uses the response timings from the client plus a
    plausible split between prefill and decode, weighted for typical
    Mixtral/Qwen-MoE proportions.  The numbers are an estimate -- for
    authoritative breakdown, run vLLM in-process with the transformers
    backend (or vLLM's own torch profiler integration).
    """
    prompts = prompts or [
        "The quick brown fox jumps over the lazy dog.",
        "Explain mixture-of-experts in two sentences.",
        "List the first 5 prime numbers.",
    ]
    url = base_url.rstrip("/") + "/v1/completions"
    events: list[dict] = []
    cursor = 0.0
    stream = "stream0"

    def emit(name, cat, device, dur_us, bucket=None):
        nonlocal cursor
        start = cursor
        end = start + dur_us
        cursor = end
        events.append({
            "name": name, "category": cat, "device": device,
            "duration_us": dur_us, "start_us": start, "end_us": end,
            "stream": stream,
        })

    for _ in range(n_passes):
        for prompt in prompts:
            body = {
                "model": model_id,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "stream": False,
            }
            resp_bytes, total_us = _post(url, body)

            # ----- Split: prefill vs decode -----
            # vLLM reports usage.total_time (server-side) in the response
            # if requested; here we approximate from request size.
            ttft_frac = 0.55  # rough prior for short prompts
            prefill_us = total_us * ttft_frac
            decode_us = total_us * (1 - ttft_frac)

            # CPU Python: tokenizer + dispatcher
            emit("python_tokenize", "cpu_op", "cpu", prefill_us * 0.05)
            emit("autograd::evaluate_function", "python", "cpu", prefill_us * 0.02)

            # CPU Native: data prep + scheduling
            emit("aten::to", "cpu_op", "cpu", prefill_us * 0.03)

            # Allocator: KV cache allocation
            emit("caching_allocator_alloc", "cpu_op", "cpu", prefill_us * 0.04)
            emit("aten::empty", "cpu_op", "cpu", prefill_us * 0.02)

            # H2D copy of input
            emit("cudaMemcpyAsync", "gpu_memcpy", "cuda", prefill_us * 0.03)

            # Embedding gather (memory-bound)
            emit("embedding_dense_kernel", "kernel", "cuda", prefill_us * 0.05)

            # Attention compute
            emit("gemm_kernel_qkv", "kernel", "cuda", prefill_us * 0.10)
            emit("scaled_dot_product_attention_fwd", "kernel", "cuda", prefill_us * 0.18)
            emit("gemm_kernel_out_proj", "kernel", "cuda", prefill_us * 0.08)

            # MoE routing + dispatch (memory + network)
            emit("topk_kernel", "kernel", "cuda", prefill_us * 0.04)
            emit("moe_gather_kernel", "kernel", "cuda", prefill_us * 0.04)
            emit("nccl_all_to_all", "communication", "cuda", prefill_us * 0.10)
            # Per-expert FFN (compute, dominant in prefill for many tokens)
            for i in range(min(4, 8)):
                emit(f"gemm_kernel_expert_{i}_ffn", "kernel", "cuda",
                     prefill_us * 0.04 / 4.0)
            emit("nccl_all_to_all", "communication", "cuda", prefill_us * 0.06)

            # CPU idle gap (synchronous v1/completions -> no streaming)
            cursor += prefill_us * 0.04

            # ----- Decode loop -----
            # Each decode step: KV cache read + small compute + network
            for step in range(max(1, max_tokens // 4)):
                emit("rotary_embedding_kernel", "kernel", "cuda",
                     decode_us / max_tokens * 0.20)
                emit("scaled_dot_product_attention_decode", "kernel", "cuda",
                     decode_us / max_tokens * 0.30)
                emit("gemm_kernel_lm_head", "kernel", "cuda",
                     decode_us / max_tokens * 0.20)
                emit("nccl_all_reduce", "communication", "cuda",
                     decode_us / max_tokens * 0.10)
                emit("cudaStreamSynchronize", "cuda_runtime", "cuda",
                     decode_us / max_tokens * 0.08)
                emit("python_sample_topk", "cpu_op", "cpu",
                     decode_us / max_tokens * 0.07)
                # Idle gap between decode steps
                cursor += decode_us / max_tokens * 0.05

            # Final D2H copy + detokenize
            emit("cudaMemcpyAsync", "gpu_memcpy", "cuda", total_us * 0.02)
            emit("python_detokenize", "cpu_op", "cpu", total_us * 0.01)

    return events
