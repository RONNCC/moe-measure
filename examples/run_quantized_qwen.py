#!/usr/bin/env python3
"""
Profile a quantized Qwen MoE model with the framework.

Use this when you want to swap the in-tree `tiny` MoE for a real quantized
checkpoint.  It uses `transformers.AutoModelForCausalLM.from_pretrained`
with bitsandbytes 4-bit (or 8-bit) quantization, then runs the model
under torch.profiler and hands the events to the categorizer -- same
pipeline as the tiny backend, so you get the same breakdown chart.

Run:
    pip install -e ".[hf]" bitsandbytes accelerate
    python examples/run_quantized_qwen.py --model Qwen/Qwen1.5-MoE-A2.7B
    python examples/run_quantized_qwen.py --model Qwen/Qwen1.5-MoE-A2.7B --bits 8
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

sys_path = str(Path(__file__).resolve().parent.parent / "src")
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from moe_breakdown import categorize, render_chart
from moe_breakdown.report import write as write_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   help="HuggingFace model id, e.g. Qwen/Qwen1.5-MoE-A2.7B")
    p.add_argument("--bits", type=int, choices=[4, 8], default=4)
    p.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--n-passes", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--out", default="runs/quantized-qwen")
    p.add_argument("--full-events", action="store_true",
                   help="Per-instance events (slower; enables gap detection)")
    args = p.parse_args()

    print(f"[quantized-qwen] loading {args.model} with {args.bits}-bit quantization")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.bits == 4:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    inputs = tok(args.prompt, return_tensors="pt").to(model.device)

    print("[quantized-qwen] warm-up ...")
    with torch.no_grad():
        for _ in range(args.warmup):
            model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                           do_sample=False, pad_token_id=tok.pad_token_id)

    print(f"[quantized-qwen] profiling {args.n_passes} passes ...")
    activities = [torch.profiler.ProfilerActivity.CPU,
                  torch.profiler.ProfilerActivity.CUDA]
    if args.full_events:
        os.environ["MOE_BREAKDOWN_FULL_EVENTS"] = "1"

    t0 = time.perf_counter()
    with torch.no_grad():
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            with_stack=False,
            acc_events=True,
        ) as prof:
            for _ in range(args.n_passes):
                model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                               do_sample=False, pad_token_id=tok.pad_token_id)
    wall_s = time.perf_counter() - t0

    breakdown = categorize(prof)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    title = f"Quantized Qwen ({args.bits}-bit) -- {args.model}"
    chart_path = render_chart(breakdown, title=title, out_path=out_dir / "breakdown.png")
    paths = write_report(
        breakdown,
        metadata={
            "model": args.model, "quantization_bits": args.bits,
            "wall_clock_s": wall_s, "n_passes": args.n_passes,
            "max_new_tokens": args.max_new_tokens,
        },
        out_dir=out_dir,
    )

    print()
    print(f"== Quantized Qwen ({args.bits}-bit) :: {args.model} ==")
    print(f"   wall-clock        : {wall_s:6.2f} s")
    print(f"   total profiled    : {breakdown.total_us/1000:8.2f} ms")
    print(f"   chart             : {chart_path}")
    print(f"   json              : {paths['report']}")
    print()
    print("   bucket              %     time(ms)   count")
    print("   " + "-" * 55)
    import moe_breakdown as mb_pkg
    for b in mb_pkg.BUCKETS:
        pct = breakdown.percent(b)
        ms = breakdown.per_bucket_us.get(b, 0.0) / 1000.0
        cnt = breakdown.per_bucket_count.get(b, 0)
        if cnt == 0 and ms == 0:
            continue
        bar = "#" * int(round(pct / 2))
        print(f"   {b:18s} {pct:5.1f}%   {ms:8.3f}   {cnt:6d}   {bar}")


if __name__ == "__main__":
    main()
