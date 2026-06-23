#!/usr/bin/env python3
"""
Plug your own model into the framework in 5 lines.

Use this when:
  * Your model is a HuggingFace MoE that the `transformers` backend
    doesn't support directly (e.g. custom modeling code).
  * You want a custom forward loop (KV cache, speculative decoding,
    chunked prefill, etc.).
  * You're experimenting with a new architecture and want the same
    breakdown chart the rest of the team uses.

Just write your forward pass, wrap it in torch.profiler, and call
`categorize(prof)` -- everything else is the same as the tiny / vLLM
backends.
"""

from __future__ import annotations

import argparse
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
    p.add_argument("--out", default="runs/custom-model")
    p.add_argument("--n-passes", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    args = p.parse_args()

    # --- 1. Build your model -------------------------------------------------
    from moe_breakdown.models.tiny_moe import TinyMoE, TinyMoEConfig, build
    model = build(TinyMoEConfig()).eval()
    inputs = TinyMoE.make_inputs(TinyMoEConfig())

    # --- 2. Warm-up (untimed) ------------------------------------------------
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(inputs)

    # --- 3. Profile ----------------------------------------------------------
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.no_grad():
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            with_stack=False,
            acc_events=True,
        ) as prof:
            for _ in range(args.n_passes):
                _ = model(inputs)

    # --- 4. Categorize & chart (the whole point) -----------------------------
    breakdown = categorize(prof)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    chart = render_chart(breakdown, title="My own model",
                         out_path=out_dir / "breakdown.png")
    paths = write_report(
        breakdown,
        metadata={"script": "run_my_own_model.py", "n_passes": args.n_passes},
        out_dir=out_dir,
    )

    print(f"chart : {chart}")
    print(f"json  : {paths['report']}")
    print(f"csv   : {paths['csv']}")


if __name__ == "__main__":
    main()
