#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fused_moe_kernel_study import MoEKernelConfig, measure_moe_kernel_latency


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Measure one direct fused-MoE kernel condition")
    p.add_argument("--shape-name", required=True)
    p.add_argument("--hidden-size", type=int, required=True)
    p.add_argument("--intermediate-size", type=int, required=True)
    p.add_argument("--num-experts", type=int, required=True)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--activation", default="silu")
    p.add_argument("--all2all-backend", default="deepep_low_latency")
    p.add_argument("--tp-size", type=int, required=True)
    p.add_argument("--ep-size", type=int, required=True)
    p.add_argument("--num-tokens", type=int, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--warmup-iters", type=int, default=20)
    p.add_argument("--measure-iters", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hot-expert-count", type=int, default=1)
    p.add_argument("--routing-weight-mode", default="uniform")
    p.add_argument("--collect-buckets", action="store_true")
    p.add_argument("--bucket-profile-iters", type=int, default=5)
    p.add_argument("--bucket-full-events", action="store_true")
    p.add_argument("--output-root", default="runs")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)

    config = MoEKernelConfig(
        shape_name=args.shape_name,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_experts=args.num_experts,
        topk=args.topk,
        dtype=args.dtype,
        activation=args.activation,
        all2all_backend=args.all2all_backend,
        tp_size=args.tp_size,
        ep_size=args.ep_size,
        num_tokens=args.num_tokens,
        alpha=args.alpha,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        seed=args.seed,
        hot_expert_count=args.hot_expert_count,
        routing_weight_mode=args.routing_weight_mode,
        collect_buckets=args.collect_buckets,
        bucket_profile_iters=args.bucket_profile_iters,
        bucket_full_events=args.bucket_full_events,
        output_root=args.output_root,
        out_dir=args.out_dir,
    )
    measurement = measure_moe_kernel_latency(config)
    payload = asdict(measurement)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
