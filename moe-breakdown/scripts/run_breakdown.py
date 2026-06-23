"""
moe-breakdown CLI.

Usage examples:

  # Run a quick demo (no GPU / no model needed) -- outputs go to
  # runs/synthetic-<timestamp>/ by default:
  python scripts/run_breakdown.py --backend synthetic

  # Profile a HuggingFace MoE model (Mixtral, Qwen-MoE, DeepSeek-MoE, ...):
  python scripts/run_breakdown.py --backend transformers \\
      --model mistralai/Mixtral-8x7B-Instruct-v0.1 --tokens 32 \
      --out runs/mixtral-8x7b-bf16

  # Profile a running vLLM server:
  python scripts/run_breakdown.py --backend vllm \\
      --model mistralai/Mixtral-8x7B-Instruct-v0.1 \\
      --base-url http://localhost:8000 --out runs/vllm_mixtral
"""

from __future__ import annotations

import argparse

import numpy as np
import json
import os
import sys
import time
from pathlib import Path

# Make the package importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml
from moe_breakdown import categorize, categorize_dicts, render_chart
from moe_breakdown.backends import available, get
from moe_breakdown.report import write as write_report


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _detect_hardware() -> dict:
    """Return a dict describing the local compute environment.

    The CLI prints this at startup so it's always obvious whether the run
    was on GPU or CPU -- no silent fallback surprises.
    """
    import torch
    info = {
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_devices": [],
        "cpu_threads": int(torch.get_num_threads()),
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["cuda_devices"].append({
                "index": i,
                "name": props.name,
                "memory_gb": round(props.total_memory / 1024**3, 1),
            })
    return info


def _print_hardware_banner(hw: dict, backend: str, model: str | None) -> None:
    """Print a one-liner at startup so the user knows what's running."""
    if hw["cuda_available"]:
        names = ", ".join(f"{d['name']} ({d['memory_gb']}GB)" for d in hw["cuda_devices"])
        print(f"[hw] CUDA detected: {hw['cuda_device_count']} GPU(s) -> {names}")
        print(f"[hw] Backend: {backend}" + (f" / model: {model}" if model else ""))
    else:
        print(f"[hw] WARNING: no CUDA GPU detected -> falling back to CPU")
        print(f"[hw] CPU threads: {hw['cpu_threads']}")
        print(f"[hw] Backend: {backend}" + (f" / model: {model}" if model else ""))
        print(f"[hw] NOTE: GPU-only buckets (gpu_compute, gpu_memory, gpu_idle_gap,")
        print(f"[hw]       network, mem_transfer) will be empty on CPU runs.")
        print(f"[hw]       Pass --hybrid to layer a synthetic GPU projection on top,")
        print(f"[hw]       or run on a GPU box for ground-truth numbers.")
        print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="moe-breakdown",
        description="Categorize profiler events of a MoE model into CPU / GPU / Network / Memory-Transfer buckets.",
    )
    p.add_argument("--backend", choices=available(), default="synthetic")
    p.add_argument("--config", help="YAML config file (optional)")
    p.add_argument("--out", default=None,
                   help="output directory (default: runs/<backend>-<timestamp>/)")
    p.add_argument("--title", default=None)

    # transformers args
    p.add_argument("--model", help="HF model id (transformers / vllm backends)")
    p.add_argument("--base-url", default="http://localhost:8000", help="vLLM base URL")
    p.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--hybrid", action="store_true",
                   help="(transformers backend) layer a synthetic GPU "
                        "projection on top of the real CPU run so all "
                        "9 buckets are populated")
    # Topology / expert-placement analysis
    p.add_argument("--topology", action="store_true",
                   help="After the breakdown, run expert-placement analysis "
                        "(transfer matrix + placement visualizations)")
    p.add_argument("--num-experts", type=int, default=None,
                   help="Override expert count for topology analysis "
                        "(default: read from model config)")
    p.add_argument("--num-racks", type=int, default=2,
                   help="Number of racks in the cluster (default 2)")
    p.add_argument("--gpus-per-rack", type=int, default=8,
                   help="GPUs per rack (default 8)")
    p.add_argument("--strategies", default="round-robin,greedy,cluster",
                   help="Comma-separated placement strategies to compare")

    args = p.parse_args(argv)
    cfg = load_config(args.config)

    # Print hardware detection banner before any work
    hw = _detect_hardware()
    _print_hardware_banner(hw, args.backend, args.model if hasattr(args, "model") else None)

    # If --out not given, auto-name as runs/<backend>-<YYYY-MM-DD-HHMM>/
    if args.out is None:
        ts = time.strftime("%Y-%m-%d-%H%M")
        args.out = f"runs/{args.backend}-{ts}"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "backend": args.backend,
        "hybrid": bool(getattr(args, "hybrid", False)),
        "argv": sys.argv[1:],
        "config_path": args.config,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    profile_fn = get(args.backend)

    # Dispatch on backend
    if args.backend == "synthetic":
        from moe_breakdown.backends.synthetic import SyntheticConfig
        s_cfg = SyntheticConfig(**(cfg.get("synthetic") or {}))
        raw_events = profile_fn(s_cfg)
    elif args.backend == "tiny":
        # Build kwargs from config file's "tiny" section, then overlay CLI.
        tiny_cfg = cfg.get("tiny") or {}
        tiny_kwargs = dict(
            n_passes=args.passes,
            warmup=args.warmup,
            seed=tiny_cfg.get("seed", 0),
            vocab_size=tiny_cfg.get("vocab_size", 1024),
            hidden_dim=tiny_cfg.get("hidden_dim", 256),
            expert_hidden_dim=tiny_cfg.get("expert_hidden_dim", 2048),
            num_experts=tiny_cfg.get("num_experts", 3),
            top_k=tiny_cfg.get("top_k", 1),
            seq_len=tiny_cfg.get("seq_len", 32),
            batch_size=tiny_cfg.get("batch_size", 1),
        )
        raw_events = profile_fn(**tiny_kwargs)
    elif args.backend == "transformers":
        raw_events = profile_fn(
            model_id=args.model or cfg.get("model"),
            prompt=args.prompt,
            max_new_tokens=args.tokens,
            n_passes=args.passes,
            warmup=args.warmup,
        )
        if args.hybrid:
            # Layer a synthetic GPU projection on top of the real CPU run
            # so the chart shows all 9 buckets.  Real CPU events stay
            # untouched; the synthetic events are emitted on the same
            # wall-clock timeline.  Synthetic events are tagged with
            # "synthetic": True so they can be filtered out later.
            from moe_breakdown.backends.synthetic import (
                calibrated_from_model_config, profile as synth_profile,
            )
            from transformers import AutoConfig
            model_id = args.model or cfg.get("model")
            model_config = AutoConfig.from_pretrained(model_id)
            synth_cfg = calibrated_from_model_config(model_config)
            synth_events = synth_profile(synth_cfg)
            for e in synth_events:
                e["synthetic"] = True
            raw_events = raw_events + synth_events
    elif args.backend == "vllm":
        raw_events = profile_fn(
            model_id=args.model or cfg.get("model"),
            base_url=args.base_url,
            max_tokens=args.tokens,
            n_passes=args.passes,
        )
    else:
        raise SystemExit(f"unhandled backend {args.backend}")

    if not raw_events:
        print(f"[moe-breakdown] backend {args.backend!r} produced 0 events", file=sys.stderr)
        return 1

    breakdown = categorize_dicts(raw_events)

    # Chart
    suffix = " (hybrid CPU+synthetic GPU)" if getattr(args, "hybrid", False) else ""
    title = args.title or f"MoE execution breakdown -- {args.backend}{suffix}" + (
        f" ({args.model})" if args.model else ""
    )
    chart_path = render_chart(breakdown, title=title, out_path=out_dir / "breakdown.png")

    # Reports
    paths = write_report(breakdown, metadata=metadata, out_dir=out_dir)

    # Optional: topology / expert-placement analysis
    if args.topology:
        from moe_breakdown.topology import (
            extract_from_routing, place_round_robin, place_greedy, place_cluster,
            evaluate, PLACEMENT_STRATEGIES, TransferMatrix, Placement,
        )
        from moe_breakdown.topology_chart import (
            render_transfer_matrix, render_topology,
            render_placement_comparison,
        )
        from moe_breakdown.report import write as _write

        # Build the transfer matrix.  On CPU we can't measure real AllToAll,
        # so synthesize a routing pattern from the model config.
        N = args.num_experts
        if N is None and args.backend in ("transformers", "tiny"):
            try:
                from transformers import AutoConfig
                mcfg = AutoConfig.from_pretrained(args.model or cfg.get("model", ""))
                N = int(getattr(mcfg, "num_local_experts", 8))
            except Exception:
                N = 8
        if N is None:
            N = 8  # fallback

        # Synthetic routing: top-K with a few "hot" experts that absorb
        # most tokens.  This mimics the typical MoE serving pattern where
        # a handful of "shared" or popular experts get 80% of traffic.
        rng = np.random.default_rng(0)
        T = 65536              # ~64k tokens per batch (production-sized)
        K = 2                  # top-2 routing
        routing = np.zeros((T, K), dtype=np.int64)
        hot = rng.choice(N, size=max(2, N // 10), replace=False)
        for t in range(T):
            if rng.random() < 0.8:
                routing[t] = rng.choice(hot, size=K, replace=False)
            else:
                routing[t] = rng.choice(N, size=K, replace=(K > N))

        # 8 KB per activation message (hidden_dim=2048 x bf16).  Multiplied
        # across all expert pairs this gives GB-scale totals, matching
        # what a real AllToAll between 100 experts would move.
        matrix = extract_from_routing(
            routing,
            tokens_per_msg_bytes=8192,
            intra_rack_bw_gbps=200.0,
            inter_rack_bw_gbps=25.0,
            num_racks=args.num_racks,
            gpus_per_rack=args.gpus_per_rack,
        )

        # Run each requested strategy.
        strategy_names = [s.strip() for s in args.strategies.split(",") if s.strip()]
        placements = {n: PLACEMENT_STRATEGIES[n](matrix, args.num_racks, args.gpus_per_rack)
                      for n in strategy_names if n in PLACEMENT_STRATEGIES}
        costs = {n: evaluate(matrix, p) for n, p in placements.items()}

        # Render charts (transfer matrix produces TWO files: time + bytes).
        topo_dir = out_dir
        tm_paths = render_transfer_matrix(
            matrix, out_path_base=topo_dir / "transfer_matrix",
            title=f"Transfer matrix -- {N} experts",
        )
        # Use the first strategy as the headline placement visualization.
        headline = "cluster" if "cluster" in placements else strategy_names[0]
        p_headline = placements[headline]
        c_headline = costs[headline]
        render_topology(matrix, p_headline, c_headline,
                        out_path=topo_dir / "topology.png",
                        title=f"Expert placement ({headline})")
        render_placement_comparison(matrix, placements,
                                    out_path=topo_dir / "placement_comparison.png",
                                    title=f"Placement comparison -- {N} experts, "
                                          f"{args.num_racks} racks x {args.gpus_per_rack} GPUs")

        # Save placement JSON.
        import json
        placement_data = {
            "matrix_total_us": matrix.total_us,
            "num_experts": N,
            "num_racks": args.num_racks,
            "gpus_per_rack": args.gpus_per_rack,
            "strategies": {
                n: {"placement": p.as_dict(), "cost": costs[n].as_dict()}
                for n, p in placements.items()
            },
        }
        with open(topo_dir / "placement.json", "w") as f:
            json.dump(placement_data, f, indent=2)

        # Print summary
        print()
        print(f"== topology :: {N} experts, {args.num_racks} racks x {args.gpus_per_rack} GPUs ==")
        print(f"   total transfer TIME : {matrix.total_us/1e3:7.2f} ms")
        print(f"   total transfer DATA : {matrix.total_gbs:7.2f} GB  ({matrix.total_bytes:,} bytes)")
        print()
        print("   strategy         intra(ms)  inter(ms)  total(ms)")
        print("   " + "-" * 50)
        for n in strategy_names:
            if n not in costs:
                continue
            c = costs[n]
            print(f"   {n:14s}   {c.intra_rack_us/1e3:7.2f}    {c.inter_rack_us/1e3:7.2f}    {c.total_us/1e3:7.2f}")
        print()
        for p in tm_paths:
            print(f"   {p.name:32s} : {p}")
        print(f"   topology.png ({headline:8s}) : {topo_dir / 'topology.png'}")
        print(f"   placement_comparison.png   : {topo_dir / 'placement_comparison.png'}")
        print(f"   placement.json             : {topo_dir / 'placement.json'}")
        print()

    # Console summary
    import moe_breakdown as mb_pkg
    print(f"\n== moe-breakdown :: {args.backend} ==")
    print(f"   total profiled time : {breakdown.total_us/1000.0:8.2f} ms  ({breakdown.total_us/1e6:6.3f} s)")
    print(f"   events              : {len(breakdown.events):8d}")
    print(f"   chart               : {chart_path}")
    print(f"   json                : {paths['report']}")
    print(f"   csv                 : {paths['csv']}")
    print()
    print("   bucket              %     time(ms)   count")
    print("   " + "-" * 55)
    for b in mb_pkg.BUCKETS:
        pct = breakdown.percent(b)
        ms = breakdown.per_bucket_us.get(b, 0.0) / 1000.0
        cnt = breakdown.per_bucket_count.get(b, 0)
        bar = "#" * int(round(pct / 2))
        label = mb_pkg.BUCKET_STYLE[b]["label"]
        print(f"   {b:18s} {pct:5.1f}%   {ms:8.3f}   {cnt:6d}   {bar}  ({label})")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
