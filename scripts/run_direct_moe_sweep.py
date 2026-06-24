#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fused_moe_kernel_study import run_sweep


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Direct fused-MoE kernel study for one fixed TP/EP point")
    p.add_argument("--config", required=True, help="Study YAML config")
    p.add_argument("--tp-size", type=int, required=True)
    p.add_argument("--ep-size", type=int, required=True)
    p.add_argument("--out-dir", default=None, help="Override output directory")
    args = p.parse_args(argv)

    run_sweep(config=args.config, tp_size=args.tp_size, ep_size=args.ep_size, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
