#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fused_moe_kernel_study.config import load_study_config


def _add_if(cmd: list[str], flag: str, value: str | None) -> None:
    if value:
        cmd.extend([flag, value])


def _normalize_path_for_export(value: str | None, default: str | None = None) -> str:
    raw = value or default or ""
    if not raw:
        return ""
    raw = os.path.expanduser(raw)
    # Preserve shell variables like $TMPDIR for expansion on the compute node.
    if "$" in raw:
        return raw
    return str(Path(os.path.expandvars(raw)).resolve())


def build_sbatch_command(config_path: Path, tp: int, ep: int, dry_run: bool = False) -> list[str]:
    cfg = load_study_config(config_path)
    slurm = cfg.slurm
    world_size = tp * ep
    nodes = max(1, math.ceil(world_size / slurm.gpus_per_node))
    out_dir = Path(cfg.output_root) / cfg.study_name / f"tp{tp}-ep{ep}"
    logs_dir = Path(cfg.output_root) / cfg.study_name / "slurm-logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "sbatch",
        f"--job-name={cfg.study_name}-tp{tp}-ep{ep}",
        f"--nodes={nodes}",
        f"--ntasks={world_size}",
        f"--cpus-per-task={slurm.cpus_per_task}",
        f"--mem={slurm.mem}",
        f"--time={slurm.time}",
        f"--output={logs_dir / ('%x-%j.out')}",
        f"--error={logs_dir / ('%x-%j.err')}",
    ]
    gpus_per_node_request = min(world_size, slurm.gpus_per_node)
    if slurm.use_gres:
        if slurm.gpu_type:
            cmd.append(f"--gres=gpu:{slurm.gpu_type}:{gpus_per_node_request}")
        else:
            cmd.append(f"--gres=gpu:{gpus_per_node_request}")
    else:
        cmd.append(f"--gpus-per-node={gpus_per_node_request}")
    _add_if(cmd, "--partition", slurm.partition)
    _add_if(cmd, "--account", slurm.account)
    _add_if(cmd, "--qos", slurm.qos)
    _add_if(cmd, "--constraint", slurm.constraint)
    cmd.extend(slurm.extra_sbatch_args)

    workdir = _normalize_path_for_export(slurm.workdir, default=str(ROOT))
    venv = _normalize_path_for_export(slurm.venv)
    uv_env_dir = _normalize_path_for_export(slurm.uv_env_dir)

    export_bits = {
        "ALL": None,
        "STUDY_CONFIG": str(config_path.resolve()),
        "TP_SIZE": str(tp),
        "EP_SIZE": str(ep),
        "OUT_DIR": str(out_dir.resolve()),
        "WORKDIR": workdir,
        "VENV": venv,
        "MODULES": " ".join(slurm.modules),
        "CPUS_PER_TASK": str(slurm.cpus_per_task),
        "UV_ENV_DIR": uv_env_dir,
    }
    export_arg = ",".join([k if v is None else f"{k}={v}" for k, v in export_bits.items()])
    cmd.append(f"--export={export_arg}")
    cmd.append(str((ROOT / "slurm" / "run_direct_moe_sweep.sbatch").resolve()))
    return cmd


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Submit one Slurm job per TP/EP point")
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    cfg = load_study_config(args.config)
    for point in cfg.parallel_points:
        cmd = build_sbatch_command(Path(args.config), point.tp, point.ep, dry_run=args.dry_run)
        print("[submit]", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
