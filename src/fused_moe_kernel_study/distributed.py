from __future__ import annotations

import inspect
import os
import socket
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedEnv:
    rank: int
    local_rank: int
    world_size: int
    master_addr: str
    master_port: int


def _first_slurm_hostname() -> str:
    nodelist = os.environ.get("SLURM_NODELIST")
    if not nodelist:
        return socket.gethostname()
    try:
        out = subprocess.check_output(["scontrol", "show", "hostnames", nodelist], text=True)
        hosts = [line.strip() for line in out.splitlines() if line.strip()]
        if hosts:
            return hosts[0]
    except Exception:
        pass
    return socket.gethostname()


def ensure_distributed_env() -> DistributedEnv:
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
    if "WORLD_SIZE" not in os.environ and "SLURM_NTASKS" in os.environ:
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    if "LOCAL_RANK" not in os.environ and "SLURM_LOCALID" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = _first_slurm_hostname()
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = os.environ.get("VLLM_MASTER_PORT", "29500")

    missing = [
        key
        for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
        if key not in os.environ
    ]
    if missing:
        raise RuntimeError(f"Missing distributed environment variables: {missing}")

    return DistributedEnv(
        rank=int(os.environ["RANK"]),
        local_rank=int(os.environ["LOCAL_RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        master_addr=os.environ["MASTER_ADDR"],
        master_port=int(os.environ["MASTER_PORT"]),
    )


def init_vllm_distributed(tp_size: int, ep_size: int, backend: str = "nccl", all2all_backend: str = "nccl") -> DistributedEnv:
    env = ensure_distributed_env()
    expected_world_size = tp_size * ep_size
    if env.world_size != expected_world_size:
        raise RuntimeError(
            f"WORLD_SIZE={env.world_size} but this run expects tp*ep={tp_size}*{ep_size}={expected_world_size}. "
            "Run one Slurm/torchrun job per TP/EP point."
        )

    import torch
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    # With --gpus-per-task=1, each process only sees cuda:0 via CUDA_VISIBLE_DEVICES.
    # vLLM's CustomAllreduce (used for TP allreduce) calls can_device_access_peer()
    # using global rank as device ID, which fails when only device 0 is visible.
    # Disable it so vLLM falls back to NCCL allreduce.
    try:
        import vllm.distributed.device_communicators.custom_all_reduce as _car
        _car._can_p2p = lambda rank, world_size: False
    except Exception:
        pass

    # When srun uses --gpus-per-task=1, each process sees only 1 GPU (cuda:0)
    # via CUDA_VISIBLE_DEVICES. Fall back to 0 if local_rank exceeds visible count.
    visible_device_count = torch.cuda.device_count()
    effective_local_rank = env.local_rank if env.local_rank < visible_device_count else 0
    torch.cuda.set_device(effective_local_rank)
    init_distributed_environment(
        world_size=env.world_size,
        rank=env.rank,
        distributed_init_method="env://",
        local_rank=effective_local_rank,
        backend=backend,
    )

    sig = inspect.signature(ensure_model_parallel_initialized)
    kwargs = {}
    if "tensor_model_parallel_size" in sig.parameters:
        kwargs["tensor_model_parallel_size"] = tp_size
    if "pipeline_model_parallel_size" in sig.parameters:
        kwargs["pipeline_model_parallel_size"] = 1
    if "decode_context_model_parallel_size" in sig.parameters:
        kwargs["decode_context_model_parallel_size"] = 1
    if "prefill_context_model_parallel_size" in sig.parameters:
        kwargs["prefill_context_model_parallel_size"] = 1
    if "enable_expert_parallel" in sig.parameters:
        kwargs["enable_expert_parallel"] = ep_size > 1
    if "backend" in sig.parameters:
        kwargs["backend"] = backend

    from contextlib import nullcontext
    try:
        from .vllm_adapter import build_vllm_config, vllm_config_context
        ctx = vllm_config_context(build_vllm_config(all2all_backend=all2all_backend, ep_size=ep_size))
    except Exception:
        ctx = nullcontext()
    with ctx:
        ensure_model_parallel_initialized(**kwargs)
    return env


def cleanup_vllm_distributed() -> None:
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
