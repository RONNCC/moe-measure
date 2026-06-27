from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch


def _call_with_supported_kwargs(fn: Any, **kwargs: Any) -> Any:
    sig = inspect.signature(fn)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filtered)


def _maybe_enum_member(enum_cls: Any, names: list[str]) -> Any | None:
    for name in names:
        if hasattr(enum_cls, name):
            return getattr(enum_cls, name)
    return None


@contextmanager
def sp_local_sizes_context():
    """Enter dp_metadata.sp_local_sizes(1) if dp_metadata is present.

    vllm's MoE runner enters this before dispatch so local_sizes is populated
    for the allgather all2all path. We bypass the runner so we must do it ourselves.

    NOTE: must not wrap the yield in try/except — if an exception is thrown into
    the generator by contextmanager's __exit__, catching it here would cause the
    generator to yield a second time, raising "generator didn't stop after throw()".
    """
    from contextlib import nullcontext

    cm = nullcontext()
    try:
        from vllm.forward_context import get_forward_context
        ctx = get_forward_context()
        if ctx is not None and getattr(ctx, "dp_metadata", None) is not None:
            cm = ctx.dp_metadata.sp_local_sizes(1)
    except Exception:
        pass

    with cm:
        yield


@contextmanager
def vllm_config_context(vllm_config: Any):
    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(vllm_config):
        yield


def build_vllm_config(all2all_backend: str, ep_size: int) -> Any:
    from vllm.config import VllmConfig

    cfg = VllmConfig()
    cfg.model_config = SimpleNamespace(enforce_eager=True, is_moe=True, model="direct-fused-moe-study")
    cfg.parallel_config.data_parallel_size = ep_size
    cfg.parallel_config.enable_expert_parallel = ep_size > 1
    cfg.parallel_config.all2all_backend = all2all_backend
    if hasattr(cfg, "device_config") and hasattr(cfg.device_config, "device"):
        cfg.device_config.device = "cuda"
    return cfg


@dataclass
class KernelArtifacts:
    vllm_config: Any
    moe_config: Any
    quant_config: Any
    kernel: Any
    topk_index_dtype: torch.dtype


def make_moe_parallel_config(vllm_config: Any) -> Any:
    from vllm.distributed import get_dp_group, get_pcp_group, get_tensor_model_parallel_world_size
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

    sig = inspect.signature(FusedMoEParallelConfig.make)
    kwargs: dict[str, Any] = {
        "tp_size_": get_tensor_model_parallel_world_size(),
        "dp_size_": get_dp_group().world_size,
        "vllm_parallel_config": vllm_config.parallel_config,
    }
    if "pcp_size_" in sig.parameters:
        kwargs["pcp_size_"] = get_pcp_group().world_size
    if "sp_size_" in sig.parameters:
        kwargs["sp_size_"] = 1
    return FusedMoEParallelConfig.make(**kwargs)


def make_unquantized_config() -> Any:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

    try:
        from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG

        return FUSED_MOE_UNQUANTIZED_CONFIG
    except Exception:
        pass

    try:
        return FusedMoEQuantConfig.make(None)
    except TypeError:
        return _call_with_supported_kwargs(FusedMoEQuantConfig.make, quant_dtype=None)


def make_moe_config(
    *,
    shape_name: str,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_local_experts: int,
    topk: int,
    dtype: torch.dtype,
    max_num_tokens: int,
    vllm_config: Any,
) -> Any:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, RoutingMethodType

    moe_parallel_config = make_moe_parallel_config(vllm_config)
    routing_method = _maybe_enum_member(RoutingMethodType, ["TopK", "DeepSeekV3"])
    activation = _maybe_enum_member(MoEActivation, ["SILU", "silu"]) or "silu"

    kwargs: dict[str, Any] = {
        "num_experts": num_experts,
        "experts_per_token": topk,
        "hidden_dim": hidden_size,
        "num_local_experts": num_local_experts,
        "moe_parallel_config": moe_parallel_config,
        "in_dtype": dtype,
        "max_num_tokens": max_num_tokens,
        "activation": activation,
        "device": "cuda",
        "routing_method": routing_method,
        "num_logical_experts": num_experts,
    }

    sig = inspect.signature(FusedMoEConfig)
    if "intermediate_size" in sig.parameters:
        kwargs["intermediate_size"] = intermediate_size
    if "intermediate_size_per_partition" in sig.parameters:
        kwargs["intermediate_size_per_partition"] = intermediate_size // moe_parallel_config.tp_size
    if "name" in sig.parameters:
        kwargs["name"] = shape_name
    return FusedMoEConfig(**{k: v for k, v in kwargs.items() if k in sig.parameters})


def _is_batched_activation_format(prepare_finalize: Any) -> bool:
    activation_format = getattr(prepare_finalize, "activation_format", None)
    if callable(activation_format):
        activation_format = activation_format()
    return "Batched" in str(activation_format)


def make_fused_experts(moe_config: Any, quant_config: Any, prepare_finalize: Any) -> Any:
    num_dispatchers = prepare_finalize.num_dispatchers() if hasattr(prepare_finalize, "num_dispatchers") else 1
    if _is_batched_activation_format(prepare_finalize):
        max_num_tokens = None
        if hasattr(prepare_finalize, "max_num_tokens_per_rank"):
            max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        if max_num_tokens is None:
            max_num_tokens = getattr(moe_config, "max_num_tokens", None)
        constructors: list[tuple[Any, dict[str, Any]]] = []
        try:
            from vllm.model_executor.layers.fused_moe import BatchedTritonExperts

            constructors.append((BatchedTritonExperts, {
                "moe_config": moe_config,
                "quant_config": quant_config,
                "max_num_tokens": max_num_tokens,
                "num_dispatchers": num_dispatchers,
            }))
        except Exception:
            pass
        try:
            from vllm.model_executor.layers.fused_moe.batched_triton_or_deep_gemm_moe import BatchedTritonOrDeepGemmExperts

            constructors.append((BatchedTritonOrDeepGemmExperts, {
                "moe_config": moe_config,
                "quant_config": quant_config,
                "max_num_tokens": max_num_tokens,
                "num_dispatchers": num_dispatchers,
            }))
        except Exception:
            pass
        for cls, kwargs in constructors:
            try:
                return _call_with_supported_kwargs(cls, **kwargs)
            except Exception:
                continue
    else:
        constructors = []
        try:
            from vllm.model_executor.layers.fused_moe import TritonExperts

            constructors.append((TritonExperts, {"moe_config": moe_config, "quant_config": quant_config}))
        except Exception:
            pass
        try:
            from vllm.model_executor.layers.fused_moe import TritonOrDeepGemmExperts

            constructors.append((TritonOrDeepGemmExperts, {"moe_config": moe_config, "quant_config": quant_config}))
        except Exception:
            pass
        for cls, kwargs in constructors:
            try:
                return _call_with_supported_kwargs(cls, **kwargs)
            except Exception:
                continue

    raise RuntimeError(
        "Could not construct a fused-experts implementation for the installed vLLM version. "
        "Try swapping make_fused_experts() to the exact class used by your vLLM checkout."
    )


def make_prepare_finalize(moe_config: Any, quant_config: Any) -> Any:
    from vllm.model_executor.layers.fused_moe.all2all_utils import maybe_make_prepare_finalize

    try:
        return maybe_make_prepare_finalize(
            moe=moe_config,
            quant_config=quant_config,
            allow_new_interface=True,
            use_monolithic=False,
        )
    except TypeError:
        try:
            return maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=quant_config,
                allow_new_interface=True,
            )
        except TypeError:
            return maybe_make_prepare_finalize(moe=moe_config, quant_config=quant_config)


def make_kernel(
    *,
    vllm_config: Any,
    moe_config: Any,
    quant_config: Any,
) -> Any:
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk

    prepare_finalize = make_prepare_finalize(moe_config, quant_config)
    if prepare_finalize is None:
        raise RuntimeError("maybe_make_prepare_finalize() returned None; backend may be unsupported in this environment.")

    fused_experts = make_fused_experts(moe_config, quant_config, prepare_finalize)
    kernel_cls = getattr(mk, "FusedMoEKernel", None) or getattr(mk, "FusedMoEModularKernel", None)
    if kernel_cls is None:
        raise RuntimeError("Could not find FusedMoEKernel/FusedMoEModularKernel in vLLM modular_kernel module.")

    kwargs = {"prepare_finalize": prepare_finalize, "fused_experts": fused_experts}
    sig = inspect.signature(kernel_cls)
    if "moe_parallel_config" in sig.parameters:
        kwargs["moe_parallel_config"] = moe_config.moe_parallel_config
    return kernel_cls(**{k: v for k, v in kwargs.items() if k in sig.parameters})


def _maybe_init_workspace_manager() -> None:
    try:
        from vllm.v1.worker.workspace import init_workspace_manager
        device = torch.device("cuda", torch.cuda.current_device())
        init_workspace_manager(device)
    except Exception:
        pass


def build_kernel_artifacts(
    *,
    shape_name: str,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_local_experts: int,
    topk: int,
    dtype: torch.dtype,
    max_num_tokens: int,
    all2all_backend: str,
    ep_size: int,
) -> KernelArtifacts:
    _maybe_init_workspace_manager()
    vllm_config = build_vllm_config(all2all_backend=all2all_backend, ep_size=ep_size)
    with vllm_config_context(vllm_config):
        quant_config = make_unquantized_config()
        moe_config = make_moe_config(
            shape_name=shape_name,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            topk=topk,
            dtype=dtype,
            max_num_tokens=max_num_tokens,
            vllm_config=vllm_config,
        )
        kernel = make_kernel(vllm_config=vllm_config, moe_config=moe_config, quant_config=quant_config)
        topk_index_dtype = kernel.prepare_finalize.topk_indices_dtype() if hasattr(kernel.prepare_finalize, "topk_indices_dtype") else torch.int32
    return KernelArtifacts(
        vllm_config=vllm_config,
        moe_config=moe_config,
        quant_config=quant_config,
        kernel=kernel,
        topk_index_dtype=topk_index_dtype,
    )
