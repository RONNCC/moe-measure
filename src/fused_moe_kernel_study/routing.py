from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RoutingStats:
    requested_alpha: float
    observed_alpha: float
    counts: list[int]
    probabilities: list[float]


@dataclass(frozen=True)
class RoutingBatch:
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    stats: RoutingStats


def make_probability_vector(
    num_experts: int,
    alpha: float,
    hot_expert_count: int = 1,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    if num_experts <= 0:
        raise ValueError("num_experts must be > 0")
    if hot_expert_count <= 0 or hot_expert_count > num_experts:
        raise ValueError("hot_expert_count must be in [1, num_experts]")
    if alpha < 1.0:
        raise ValueError("alpha is treated as an imbalance ratio and must be >= 1.0")

    probs = torch.ones(num_experts, dtype=torch.float64, device=device)
    probs[:hot_expert_count] = float(alpha)
    probs /= probs.sum()
    return probs


def make_routing_batch(
    num_tokens: int,
    num_experts: int,
    topk: int,
    alpha: float,
    hot_expert_count: int,
    device: torch.device,
    seed: int,
    weight_mode: str = "uniform",
    topk_index_dtype: torch.dtype = torch.int32,
) -> RoutingBatch:
    if topk > num_experts:
        raise ValueError(f"topk={topk} cannot exceed num_experts={num_experts}")
    if num_tokens <= 0:
        raise ValueError("num_tokens must be > 0")

    probs = make_probability_vector(num_experts, alpha, hot_expert_count=hot_expert_count, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    expanded = probs.expand(num_tokens, -1)
    topk_ids = torch.multinomial(expanded, num_samples=topk, replacement=False, generator=gen)
    topk_ids = topk_ids.to(dtype=topk_index_dtype)

    if weight_mode == "uniform":
        topk_weights = torch.full(
            (num_tokens, topk),
            fill_value=1.0 / float(topk),
            device=device,
            dtype=torch.float32,
        )
    elif weight_mode == "probability":
        sampled = torch.gather(probs.float().expand(num_tokens, -1), 1, topk_ids.to(dtype=torch.long))
        topk_weights = sampled / sampled.sum(dim=1, keepdim=True)
        topk_weights = topk_weights.to(dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported weight_mode={weight_mode!r}")

    counts = torch.bincount(topk_ids.reshape(-1).to(torch.long), minlength=num_experts)
    max_count = int(counts.max().item())
    min_positive = int(counts[counts > 0].min().item()) if torch.any(counts > 0) else 0
    observed_alpha = float(max_count / min_positive) if min_positive > 0 else float("inf")

    stats = RoutingStats(
        requested_alpha=float(alpha),
        observed_alpha=observed_alpha,
        counts=[int(x) for x in counts.tolist()],
        probabilities=[float(x) for x in probs.tolist()],
    )
    return RoutingBatch(topk_ids=topk_ids.contiguous(), topk_weights=topk_weights.contiguous(), stats=stats)
