from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


# Supported routing modes.
#
# "skewed"      — existing probability-vector approach; alpha controls imbalance;
#                 hot_expert_count experts get alpha× the probability of the rest.
# "uniform"     — uniform probability across all experts (alpha=1.0 special case).
# "zipfian"     — power-law: P(expert_k) ∝ 1/k^zipf_s, experts ranked by popularity.
# "random"      — one Dirichlet(1) draw per batch; mild natural imbalance.
# "skewed-2x"   — one hot expert gets exactly 2× average; rest share uniformly.
# "skewed-4x"   — one hot expert gets exactly 4× average; rest share uniformly.
# "worst-case"  — all tokens assigned to the same first `topk` experts.
ROUTING_MODES = ("skewed", "uniform", "zipfian", "random", "skewed-2x", "skewed-4x", "worst-case")


@dataclass(frozen=True)
class RoutingStats:
    routing_mode: str
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


def _topk_from_probs(
    probs: torch.Tensor,
    num_tokens: int,
    topk: int,
    gen: torch.Generator,
    topk_index_dtype: torch.dtype,
) -> torch.Tensor:
    """Sample topk expert indices without replacement from a probability vector."""
    expanded = probs.to(dtype=torch.float32).expand(num_tokens, -1)
    topk_ids = torch.multinomial(expanded, num_samples=topk, replacement=False, generator=gen)
    return topk_ids.to(dtype=topk_index_dtype)


def make_routing_batch(
    num_tokens: int,
    num_experts: int,
    topk: int,
    alpha: float = 1.0,
    hot_expert_count: int = 1,
    device: torch.device | str = "cuda",
    seed: int = 0,
    weight_mode: str = "uniform",
    topk_index_dtype: torch.dtype = torch.int32,
    routing_mode: str = "skewed",
    zipf_s: float = 1.0,
) -> RoutingBatch:
    if topk > num_experts:
        raise ValueError(f"topk={topk} cannot exceed num_experts={num_experts}")
    if num_tokens <= 0:
        raise ValueError("num_tokens must be > 0")
    if routing_mode not in ROUTING_MODES:
        raise ValueError(f"routing_mode={routing_mode!r} not in {ROUTING_MODES}")

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    if routing_mode == "worst-case":
        # All tokens assigned to the first `topk` experts — maximum imbalance.
        topk_ids = (
            torch.arange(topk, dtype=topk_index_dtype, device=device)
            .unsqueeze(0)
            .expand(num_tokens, -1)
            .contiguous()
        )
        probs = torch.zeros(num_experts, dtype=torch.float64, device=device)
        probs[:topk] = 1.0 / topk
    elif routing_mode == "uniform":
        probs = torch.ones(num_experts, dtype=torch.float64, device=device) / num_experts
        topk_ids = _topk_from_probs(probs, num_tokens, topk, gen, topk_index_dtype)
    elif routing_mode == "zipfian":
        ranks = torch.arange(1, num_experts + 1, dtype=torch.float64, device=device)
        probs = 1.0 / (ranks ** zipf_s)
        probs /= probs.sum()
        topk_ids = _topk_from_probs(probs, num_tokens, topk, gen, topk_index_dtype)
    elif routing_mode == "random":
        # One Dirichlet(1) draw — batch-level mild imbalance.
        concentration = torch.ones(num_experts, dtype=torch.float32, device=device)
        probs = torch.distributions.Dirichlet(concentration).sample().to(torch.float64)
        topk_ids = _topk_from_probs(probs, num_tokens, topk, gen, topk_index_dtype)
    elif routing_mode in ("skewed-2x", "skewed-4x"):
        skew_alpha = 2.0 if routing_mode == "skewed-2x" else 4.0
        probs = make_probability_vector(num_experts, skew_alpha, hot_expert_count=1, device=device)
        topk_ids = _topk_from_probs(probs, num_tokens, topk, gen, topk_index_dtype)
    else:
        # "skewed" — original behaviour with caller-supplied alpha / hot_expert_count.
        probs = make_probability_vector(num_experts, alpha, hot_expert_count=hot_expert_count, device=device)
        topk_ids = _topk_from_probs(probs, num_tokens, topk, gen, topk_index_dtype)

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
    avg_count = float(num_tokens * topk) / num_experts
    observed_alpha = float(max_count) / avg_count if avg_count > 0 else float("inf")

    stats = RoutingStats(
        routing_mode=routing_mode,
        requested_alpha=float(alpha),
        observed_alpha=observed_alpha,
        counts=[int(x) for x in counts.tolist()],
        probabilities=[float(x) for x in probs.tolist()],
    )
    return RoutingBatch(topk_ids=topk_ids.contiguous(), topk_weights=topk_weights.contiguous(), stats=stats)
