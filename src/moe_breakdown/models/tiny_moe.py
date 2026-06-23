"""
Tiny mixture-of-experts model used for end-to-end testing of the profiler
framework.

Architecture (intentionally small so it runs anywhere, even on CPU):

    Embedding(vocab=1024, hidden=256)                 -> 262 144 params
    Gate Linear(hidden=256, num_experts=3, bias=False) ->      768 params
    3x Expert MLP(hidden=256, expert_hidden=2048)        -> ~1.05 M params each
    LM Head Linear(hidden=256, vocab=1024, bias=False) -> 262 144 params
    --------------------------------------------------------------
    Total: ~3.68 M parameters, of which ~3.15 M are in the experts.

Each Expert is a two-layer MLP with SiLU activation (SwiGLU-like):
    expert(x) = W2 * silu(W1 * x)

Routing is top-1 (one expert per token).  Tokens are dispatched to the
chosen expert, processed, and the result is gathered back.  No
all-to-all -- this is a single-device model.

To replace this with a quantized Qwen later:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-MoE-A2.7B",
        torch_dtype=torch.float16,
        load_in_4bit=True,                    # bitsandbytes 4-bit
        device_map="auto",
    )
    # then run the same `forward` interface under torch.profiler.
The categorizer does not care about the model -- it only sees Kineto events.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TinyMoEConfig:
    vocab_size: int = 1024
    hidden_dim: int = 256
    expert_hidden_dim: int = 2048     # chosen so each expert is ~1 M params
    num_experts: int = 3
    top_k: int = 1                    # top-1 routing
    seq_len: int = 32
    batch_size: int = 1


class Expert(nn.Module):
    """Single expert MLP:  hidden -> expert_hidden -> hidden, with SiLU."""

    def __init__(self, hidden_dim: int, expert_hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, expert_hidden_dim, bias=True)
        self.w2 = nn.Linear(expert_hidden_dim, hidden_dim, bias=True)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)))


class TinyMoE(nn.Module):
    """A small but real MoE model suitable for profiler testing."""

    def __init__(self, cfg: TinyMoEConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.gate = nn.Linear(cfg.hidden_dim, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList([
            Expert(cfg.hidden_dim, cfg.expert_hidden_dim)
            for _ in range(cfg.num_experts)
        ])
        # Tied with embedding by default; separate head is also fine.
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)

        # Init: small weights so outputs are reasonable.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """One forward pass.  input_ids: [B, T] of token ids, returns logits [B, T, V]."""
        h = self.embed(input_ids)                      # [B, T, H]
        # MoE routing: pick one expert per token.
        gate_logits = self.gate(h)                     # [B, T, E]
        topk = torch.topk(gate_logits, k=self.cfg.top_k, dim=-1)
        expert_idx = topk.indices[..., 0]              # [B, T]
        # Apply the chosen expert per token (vectorised over experts below).
        out = torch.zeros_like(h)
        for e_idx, expert in enumerate(self.experts):
            mask = (expert_idx == e_idx)
            if not mask.any():
                continue
            tokens = h[mask]                           # [N, H]
            out[mask] = expert(tokens)
        # LM head
        return self.lm_head(out)                       # [B, T, V]

    @staticmethod
    def make_inputs(cfg: TinyMoEConfig, seed: int = 0) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        return torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), generator=g)


def build(cfg: TinyMoEConfig | None = None) -> TinyMoE:
    cfg = cfg or TinyMoEConfig()
    model = TinyMoE(cfg)
    model.eval()
    return model
