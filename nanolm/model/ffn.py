"""Feed-Forward Networks for NanoLM: Dense SwiGLU and Fine-Grained Aux-Loss-Free MoE."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from nanolm.config import NanoLMConfig


class SwiGLUFFN(nn.Module):
    """Bias-free SwiGLU feed-forward network."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class FineGrainedMoEFFN(nn.Module):
    """Shared-expert + Fine-Grained Top-K Routed SwiGLU MoE with aux-loss-free bias balancing."""

    def __init__(self, config: NanoLMConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.num_shared = config.num_shared_experts
        self.bias_update_rate = config.router_bias_update_rate

        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        # Non-parametric routing bias adjusted dynamically during training (DeepSeek-V3 style)
        # so load balancing does NOT distort language modeling gradients!
        self.register_buffer("expert_bias", torch.zeros(self.num_experts, dtype=torch.float32))
        self.register_buffer("last_expert_fractions", torch.full((self.num_experts,), 1.0 / self.num_experts))

        self.shared_experts = nn.ModuleList(
            SwiGLUFFN(self.hidden_size, config.moe_expert_intermediate_size)
            for _ in range(self.num_shared)
        )
        self.experts = nn.ModuleList(
            SwiGLUFFN(self.hidden_size, config.moe_expert_intermediate_size)
            for _ in range(self.num_experts)
        )

    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        flat_x = x.view(-1, self.hidden_size)  # [N, D]
        n_tokens = flat_x.shape[0]

        # Shared always-active expert pathway
        out = torch.zeros_like(flat_x)
        for shared in self.shared_experts:
            out = out + shared(flat_x)

        # Sigmoid affinity scores + bias-guided selection (unbiased weights!)
        scores = torch.sigmoid(self.router(flat_x).float())  # [N, E]
        selection_scores = scores + self.expert_bias.unsqueeze(0)
        _, topk_idx = torch.topk(selection_scores, k=self.top_k, dim=-1)  # [N, K]

        topk_weights = scores.gather(dim=-1, index=topk_idx)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        topk_weights = topk_weights.to(flat_x.dtype)

        # Update load-balancing bias in training mode without gradient pollution
        if self.training and n_tokens > 0:
            with torch.no_grad():
                counts = torch.bincount(topk_idx.reshape(-1), minlength=self.num_experts).float()
                fractions = counts / max(1.0, float(n_tokens * self.top_k))
                self.last_expert_fractions.copy_(fractions)
                target_frac = 1.0 / self.num_experts
                # Overloaded experts get slightly lower bias; underloaded get slightly higher bias
                self.expert_bias.add_(self.bias_update_rate * torch.sign(target_frac - fractions))
                self.expert_bias.clamp_(-1.0, 1.0)

        # Dispatch tokens per active expert (zero token dropping)
        for expert_id, expert in enumerate(self.experts):
            mask = topk_idx == expert_id  # [N, K]
            if not bool(mask.any()):
                continue
            token_indices, slot_indices = torch.where(mask)
            expert_in = flat_x.index_select(0, token_indices)
            expert_out = expert(expert_in) * topk_weights[token_indices, slot_indices].unsqueeze(-1)
            out.index_add_(0, token_indices, expert_out)

        return out.view(orig_shape)


def build_ffn(config: NanoLMConfig, layer_idx: int, *, force_dense: bool = False) -> nn.Module:
    if not force_dense and config.is_moe_layer(layer_idx):
        return FineGrainedMoEFFN(config)
    return SwiGLUFFN(config.hidden_size, config.intermediate_size)
