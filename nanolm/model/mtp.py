"""Multi-Token Prediction (MTP) auxiliary lookahead module for NanoLM."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from nanolm.config import NanoLMConfig
from nanolm.model.attention import build_attention
from nanolm.model.ffn import build_ffn
from nanolm.model.norms import RMSNorm


class MTPBlock(nn.Module):
    """Single causal transformer block inside an MTP lookahead head."""

    def __init__(self, config: NanoLMConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = build_attention(config, layer_idx=config.num_layers - 1)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = build_ffn(config, layer_idx=0, force_dense=True)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        doc_ids: Tensor | None = None,
    ) -> Tensor:
        attn_out, _ = self.attn(self.attn_norm(x), cos, sin, doc_ids=doc_ids, use_cache=False)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MTPHead(nn.Module):
    """Predicts future token t + (k + 1) from trunk representation h_t and ground-truth token x_{t+k}."""

    def __init__(self, config: NanoLMConfig) -> None:
        super().__init__()
        d = config.hidden_size
        self.h_norm = RMSNorm(d, eps=config.rms_norm_eps)
        self.emb_norm = RMSNorm(d, eps=config.rms_norm_eps)
        self.fusion_proj = nn.Linear(2 * d, d, bias=False)
        self.block = MTPBlock(config)
        self.out_norm = RMSNorm(d, eps=config.rms_norm_eps)

    def forward(
        self,
        prev_hidden: Tensor,
        next_token_embeds: Tensor,
        cos: Tensor,
        sin: Tensor,
        doc_ids: Tensor | None = None,
    ) -> Tensor:
        fused = self.fusion_proj(torch.cat([self.h_norm(prev_hidden), self.emb_norm(next_token_embeds)], dim=-1))
        h = self.block(fused, cos, sin, doc_ids=doc_ids)
        return self.out_norm(h)
