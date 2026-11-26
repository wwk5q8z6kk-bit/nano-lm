"""Attention mechanisms for NanoLM: GQA, Hybrid Sliding-Window + Global, and MLA."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from nanolm.config import AttentionType, NanoLMConfig
from nanolm.model.norms import RMSNorm
from nanolm.model.rope import apply_rotary_emb


@dataclass
class LayerKVCache:
    """Stateful KV cache entry for one execution step (supports GQA, SWA, and MLA)."""

    k_or_latent: Tensor  # [B, KV, T, hd] for GQA/SWA, or [B, T, kv_lora_rank] for MLA
    v_or_rope: Tensor    # [B, KV, T, hd] for GQA/SWA, or [B, 1, T, qk_rope_head_dim] for MLA
    seen_tokens: int


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention with QK-RMSNorm, intra-doc causal masking, and optional Sliding Window."""

    def __init__(self, config: NanoLMConfig, *, is_sliding: bool = False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.is_sliding = is_sliding
        self.sliding_window_size = config.sliding_window_size

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.qk_norm = config.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        doc_ids: Tensor | None = None,
        kv_cache: LayerKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, LayerKVCache | None]:
        bsz, q_len, _ = x.shape
        q = self.q_proj(x).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        past_seen = 0
        if kv_cache is not None:
            past_seen = kv_cache.seen_tokens
            k = torch.cat([kv_cache.k_or_latent, k], dim=2)
            v = torch.cat([kv_cache.v_or_rope, v], dim=2)

        new_cache: LayerKVCache | None = None
        if use_cache:
            total_seen = past_seen + q_len
            if self.is_sliding and k.shape[2] > self.sliding_window_size:
                k_store = k[:, :, -self.sliding_window_size :, :]
                v_store = v[:, :, -self.sliding_window_size :, :]
            else:
                k_store, v_store = k, v
            new_cache = LayerKVCache(k_or_latent=k_store, v_or_rope=v_store, seen_tokens=total_seen)

        kv_len = k.shape[2]
        if self.num_kv_groups > 1:
            k_exp = k.repeat_interleave(self.num_kv_groups, dim=1)
            v_exp = v.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_exp, v_exp = k, v

        # Fast path: during single-token cached decode (q_len == 1) with cropped SWA cache,
        # all cached keys are strictly causal and within sliding_window_size!
        swa_requires_mask = self.is_sliding and (q_len > 1 or kv_len > self.sliding_window_size)
        need_custom_mask = (doc_ids is not None) or swa_requires_mask or (kv_cache is not None and q_len > 1)
        if not need_custom_mask:
            is_causal = q_len > 1 and kv_cache is None
            attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=is_causal)
        else:
            # Build boolean mask [B or 1, 1, q_len, kv_len] where True = allowed to attend
            q_pos = torch.arange(kv_len - q_len, kv_len, device=x.device).unsqueeze(1)  # [q_len, 1]
            k_pos = torch.arange(0, kv_len, device=x.device).unsqueeze(0)              # [1, kv_len]
            mask = k_pos <= q_pos  # [q_len, kv_len]
            if self.is_sliding:
                mask = mask & ((q_pos - k_pos) < self.sliding_window_size)
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, kv_len]
            if doc_ids is not None and kv_len == q_len:
                same_doc = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)  # [B, q_len, kv_len]
                mask = mask & same_doc.unsqueeze(1)
            attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=mask)

        out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(out), new_cache


class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention (MLA) with compressed joint KV cache and decoupled RoPE."""

    def __init__(self, config: NanoLMConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.scale = 1.0 / math.sqrt(self.qk_head_dim)

        self.q_down = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_up = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_down = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_up = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        doc_ids: Tensor | None = None,
        kv_cache: LayerKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, LayerKVCache | None]:
        bsz, q_len, _ = x.shape

        q_latent = self.q_norm(self.q_down(x))
        q = self.q_up(q_latent).view(bsz, q_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, cos, sin)
        q_full = torch.cat([q_nope, q_pe], dim=-1)  # [B, H, q_len, qk_head_dim]

        kv_compressed = self.kv_down(x)
        c_kv, k_pe_raw = torch.split(kv_compressed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_norm(c_kv)  # [B, q_len, kv_lora_rank]
        k_pe = apply_rotary_emb(k_pe_raw.unsqueeze(1), cos, sin)  # [B, 1, q_len, qk_rope_head_dim]

        past_seen = 0
        if kv_cache is not None:
            past_seen = kv_cache.seen_tokens
            c_kv = torch.cat([kv_cache.k_or_latent, c_kv], dim=1)
            k_pe = torch.cat([kv_cache.v_or_rope, k_pe], dim=2)

        new_cache: LayerKVCache | None = None
        if use_cache:
            new_cache = LayerKVCache(
                k_or_latent=c_kv,
                v_or_rope=k_pe,
                seen_tokens=past_seen + q_len,
            )

        kv_len = c_kv.shape[1]
        kv_up = self.kv_up(c_kv).view(bsz, kv_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, v = torch.split(kv_up, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_full = torch.cat([k_nope, k_pe.expand(bsz, self.num_heads, kv_len, self.qk_rope_head_dim)], dim=-1)

        # Apple Metal MPS SDPA requires v_head_dim == qk_head_dim; zero-padding v and slicing
        # after SDPA is mathematically exact and preserves fused FlashAttention on MPS!
        v_pad = self.qk_head_dim - self.v_head_dim
        v_in = F.pad(v, (0, v_pad)) if (v_pad > 0 and x.device.type == "mps") else v

        if doc_ids is None and not (kv_cache is not None and q_len > 1):
            is_causal = q_len > 1 and kv_cache is None
            attn_out = F.scaled_dot_product_attention(q_full, k_full, v_in, is_causal=is_causal, scale=self.scale)
        else:
            q_pos = torch.arange(kv_len - q_len, kv_len, device=x.device).unsqueeze(1)
            k_pos = torch.arange(0, kv_len, device=x.device).unsqueeze(0)
            mask = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)
            if doc_ids is not None and kv_len == q_len:
                mask = mask & (doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)).unsqueeze(1)
            attn_out = F.scaled_dot_product_attention(q_full, k_full, v_in, attn_mask=mask, scale=self.scale)

        if v_pad > 0 and x.device.type == "mps":
            attn_out = attn_out[..., : self.v_head_dim]

        out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.v_head_dim)
        return self.o_proj(out), new_cache


def build_attention(config: NanoLMConfig, layer_idx: int, *, force_dense_gqa: bool = False) -> nn.Module:
    if force_dense_gqa or config.attention_type == AttentionType.GQA:
        return GroupedQueryAttention(config, is_sliding=False)
    if config.attention_type == AttentionType.HYBRID_SWA:
        return GroupedQueryAttention(config, is_sliding=config.is_sliding_layer(layer_idx))
    if config.attention_type == AttentionType.MLA:
        return MultiHeadLatentAttention(config)
    raise ValueError(f"Unsupported attention_type: {config.attention_type}")
