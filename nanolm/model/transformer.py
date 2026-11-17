"""NanoLM Causal Language Model supporting Dense, HybridSWA, MLA, MoE, Depth-Loop, and MTP."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from nanolm.config import AttentionType, HARD_MAX_PARAMETERS, NanoLMConfig, ParameterBudgetExceededError
from nanolm.model.attention import LayerKVCache, build_attention
from nanolm.model.ffn import FineGrainedMoEFFN, build_ffn
from nanolm.model.mtp import MTPHead
from nanolm.model.norms import RMSNorm
from nanolm.model.rope import RotaryEmbedding


@dataclass
class CausalLMOutput:
    logits: Tensor
    loss: Tensor | None = None
    ce_loss: Tensor | None = None
    mtp_loss: Tensor | None = None
    z_loss: Tensor | None = None
    past_key_values: list[LayerKVCache] | None = None


class NanoLMBlock(nn.Module):
    """Single Pre-RMSNorm Transformer / MoE / MLA block."""

    def __init__(self, config: NanoLMConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = build_attention(config, layer_idx)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = build_ffn(config, layer_idx)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        doc_ids: Tensor | None = None,
        kv_cache: LayerKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, LayerKVCache | None]:
        attn_out, next_cache = self.attn(
            self.attn_norm(x),
            cos,
            sin,
            doc_ids=doc_ids,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, next_cache


class NanoLMForCausalLM(nn.Module):
    """Independent general-purpose NanoLM foundation model (strictly <= 4B deployed parameters)."""

    def __init__(self, config: NanoLMConfig) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(NanoLMBlock(config, i) for i in range(config.num_layers))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.tie_word_embeddings:
            self.lm_head: nn.Linear | None = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        rope_dim = (
            config.qk_rope_head_dim
            if config.attention_type == AttentionType.MLA
            else config.head_dim
        )
        self.rotary_emb = RotaryEmbedding(
            dim=rope_dim,
            max_position_embeddings=config.max_position_embeddings,
            theta=config.rope_theta,
            yarn_scale_factor=config.yarn_scale_factor,
        )

        self.mtp_heads = nn.ModuleList(MTPHead(config) for _ in range(config.mtp_num_heads))
        self.reset_parameters()

        # Verify actual instantiated parameter count matches analytical budget and <= 4B ceiling
        deployed = self.count_deployed_parameters()
        if deployed > HARD_MAX_PARAMETERS:
            raise ParameterBudgetExceededError(
                f"Instantiated model {config.name!r} has {deployed:,} deployed parameters (> 4,000,000,000)"
            )

    def reset_parameters(self) -> None:
        std = 0.02
        eff_depth = max(1, len(self.config.execution_layer_indices()))
        resid_std = std / math.sqrt(2.0 * eff_depth)
        for name, param in self.named_parameters():
            if param.ndim < 2:
                continue
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(param, mean=0.0, std=resid_std)
            else:
                nn.init.normal_(param, mean=0.0, std=std)

    def count_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_deployed_parameters(self) -> int:
        total = self.count_total_parameters()
        if not self.config.deploy_mtp and len(self.mtp_heads) > 0:
            mtp_params = sum(p.numel() for p in self.mtp_heads.parameters())
            return total - mtp_params
        return total

    def strip_training_only_heads(self) -> None:
        """Remove non-deployed MTP auxiliary heads before exporting an inference checkpoint."""
        if not self.config.deploy_mtp and len(self.mtp_heads) > 0:
            self.mtp_heads = nn.ModuleList()

    def project_logits(self, hidden_states: Tensor) -> Tensor:
        weight = self.embed_tokens.weight if self.lm_head is None else self.lm_head.weight
        logits = F.linear(hidden_states, weight)
        if self.config.logit_softcap is not None:
            cap = self.config.logit_softcap
            logits = cap * torch.tanh(logits / cap)
        return logits

    def moe_load_stats(self) -> dict[str, float]:
        """Inspect MoE expert load balance (coefficient of variation and min/max fraction)."""
        cvs: list[float] = []
        max_fracs: list[float] = []
        for layer in self.layers:
            if isinstance(layer.ffn, FineGrainedMoEFFN):
                fracs = layer.ffn.last_expert_fractions
                mean = fracs.mean().clamp(min=1e-9)
                cvs.append(float((fracs.std(unbiased=False) / mean).item()))
                max_fracs.append(float(fracs.max().item()))
        if not cvs:
            return {}
        return {
            "moe_load_cv": sum(cvs) / len(cvs),
            "moe_max_expert_frac": max(max_fracs),
        }

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        loss_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        doc_ids: Tensor | None = None,
        past_key_values: list[LayerKVCache] | None = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        bsz, seq_len = input_ids.shape
        exec_indices = self.config.execution_layer_indices()

        if position_ids is None:
            offset = past_key_values[0].seen_tokens if (past_key_values and len(past_key_values) > 0) else 0
            position_ids = torch.arange(offset, offset + seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)

        hidden = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(position_ids, dtype=hidden.dtype)

        next_caches: list[LayerKVCache] | None = [] if use_cache else None
        for step_idx, physical_layer_idx in enumerate(exec_indices):
            block = self.layers[physical_layer_idx]
            layer_cache = past_key_values[step_idx] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training and not use_cache:
                hidden, step_cache = grad_checkpoint(
                    block,
                    hidden,
                    cos,
                    sin,
                    doc_ids,
                    None,
                    False,
                    use_reentrant=False,
                )
            else:
                hidden, step_cache = block(
                    hidden,
                    cos,
                    sin,
                    doc_ids=doc_ids,
                    kv_cache=layer_cache,
                    use_cache=use_cache,
                )
            if next_caches is not None and step_cache is not None:
                next_caches.append(step_cache)

        normed_hidden = self.norm(hidden)
        logits = self.project_logits(normed_hidden)

        if targets is None:
            return CausalLMOutput(logits=logits, past_key_values=next_caches)

        # Primary next-token cross-entropy loss (supports masked SFT or packed boundaries via ignore_index=-100)
        vocab = self.config.vocab_size
        if loss_mask is not None:
            ce_per_tok = F.cross_entropy(
                logits.view(-1, vocab).float(),
                targets.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(bsz, seq_len)
            valid = loss_mask.float() * (targets != -100).float()
            ce_loss = (ce_per_tok * valid).sum() / valid.sum().clamp(min=1.0)
        else:
            ce_loss = F.cross_entropy(
                logits.view(-1, vocab).float(),
                targets.view(-1),
                ignore_index=-100,
            )

        # Z-loss stabilization on partition function
        z_loss = self.config.z_loss_weight * (torch.logsumexp(logits.float(), dim=-1) ** 2).mean()
        total_loss = ce_loss + z_loss

        # Auxiliary Multi-Token Prediction (MTP) loss if enabled and sequence is long enough
        mtp_loss_val: Tensor | None = None
        if len(self.mtp_heads) > 0 and seq_len > len(self.mtp_heads):
            mtp_losses: list[Tensor] = []
            prev_h = hidden  # Un-normed residual stream fed into MTP head
            for k_idx, mtp_head in enumerate(self.mtp_heads):
                # Predict targets[:, k_idx + 1 :] using prev_h[:, :-1] and ground-truth targets[:, k_idx : k_idx + L]
                span = seq_len - (k_idx + 1)
                if span <= 0:
                    break
                # Replace ignore_index (-100) in teacher-forced conditioning tokens with pad_token_id
                cond_tokens = targets[:, k_idx : k_idx + span].clamp(min=0)
                future_targets = targets[:, k_idx + 1 : k_idx + 1 + span]
                # Mask cross-document MTP transitions if doc_ids provided
                if doc_ids is not None:
                    same_doc = doc_ids[:, :span] == doc_ids[:, k_idx + 1 : k_idx + 1 + span]
                    future_targets = future_targets.masked_fill(~same_doc, -100)

                cond_embeds = self.embed_tokens(cond_tokens)
                sub_cos = cos[:, :, :span, :]
                sub_sin = sin[:, :, :span, :]
                sub_doc = doc_ids[:, :span] if doc_ids is not None else None
                prev_h = mtp_head(prev_h[:, :span, :], cond_embeds, sub_cos, sub_sin, doc_ids=sub_doc)
                mtp_logits = self.project_logits(prev_h)
                head_ce = F.cross_entropy(
                    mtp_logits.reshape(-1, vocab).float(),
                    future_targets.reshape(-1),
                    ignore_index=-100,
                )
                mtp_losses.append(head_ce)
            if mtp_losses:
                mtp_loss_val = sum(mtp_losses) / len(mtp_losses)
                total_loss = total_loss + self.config.mtp_loss_weight * mtp_loss_val

        return CausalLMOutput(
            logits=logits,
            loss=total_loss,
            ce_loss=ce_loss,
            mtp_loss=mtp_loss_val,
            z_loss=z_loss,
            past_key_values=next_caches,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.9,
        stop_token_ids: tuple[int, ...] | None = None,
    ) -> Tensor:
        """Fast KV-cached autoregressive generation."""
        was_training = self.training
        self.eval()
        try:
            stops = set(stop_token_ids) if stop_token_ids is not None else {self.config.eos_token_id}
            out = self(input_ids=input_ids, use_cache=True)
            kv = out.past_key_values
            generated = input_ids
            finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

            for _ in range(max_new_tokens):
                next_logits = out.logits[:, -1, :]
                if temperature <= 0.0:
                    next_tok = next_logits.argmax(dim=-1, keepdim=True)
                else:
                    probs = F.softmax(next_logits.float() / temperature, dim=-1)
                    if 0.0 < top_p < 1.0:
                        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                        cumprobs = torch.cumsum(sorted_probs, dim=-1)
                        cutoff = cumprobs - sorted_probs > top_p
                        sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
                        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                        sampled = torch.multinomial(sorted_probs, num_samples=1)
                        next_tok = sorted_idx.gather(-1, sampled)
                    else:
                        next_tok = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_tok], dim=1)
                for s_id in stops:
                    finished = finished | (next_tok.squeeze(-1) == s_id)
                if bool(finished.all()):
                    break
                out = self(input_ids=next_tok, past_key_values=kv, use_cache=True)
                kv = out.past_key_values
            return generated
        finally:
            if was_training:
                self.train()
