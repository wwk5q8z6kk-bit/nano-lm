"""Unified NanoLM Discovery model assembling baselines and invented architectures."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nanolm_discovery.config import (
    ArchitectureConfig,
    ParameterCensus,
    ParadigmKind,
    compute_parameter_census,
)
from nanolm_discovery.operators import (
    DiscoveryRMSNorm,
    GDN2Operator,
    GQAAttentionOperator,
    HMWROperator,
    HRWNOperator,
    MPRNOperator,
    NEBNOperator,
    PortHamiltonianOperator,
    SwiGLUSynthesis,
    SymplecticHamiltonianSynthesis,
    VariationalDendriticSynthesis,
)


@dataclass
class ModelOutput:
    """Structured forward output with CE/JEP decomposition and epistemic uncertainty."""

    logits: Tensor
    loss: Tensor | None = None
    ce_loss: Tensor | None = None
    jep_loss: Tensor | None = None
    mean_uncertainty: Tensor | None = None
    token_uncertainty: Tensor | None = None
    states: list[dict[str, Any]] | None = None


def _build_operator(cfg: ArchitectureConfig, layer_idx: int) -> nn.Module:
    p = cfg.paradigm
    if p == ParadigmKind.B0_TRANSFORMER:
        return GQAAttentionOperator(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.rope_theta)
    if p == ParadigmKind.B1_GDN2_HYBRID:
        if (layer_idx + 1) % cfg.global_attn_every == 0:
            return GQAAttentionOperator(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.rope_theta)
        return GDN2Operator(cfg.d_model, cfg.n_heads, cfg.head_dim, chunk_size=cfg.chunk_size)
    if p == ParadigmKind.I1_PHMN:
        return PortHamiltonianOperator(cfg.d_model, cfg.n_heads, cfg.head_dim, chunk_size=cfg.chunk_size)
    if p == ParadigmKind.I2_NEBN:
        return NEBNOperator(cfg.d_model, cfg.n_heads, cfg.head_dim, chunk_size=cfg.nebn_chunk_size)
    if p == ParadigmKind.I3_HRWN:
        return HRWNOperator(cfg.d_model, scam_slots=cfg.scam_slots)
    if p == ParadigmKind.G3_HMWR:
        return HMWROperator(
            cfg.d_model, cfg.n_heads, cfg.head_dim, chunk_size=cfg.chunk_size, scam_slots=cfg.scam_slots
        )
    if p == ParadigmKind.G2_MPRN:
        return MPRNOperator(
            cfg.d_model,
            cfg.n_heads,
            cfg.head_dim,
            chunk_size=cfg.chunk_size,
            scam_slots=cfg.scam_slots,
            use_symplectic_reflection=cfg.use_symplectic_reflection,
            use_precision_norm=cfg.use_precision_norm,
            use_scam_memory=cfg.use_scam_memory,
        )
    raise ValueError(f"Unsupported paradigm: {p}")


def _build_synthesis(cfg: ArchitectureConfig) -> nn.Module:
    p = cfg.paradigm
    if p in (ParadigmKind.B0_TRANSFORMER, ParadigmKind.B1_GDN2_HYBRID, ParadigmKind.I3_HRWN, ParadigmKind.G3_HMWR):
        return SwiGLUSynthesis(cfg.d_model, cfg.ffn_hidden)
    if p == ParadigmKind.I1_PHMN:
        return SymplecticHamiltonianSynthesis(cfg.d_model, cfg.ffn_hidden, n_heads=0)
    if p == ParadigmKind.I2_NEBN:
        return VariationalDendriticSynthesis(cfg.d_model, cfg.n_heads, cfg.ffn_hidden)
    if p == ParadigmKind.G2_MPRN:
        if cfg.use_phase_bilinear:
            unc_heads = cfg.n_heads if cfg.use_precision_norm else 0
            return SymplecticHamiltonianSynthesis(cfg.d_model, cfg.ffn_hidden, n_heads=unc_heads)
        return SwiGLUSynthesis(cfg.d_model, cfg.ffn_hidden)
    raise ValueError(f"Unsupported paradigm: {p}")


class DiscoveryBlock(nn.Module):
    """Single normalized sequence-operator + synthesis block (with optional depth-attention routing)."""

    def __init__(self, cfg: ArchitectureConfig, layer_idx: int) -> None:
        super().__init__()
        self.norm1 = DiscoveryRMSNorm(cfg.d_model)
        self.operator = _build_operator(cfg, layer_idx)
        self.norm2 = DiscoveryRMSNorm(cfg.d_model)
        self.synthesis = _build_synthesis(cfg)
        use_depth = cfg.paradigm == ParadigmKind.G3_HMWR and cfg.use_depth_routing
        self.depth_router = nn.Linear(cfg.d_model, layer_idx + 1, bias=True) if (use_depth and layer_idx >= 1) else None
        self.adapt_gate = nn.Linear(1, cfg.d_model, bias=True) if use_depth else None

    def forward(
        self,
        x: Tensor,
        state: dict[str, Any] | None = None,
        depth_history: list[Tensor] | None = None,
    ) -> tuple[Tensor, Tensor | None, dict[str, Any]]:
        if self.depth_router is not None and depth_history is not None:
            weights = F.softmax(self.depth_router(self.norm1(x)).float(), dim=-1).to(x.dtype)
            x_in = torch.einsum("btk,btkd->btd", weights, torch.stack(depth_history, dim=2))
        else:
            x_in = x
        op_out, unc, next_state = self.operator(self.norm1(x_in), state=state)
        x = x_in + op_out
        syn_out = self.synthesis(self.norm2(x), unc=unc)
        if self.adapt_gate is not None and unc is not None:
            u_feat = torch.log1p(unc.float().mean(dim=-1, keepdim=True).clamp(min=0.0)).to(x.dtype)
            gain = 0.5 + torch.sigmoid(self.adapt_gate(u_feat))
            syn_out = syn_out * gain
        return x + syn_out, unc, next_state


class NanoLMDiscoveryModel(nn.Module):
    """Unified multi-paradigm foundation model obeying strict <4B parameter accounting."""

    def __init__(self, cfg: ArchitectureConfig, *, skip_init_weights: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.census: ParameterCensus = compute_parameter_census(cfg, enforce_strict_4b=True)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.unemb = None if cfg.tie_embeddings else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.blocks = nn.ModuleList([DiscoveryBlock(cfg, i) for i in range(cfg.n_layers)])
        self.norm_f = DiscoveryRMSNorm(cfg.d_model)
        self.jep_head = nn.Linear(cfg.d_model, cfg.d_model, bias=False) if cfg.jep_head_enabled else None

        if not skip_init_weights:
            self._init_weights()

    def _init_weights(self) -> None:
        res_scale = 1.0 / math.sqrt(2.0 * self.cfg.n_layers)
        emb_std = min(0.06, 1.0 / math.sqrt(self.cfg.d_model))
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, mean=0.0, std=emb_std)
            elif isinstance(mod, nn.Linear):
                fan_in = max(1, mod.weight.shape[1])
                std = min(0.06, 0.8 / math.sqrt(fan_in))
                if name.endswith(("o_proj", "w_down", "w_out")):
                    std *= res_scale
                nn.init.normal_(mod.weight, mean=0.0, std=std)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.Conv1d):
                if "key_shift" in name and mod.weight.shape[-1] >= 2:
                    nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                    with torch.no_grad():
                        mod.weight[:, 0, -1] += 0.7
                        mod.weight[:, 0, -2] += 0.3
                else:
                    nn.init.normal_(mod.weight, mean=0.0, std=0.02)
        # Initialize depth routers to smoothly favor latest state while allowing skip-depth discovery
        for block in self.blocks:
            if block.depth_router is not None:
                nn.init.normal_(block.depth_router.weight, mean=0.0, std=0.01)
                with torch.no_grad():
                    block.depth_router.bias.zero_()
                    block.depth_router.bias[-1] = 2.5
        # Initialize operator decay gates for slow forgetting (gamma_0 ~ 0.99)
        for mod in self.modules():
            if hasattr(mod, "gate_proj") and isinstance(mod.gate_proj, nn.Linear) and mod.gate_proj.bias is not None:
                n_h = getattr(mod, "n_heads", self.cfg.n_heads)
                with torch.no_grad():
                    b = mod.gate_proj.bias.view(n_h, -1)
                    if isinstance(mod, MPRNOperator):
                        b[:, -1].fill_(-3.0)
                    elif b.shape[1] == 2:
                        b[:, 0].fill_(-3.0)
                    elif b.shape[1] == 3:
                        b[:, 2].fill_(-3.0)

    def embedding_param_ids(self) -> set[int]:
        ids = {id(self.tok_emb.weight)}
        if self.unemb is not None:
            ids.add(id(self.unemb.weight))
        return ids

    def deployed_parameter_count(self) -> int:
        jep_ids = {id(p) for p in self.jep_head.parameters()} if self.jep_head is not None else set()
        return sum(p.numel() for p in self.parameters() if id(p) not in jep_ids)

    def training_only_parameter_count(self) -> int:
        if self.jep_head is None:
            return 0
        return sum(p.numel() for p in self.jep_head.parameters())

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        states: list[dict[str, Any]] | None = None,
        jep_weight: float = 0.0,
        return_states: bool = False,
    ) -> ModelOutput:
        hidden = self.tok_emb(input_ids)
        next_states: list[dict[str, Any]] | None = [] if return_states else None
        unc_scalars: list[Tensor] = []
        track_depth = self.cfg.paradigm == ParadigmKind.G3_HMWR and self.cfg.use_depth_routing
        depth_history: list[Tensor] | None = [hidden] if track_depth else None

        for i, block in enumerate(self.blocks):
            st_in = states[i] if states is not None else None
            hidden, unc, st_out = block(hidden, state=st_in, depth_history=depth_history)
            if depth_history is not None:
                depth_history.append(hidden)
            if next_states is not None:
                next_states.append(st_out)
            if unc is not None:
                unc_scalars.append(unc.float().mean(dim=-1))

        hidden_norm = self.norm_f(hidden)
        if self.unemb is not None:
            logits = self.unemb(hidden_norm)
        else:
            logits = F.linear(hidden_norm, self.tok_emb.weight)

        tok_unc = torch.stack(unc_scalars, dim=0).mean(dim=0) if unc_scalars else None
        mean_unc = tok_unc.mean() if tok_unc is not None else None

        ce_loss: Tensor | None = None
        jep_loss: Tensor | None = None
        total_loss: Tensor | None = None
        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.float().reshape(-1, self.cfg.vocab_size), targets.reshape(-1), ignore_index=-1
            )
            total_loss = ce_loss
            if jep_weight > 0.0:
                if self.jep_head is None:
                    raise ValueError("jep_weight > 0 requires ArchitectureConfig(jep_head_enabled=True)")
                if input_ids.shape[1] >= 2:
                    t1 = targets[:, :-1]
                    t2 = targets[:, 1:]
                    valid = (t1 >= 0) & (t2 >= 0)
                    e1 = self.tok_emb.weight[t1.clamp(min=0)].detach().float()
                    e2 = self.tok_emb.weight[t2.clamp(min=0)].detach().float()
                    tgt_dir = F.normalize(0.5 * (e1 + e2), dim=-1)
                    pred_dir = F.normalize(self.jep_head(hidden_norm[:, :-1]).float(), dim=-1)
                    cos_loss = 1.0 - (pred_dir * tgt_dir).sum(dim=-1)
                    if tok_unc is not None:
                        w = 1.0 / (1.0 + tok_unc[:, :-1].detach())
                        w = w / w.mean().clamp(min=1e-4)
                        cos_loss = cos_loss * w
                    if bool(valid.any()):
                        jep_loss = cos_loss[valid].mean()
                        total_loss = ce_loss + jep_weight * jep_loss
                    else:
                        jep_loss = ce_loss.new_zeros(())
                else:
                    jep_loss = ce_loss.new_zeros(())

        return ModelOutput(
            logits=logits,
            loss=total_loss,
            ce_loss=ce_loss,
            jep_loss=jep_loss,
            mean_uncertainty=mean_unc,
            token_uncertainty=tok_unc,
            states=next_states,
        )

    def forward_step(
        self,
        token_id: Tensor,
        states: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Single-token autoregressive step preserving exact state duality."""
        if token_id.ndim == 1:
            token_id = token_id.unsqueeze(1)
        elif token_id.shape[1] != 1:
            raise ValueError(f"forward_step expects shape [B] or [B, 1], got {tuple(token_id.shape)}")
        return self.forward(token_id, states=states, return_states=True)
