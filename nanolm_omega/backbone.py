"""NANOLM Omega Backbones (`OMEGA-EXP-01`):

Implements and rigorously compares:
  - `X8DualUnembedModel` (`x8_dual_unembed_atrd`): Reproduces `X8_margin_atrd` (`Sharp-Delta + Gated-VRes`
    with contractive `beta in (0, 1)` + global `GQA` + dual-unembedding Top-2 Margin `CausalEntropyRefiner`)
    and accounts honestly for its dual vocabulary GEMM FLOPs.
  - `G4OmegaModel` (`g4_omega` + surgical ablations `g4_no_vres`, `g4_no_sparse_refiner`):
    Unites `G3_HMWR`'s orientation-reversing `beta in (0, 1.98)` WY scan + Dyadic Wave + Shifted SCAM-32 +
    Depth-Attention Residuals with `NanoLM-v2`'s Per-Head Layer-0 `Gated-VRes` + `QK-RMSNorm` AND an
    unembedding-free `O(d)` `LatentSurpriseRefiner` (`0` extra vocabulary GEMMs!).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nanolm_discovery.config import HARD_MAX_DEPLOYED_PARAMETERS, ParameterBudgetError, ParadigmKind, make_iso_param_suite
from nanolm_discovery.models import ModelOutput
from nanolm_discovery.operators import (
    CausalKeyShift,
    DiscoveryRMSNorm,
    SwiGLUSynthesis,
    _apply_dilated_wave_bank,
    _no_autocast_ctx,
    _run_symplectic_wy_scan,
    apply_rope,
    scam_causal_topk_attention,
    scam_step_attention,
)


def _smooth_anticirc_cosine(
    dx: Tensor,
    dh1: Tensor,
    valid_slot: Tensor | None = None,
    margin: float = 0.25,
) -> Tensor:
    """FP16/GradScaler-safe bounded anti-circularity cosine hinge loss."""
    with _no_autocast_ctx(dx.device):
        dx_f = dx.float()
        dh_f = dh1.float()
        dot = (dx_f * dh_f).sum(dim=-1)
        denom = (dx_f.norm(dim=-1) + 1e-2) * (dh_f.norm(dim=-1) + 1e-2)
        cos_sim = dot / denom
        pen = F.relu(cos_sim - margin).pow(2)
        if valid_slot is not None:
            vf = valid_slot.float()
            return (pen * vf).sum() / vf.sum().clamp_min(1.0)
        return pen.mean()


class OmegaVariant(str, Enum):
    X8_DUAL_UNEMBED_ATRD = "x8_dual_unembed_atrd"
    G4_OMEGA = "g4_omega"
    G4_NO_VRES = "g4_no_vres"
    G4_NO_SPARSE_REFINER = "g4_no_sparse_refiner"


@dataclass(frozen=True, slots=True)
class OmegaBackboneConfig:
    name: str
    variant: OmegaVariant
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    ffn_hidden: int
    refiner_ffn: int
    max_seq_len: int = 32_768
    rope_theta: float = 500_000.0
    chunk_size: int = 32
    scam_slots: int = 32
    global_attn_every: int = 4
    tie_embeddings: bool = True
    use_gated_vres: bool = True
    use_latent_refiner: bool = True
    target_routed_frac: float = 0.25
    anticirc_weight: float = 0.02

    def __post_init__(self) -> None:
        if self.d_model % 2 != 0 or self.n_heads * self.head_dim != self.d_model or self.head_dim % 2 != 0:
            raise ValueError("Invalid d_model / n_heads / head_dim geometry")
        if self.n_kv_heads <= 0 or self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")


@dataclass(frozen=True, slots=True)
class OmegaCensus:
    embedding_parameters: int
    operator_parameters: int
    synthesis_parameters: int
    refiner_parameters: int
    norm_parameters: int
    total_deployed_parameters: int
    state_bytes_at_32768_bf16: int
    unembed_passes: int
    true_flops_per_token_4k: int
    within_strict_4b_limit: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_x8_global_layer(layer_idx: int, n_layers: int, every: int) -> bool:
    return layer_idx == (n_layers - 1) or ((layer_idx + 1) % every == 0)


def compute_omega_census(cfg: OmegaBackboneConfig, *, enforce_strict_4b: bool = True) -> OmegaCensus:
    d = cfg.d_model
    h = cfg.n_heads
    hd = cfg.head_dim
    kv_d = cfg.n_kv_heads * hd
    emb = cfg.vocab_size * d
    unemb = 0 if cfg.tie_embeddings else cfg.vocab_size * d

    op_total = 0
    syn_total = 0
    state_32k = 0
    unembed_passes = 2 if cfg.variant == OmegaVariant.X8_DUAL_UNEMBED_ATRD else 1

    if cfg.variant == OmegaVariant.X8_DUAL_UNEMBED_ATRD:
        for i in range(cfg.n_layers):
            if _is_x8_global_layer(i, cfg.n_layers, cfg.global_attn_every):
                # q_proj(d^2) + k_proj(d*kv_d) + v_proj(d*kv_d) + o_proj(d^2) + q/k rmsnorm(2*hd) + v_gate(i>=1)
                vg = h * (d + 1) if i >= 1 else 0
                op_total += 2 * d * d + 2 * d * kv_d + 2 * hd + vg
                state_32k += 2 * 2 * kv_d * 32_768
            else:
                # key_shift(4*d) + q/k/v/o(4*d^2) + q/k rmsnorm(2*hd) + gates(2*h*(d+1)) + v_gate(i>=1)
                vg = h * (d + 1) if i >= 1 else 0
                op_total += 4 * d + 4 * d * d + 2 * hd + 2 * h * (d + 1) + vg
                state_32k += 4 * h * hd * hd + 6 * d
            syn_total += 3 * d * cfg.ffn_hidden
        refiner_params = 2 * d * d + 3 * d * cfg.refiner_ffn + 3 * d + hd
    else:
        h_met = max(1, h // 2)
        d_met = h_met * hd
        d_scam = max(2, d // 4)
        for i in range(cfg.n_layers):
            depth_p = (i + 1) * (d + 1) if i >= 1 else 0
            wave_p = 2 * d * d + 20 * d
            vg = h_met * (d + 1) if (cfg.use_gated_vres and i >= 1) else 0
            met_p = 4 * d + 5 * d * d_met + 2 * hd + vg + 3 * h_met * (d + 1)
            scam_p = 4 * d * d_scam + d + 1
            op_total += depth_p + wave_p + met_p + scam_p + d * d
            syn_total += 3 * d * cfg.ffn_hidden + 2 * d
            state_32k += (4 * h_met * hd * hd + 6 * d) + (2 * 340 * d) + cfg.scam_slots * (4 * d_scam + 4)
        if cfg.use_latent_refiner:
            # score_norm(d) + score_proj(d+1) + step_emb(d) + n1(d) + n2(d) + tok_anchor(2*d*d) + ffn(3*d*ref_ffn)
            refiner_params = 5 * d + 1 + 2 * d * d + 3 * d * cfg.refiner_ffn
        else:
            refiner_params = 0

    norms = (2 * cfg.n_layers + 1) * d
    deployed = emb + unemb + op_total + syn_total + refiner_params + norms
    if enforce_strict_4b and deployed >= HARD_MAX_DEPLOYED_PARAMETERS:
        raise ParameterBudgetError(
            f"{cfg.name} deployed parameters {deployed:,} >= strict 4B ceiling {HARD_MAX_DEPLOYED_PARAMETERS:,}"
        )

    unemb_flops = 2 * unembed_passes * cfg.vocab_size * d
    ref_active_flops = int(cfg.target_routed_frac * 2 * refiner_params) if refiner_params > 0 else 0
    core_lin_flops = 2 * (op_total + syn_total)
    if cfg.variant == OmegaVariant.X8_DUAL_UNEMBED_ATRD:
        n_g = sum(1 for i in range(cfg.n_layers) if _is_x8_global_layer(i, cfg.n_layers, cfg.global_attn_every))
        n_d = cfg.n_layers - n_g
        ctx_flops = 4 * n_g * d * 4096 + 8 * n_d * h * hd * hd
    else:
        h_met = max(1, h // 2)
        d_scam = max(2, d // 4)
        ctx_flops = 2 * cfg.n_layers * (20 * d + 4 * h_met * hd * hd + 2 * d_scam * max(1, cfg.scam_slots))

    return OmegaCensus(
        embedding_parameters=emb,
        operator_parameters=op_total,
        synthesis_parameters=syn_total,
        refiner_parameters=refiner_params,
        norm_parameters=norms,
        total_deployed_parameters=deployed,
        state_bytes_at_32768_bf16=state_32k,
        unembed_passes=unembed_passes,
        true_flops_per_token_4k=unemb_flops + core_lin_flops + ref_active_flops + ctx_flops,
        within_strict_4b_limit=deployed < HARD_MAX_DEPLOYED_PARAMETERS,
    )


class _SharpDeltaVResOperator(nn.Module):
    """Contractive beta in (0,1) Gated-Delta with QK-only KeyShift, QK-RMSNorm, and Layer-0 Gated-VRes."""

    def __init__(self, d_model: int, n_heads: int, head_dim: int, layer_idx: int, chunk_size: int = 32) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.key_shift = CausalKeyShift(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = DiscoveryRMSNorm(head_dim)
        self.k_norm = DiscoveryRMSNorm(head_dim)
        self.gate_proj = nn.Linear(d_model, 2 * n_heads, bias=True)
        self.v_gate = nn.Linear(d_model, n_heads, bias=True) if layer_idx >= 1 else None

    def forward(
        self, x: Tensor, v0: Tensor | None, state: dict[str, Any] | None = None
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        bsz, t_len, d_model = x.shape
        x_addr, next_shift = self.key_shift(x, state.get("shift") if state is not None else None)
        q = self.q_norm(self.q_proj(x_addr).view(bsz, t_len, self.n_heads, self.head_dim)).transpose(1, 2)
        k = F.normalize(
            self.k_norm(self.k_proj(x_addr).view(bsz, t_len, self.n_heads, self.head_dim)).float(), dim=-1, eps=1e-6
        ).transpose(1, 2)
        v = self.v_proj(x).view(bsz, t_len, self.n_heads, self.head_dim).transpose(1, 2)
        if v0 is None:
            v0 = v
        elif self.v_gate is not None:
            alpha = torch.sigmoid(self.v_gate(x).float()).transpose(1, 2).unsqueeze(-1).to(v.dtype)
            v = alpha * v + (1.0 - alpha) * v0.to(v.dtype)

        gates = self.gate_proj(x).view(bsz, t_len, self.n_heads, 2).permute(0, 2, 1, 3).float()
        log_gamma = -F.softplus(gates[..., 0]) * 0.25
        beta = torch.sigmoid(gates[..., 1])
        init_s = state["s"] if state is not None else None
        y, s_next = _run_symplectic_wy_scan(None, k, v.float(), q.float(), None, beta, log_gamma, init_s, self.chunk_size)
        out = self.o_proj(y.to(x.dtype).transpose(1, 2).contiguous().view(bsz, t_len, d_model))
        return out, v0, {"s": s_next, "shift": next_shift}


class _GQAVResAnchorOperator(nn.Module):
    """Global GQA with QK-RMSNorm, RoPE, Per-Head Gated-VRes, and causal K/V export for Pass-2."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int, theta: float, layer_idx: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.theta = theta
        kv_dim = n_kv_heads * head_dim
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = DiscoveryRMSNorm(head_dim)
        self.k_norm = DiscoveryRMSNorm(head_dim)
        self.v_gate = nn.Linear(d_model, n_heads, bias=True) if layer_idx >= 1 else None

    def forward(
        self, x: Tensor, v0: Tensor | None, state: dict[str, Tensor] | None = None
    ) -> tuple[Tensor, Tensor, dict[str, Tensor], Tensor, Tensor]:
        bsz, t_len, d_model = x.shape
        q = self.q_norm(self.q_proj(x).view(bsz, t_len, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(bsz, t_len, self.n_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(bsz, t_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        offset = int(state["k"].shape[2]) if state is not None and "k" in state else 0
        q = apply_rope(q, seq_offset=offset, theta=self.theta)
        k = apply_rope(k, seq_offset=offset, theta=self.theta)
        rep = self.n_heads // self.n_kv_heads
        k_exp = k.repeat_interleave(rep, dim=1) if rep > 1 else k
        v_exp = v.repeat_interleave(rep, dim=1) if rep > 1 else v
        if v0 is None:
            v0 = v_exp
        elif self.v_gate is not None:
            alpha = torch.sigmoid(self.v_gate(x).float()).transpose(1, 2).unsqueeze(-1).to(v_exp.dtype)
            v_exp = alpha * v_exp + (1.0 - alpha) * v0.to(v_exp.dtype)

        if state is not None and "k" in state:
            k_full = torch.cat([state["k"], k_exp], dim=2)
            v_full = torch.cat([state["v"], v_exp], dim=2)
        else:
            k_full, v_full = k_exp, v_exp
        attn_out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=(state is None))
        out = self.o_proj(attn_out.transpose(1, 2).contiguous().view(bsz, t_len, d_model))
        return out, v0, {"k": k_full, "v": v_full}, k_full, v_full


class _SparseCausalCrossRefiner(nn.Module):
    """Shared strictly-causal Pass-2 sparse token refiner parameterized over `(d_model, d_kv, n_heads)`."""

    def __init__(self, d_model: int, ref_heads: int, head_dim: int, ref_ffn: int, target_frac: float = 0.25) -> None:
        super().__init__()
        self.d_model = d_model
        self.ref_heads = ref_heads
        self.head_dim = head_dim
        self.target_frac = target_frac
        q_dim = ref_heads * head_dim
        self.step_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.n1 = DiscoveryRMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, q_dim, bias=False)
        self.q_norm = DiscoveryRMSNorm(head_dim)
        self.o_proj = nn.Linear(q_dim, d_model, bias=False)
        self.n2 = DiscoveryRMSNorm(d_model)
        self.gu_proj = nn.Linear(d_model, 2 * ref_ffn, bias=False)
        self.dn_proj = nn.Linear(ref_ffn, d_model, bias=False)
        self.register_buffer("ema_tau", torch.tensor(0.5))
        self.register_buffer("ema_inited", torch.tensor(False))

    def forward(
        self,
        x1: Tensor,
        delta_h1: Tensor,
        k_mem: Tensor,
        v_mem: Tensor,
        uncertainty: Tensor,
        *,
        temp_scale: float = 6.0,
    ) -> tuple[Tensor, float, Tensor]:
        bsz, t_len, d_model = x1.shape
        u_f = uncertainty.float()
        if self.training:
            with torch.no_grad():
                q_val = torch.quantile(u_f.detach(), 1.0 - self.target_frac)
                if not bool(self.ema_inited.item()):
                    self.ema_tau.copy_(q_val)
                    self.ema_inited.fill_(True)
                else:
                    self.ema_tau.lerp_(q_val, 0.05)

        tau = self.ema_tau
        hard_mask = u_f > tau
        routed_frac = float(hard_mask.float().mean().detach())
        if not bool(hard_mask.any()):
            return x1, 0.0, x1.sum() * 0.0

        counts = hard_mask.sum(dim=-1)
        raw_max_k = int(counts.max().item())
        max_k = min(t_len, max(1, ((raw_max_k + 15) // 16) * 16)) if t_len >= 16 else raw_max_k
        order_scores = hard_mask.int() * (t_len + 1) + torch.arange(t_len, device=x1.device).unsqueeze(0)
        topk_idx = torch.topk(order_scores, k=max_k, dim=-1).indices
        topk_idx, _ = torch.sort(topk_idx, dim=-1)
        valid_slot = hard_mask.gather(1, topk_idx)

        idx_d = topk_idx.unsqueeze(-1).expand(bsz, max_k, d_model)
        x_sel = x1.gather(1, idx_d) + self.step_emb.to(x1.dtype)
        q_sel = self.q_norm(
            self.q_proj(self.n1(x_sel)).view(bsz, max_k, self.ref_heads, self.head_dim)
        ).transpose(1, 2)

        all_pos = torch.arange(t_len, device=x1.device).view(1, 1, 1, t_len)
        q_pos = topk_idx.view(bsz, 1, max_k, 1)
        attn_mask = q_pos >= all_pos  # [B, 1, K_max, T], strictly causal!

        with _no_autocast_ctx(x1.device):
            scores = torch.matmul(q_sel.float(), k_mem.float().transpose(-2, -1)) * (self.head_dim ** -0.5)
            scores = scores.masked_fill(~attn_mask, -1e9)
            probs = F.softmax(scores, dim=-1)
            y_attn = torch.matmul(probs, v_mem.float())
        dx_attn = self.o_proj(y_attn.to(x1.dtype).transpose(1, 2).contiguous().view(bsz, max_k, self.ref_heads * self.head_dim))
        x_mid = x_sel + dx_attn
        g_ff, u_ff = self.gu_proj(self.n2(x_mid)).chunk(2, dim=-1)
        with _no_autocast_ctx(x1.device):
            swi_f = (F.silu(g_ff.float()) * u_ff.float()).clamp(-1e4, 1e4).to(x1.dtype)
        dx_tot = dx_attn + self.dn_proj(swi_f)

        dh1_sel = delta_h1.gather(1, idx_d).float()
        anticirc = _smooth_anticirc_cosine(dx_tot, dh1_sel, valid_slot=valid_slot)

        soft_w = torch.sigmoid((u_f - tau) * temp_scale) * u_f.clamp(0.0, 1.0)
        gate = torch.where(hard_mask, soft_w, torch.zeros_like(soft_w))
        gate_sel = gate.gather(1, topk_idx) * valid_slot.to(gate.dtype)
        scaled_dx = torch.where(
            valid_slot.unsqueeze(-1),
            dx_tot.float() * gate_sel.unsqueeze(-1),
            dx_tot.float().new_zeros(()),
        ).to(x1.dtype)
        x2 = x1.clone()
        x2.scatter_add_(1, idx_d, scaled_dx)
        return x2, routed_frac, anticirc


class X8DualUnembedModel(nn.Module):
    """Exact reproduction of X8 Margin-ATRD with honest dual vocabulary unembedding accounting."""

    def __init__(self, cfg: OmegaBackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.census = compute_omega_census(cfg, enforce_strict_4b=True)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        for i in range(cfg.n_layers):
            self.norms1.append(DiscoveryRMSNorm(cfg.d_model))
            self.norms2.append(DiscoveryRMSNorm(cfg.d_model))
            if _is_x8_global_layer(i, cfg.n_layers, cfg.global_attn_every):
                self.blocks.append(_GQAVResAnchorOperator(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.rope_theta, i))
            else:
                self.blocks.append(_SharpDeltaVResOperator(cfg.d_model, cfg.n_heads, cfg.head_dim, i, cfg.chunk_size))
            self.ffns.append(SwiGLUSynthesis(cfg.d_model, cfg.ffn_hidden))
        self.norm_f = DiscoveryRMSNorm(cfg.d_model)
        self.refiner = _SparseCausalCrossRefiner(cfg.d_model, cfg.n_heads, cfg.head_dim, cfg.refiner_ffn, cfg.target_routed_frac)
        self._init_weights()

    def _init_weights(self) -> None:
        res_scale = 1.0 / math.sqrt(2.0 * self.cfg.n_layers)
        emb_std = min(0.06, 1.0 / math.sqrt(self.cfg.d_model))
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, mean=0.0, std=emb_std)
            elif isinstance(mod, nn.Linear):
                std = min(0.06, 0.8 / math.sqrt(max(1, mod.weight.shape[1])))
                if name.endswith(("o_proj", "w_down")):
                    std *= res_scale
                nn.init.normal_(mod.weight, mean=0.0, std=std)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.Conv1d):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                if "key_shift" in name and mod.weight.shape[-1] >= 2:
                    with torch.no_grad():
                        mod.weight[:, 0, -1] += 0.7
                        mod.weight[:, 0, -2] += 0.3
        for op in self.blocks:
            if hasattr(op, "v_gate") and op.v_gate is not None:
                with torch.no_grad():
                    op.v_gate.weight.zero_()
                    op.v_gate.bias.fill_(1.0)
            if hasattr(op, "gate_proj") and op.gate_proj.bias is not None:
                with torch.no_grad():
                    op.gate_proj.bias.view(self.cfg.n_heads, 2)[:, 0].fill_(-3.0)
        with torch.no_grad():
            self.refiner.o_proj.weight.zero_()
            self.refiner.dn_proj.weight.zero_()

    def deployed_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids: Tensor, targets: Tensor | None = None, return_states: bool = False) -> ModelOutput:
        x0 = self.tok_emb(input_ids)
        hidden = x0
        v0: Tensor | None = None
        k_anchor: Tensor | None = None
        v_anchor: Tensor | None = None
        for n1, op, n2, ffn in zip(self.norms1, self.blocks, self.norms2, self.ffns):
            if isinstance(op, _GQAVResAnchorOperator):
                op_out, v0, _, k_anchor, v_anchor = op(n1(hidden), v0)
            else:
                op_out, v0, _ = op(n1(hidden), v0)
            hidden = hidden + op_out
            hidden = hidden + ffn(n2(hidden))

        # Dual unembedding #1 (16.78M-param vocab GEMM inside X8!)
        with torch.no_grad():
            lg1 = F.linear(self.norm_f(hidden), self.tok_emb.weight).float()
            top2 = torch.topk(lg1, k=2, dim=-1).values
            lse = torch.logsumexp(lg1, dim=-1, keepdim=True)
            probs = (top2 - lse).exp()
            margin_unc = 1.0 - (probs[..., 0] - probs[..., 1])

        anticirc = hidden.new_zeros(())
        if k_anchor is not None and v_anchor is not None:
            hidden, _, anticirc = self.refiner(hidden, (hidden - x0).detach(), k_anchor, v_anchor, margin_unc, temp_scale=8.0)

        # Dual unembedding #2
        logits = F.linear(self.norm_f(hidden), self.tok_emb.weight)
        loss = None
        ce_loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.float().reshape(-1, self.cfg.vocab_size), targets.reshape(-1), ignore_index=-1)
            loss = ce_loss + self.cfg.anticirc_weight * anticirc
        return ModelOutput(
            logits=logits,
            loss=loss,
            ce_loss=ce_loss,
            mean_uncertainty=margin_unc.mean(),
            token_uncertainty=margin_unc,
            states=None,
        )


class _OmegaHMWRBlock(nn.Module):
    """Tri-scale Dyadic Wave + beta in (0,1.98) WY (with QK-RMSNorm & Gated-VRes) + Shifted SCAM + Depth-Attention."""

    DILATIONS = (1, 4, 16, 64)
    KERNEL_SIZE = 5

    def __init__(self, cfg: OmegaBackboneConfig, layer_idx: int) -> None:
        super().__init__()
        d = cfg.d_model
        h_met = max(1, cfg.n_heads // 2)
        hd = cfg.head_dim
        d_met = h_met * hd
        d_scam = max(2, d // 4)
        self.d_model = d
        self.n_heads = h_met
        self.head_dim = hd
        self.d_met = d_met
        self.chunk_size = cfg.chunk_size
        self.scam_slots = cfg.scam_slots
        self.use_gated_vres = cfg.use_gated_vres

        self.norm1 = DiscoveryRMSNorm(d)
        self.depth_router = nn.Linear(d, layer_idx + 1, bias=True) if layer_idx >= 1 else None
        self.key_shift = CausalKeyShift(d)
        self.wave_proj = nn.Linear(d, 2 * d, bias=False)
        self.wave_bands = nn.ModuleList(
            nn.Conv1d(d, d, kernel_size=self.KERNEL_SIZE, groups=d, dilation=dil, bias=False) for dil in self.DILATIONS
        )
        self.u_proj = nn.Linear(d, d_met, bias=False)
        self.k_proj = nn.Linear(d, d_met, bias=False)
        self.v_proj = nn.Linear(d, d_met, bias=False)
        self.q_proj = nn.Linear(d, d_met, bias=False)
        self.q_norm = DiscoveryRMSNorm(hd)
        self.k_norm = DiscoveryRMSNorm(hd)
        self.v_gate = nn.Linear(d, h_met, bias=True) if (cfg.use_gated_vres and layer_idx >= 1) else None
        self.met_o = nn.Linear(d_met, d, bias=False)
        self.gate_proj = nn.Linear(d, 3 * h_met, bias=True)

        self.scam_q = nn.Linear(d, d_scam, bias=False)
        self.scam_k = nn.Linear(d, d_scam, bias=False)
        self.scam_v = nn.Linear(d, d_scam, bias=False)
        self.scam_o = nn.Linear(d_scam, d, bias=False)
        self.surp_gate = nn.Linear(d, 1, bias=True)
        self.o_proj = nn.Linear(d, d, bias=False)

        self.norm2 = DiscoveryRMSNorm(d)
        self.synthesis = SwiGLUSynthesis(d, cfg.ffn_hidden)
        self.adapt_gate = nn.Linear(1, d, bias=True)

    def _project_terms(self, x: Tensor, x_addr: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        bsz, t_len, _ = x.shape
        u = F.normalize(self.u_proj(x_addr).view(bsz, t_len, self.n_heads, self.head_dim).float(), dim=-1, eps=1e-6).transpose(1, 2)
        k = F.normalize(
            self.k_norm(self.k_proj(x_addr).view(bsz, t_len, self.n_heads, self.head_dim)).float(), dim=-1, eps=1e-6
        ).transpose(1, 2)
        v = self.v_proj(x).view(bsz, t_len, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(self.q_proj(x_addr).view(bsz, t_len, self.n_heads, self.head_dim)).transpose(1, 2).float()
        gates = self.gate_proj(x).view(bsz, t_len, self.n_heads, 3).permute(0, 2, 1, 3).float()
        beta = 1.98 * torch.sigmoid(gates[..., 0])
        tau = torch.sigmoid(gates[..., 1])
        log_gamma = -F.softplus(gates[..., 2]) * 0.20
        return u, k, v, q, beta, tau, log_gamma

    def forward(
        self,
        x: Tensor,
        v0: Tensor | None,
        depth_history: list[Tensor],
        state: dict[str, Any] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Any], Tensor, Tensor]:
        bsz, t_len, _ = x.shape
        if self.depth_router is not None:
            weights = F.softmax(self.depth_router(self.norm1(x)).float(), dim=-1).to(x.dtype)
            x_in = torch.einsum("btk,btkd->btd", weights, torch.stack(depth_history, dim=2))
        else:
            x_in = x

        xn = self.norm1(x_in)
        x_addr, next_shift = self.key_shift(xn, state.get("shift") if state is not None else None)
        re_in, re_ph = self.wave_proj(xn).chunk(2, dim=-1)
        wave_re, next_ring = _apply_dilated_wave_bank(
            re_in.transpose(1, 2), state.get("wave_ring") if state is not None else None,
            self.DILATIONS, self.wave_bands, self.KERNEL_SIZE,
        )
        wave_re = wave_re.transpose(1, 2)
        prev_env = state.get("prev_wave") if state is not None else None
        if prev_env is None:
            lagged = F.pad(wave_re[:, :-1, :], (0, 0, 1, 0))
        else:
            lagged = torch.cat([prev_env, wave_re[:, :-1, :]], dim=1) if t_len > 1 else prev_env
        pred_err = (re_in - lagged).float().pow(2).mean(dim=-1).add(1e-6).sqrt()
        surp_score = pred_err + F.softplus(self.surp_gate(xn).squeeze(-1).float())
        wave_out = wave_re * torch.tanh(re_ph)

        u, k, v_raw, q, beta, tau, log_gamma = self._project_terms(xn, x_addr)
        if v0 is None:
            v0_next = v_raw
            v = v_raw
        else:
            v0_next = v0
            if self.v_gate is not None:
                alpha = torch.sigmoid(self.v_gate(xn).float()).transpose(1, 2).unsqueeze(-1).to(v_raw.dtype)
                v = alpha * v_raw + (1.0 - alpha) * v0.to(v_raw.dtype)
            else:
                v = v_raw

        y_met, s_next = _run_symplectic_wy_scan(
            u, k, v.float(), q, beta, tau, log_gamma, state.get("s") if state is not None else None, min(self.chunk_size, 16)
        )
        met_out = self.met_o(y_met.to(xn.dtype).transpose(1, 2).contiguous().view(bsz, t_len, self.d_met))

        q_s = self.scam_q(x_addr)
        k_s = self.scam_k(x_addr)
        v_s = self.scam_v(xn)
        scam_state = state.get("scam") if state is not None else None
        if t_len == 1:
            scam_y, next_scam = scam_step_attention(q_s[:, 0], k_s[:, 0], v_s[:, 0], surp_score[:, 0], scam_state, self.scam_slots)
            scam_y = scam_y.unsqueeze(1)
        else:
            scam_y, next_scam = scam_causal_topk_attention(q_s, k_s, v_s, surp_score, k_slots=self.scam_slots, mem_state=scam_state)
        scam_out = self.scam_o(scam_y)

        retr_rms = (y_met.pow(2).mean(dim=(1, 3)) + scam_y.float().pow(2).mean(dim=-1) + 1e-3).sqrt()
        unc = (surp_score / retr_rms).to(xn.dtype)  # [B, T]
        x_mid = x_in + self.o_proj(wave_out + met_out + scam_out)
        u_feat = torch.log1p(unc.float().unsqueeze(-1).clamp(min=0.0)).to(xn.dtype)
        gain = 0.5 + torch.sigmoid(self.adapt_gate(u_feat))
        h_out = x_mid + self.synthesis(self.norm2(x_mid)) * gain
        next_state = {"s": s_next, "shift": next_shift, "wave_ring": next_ring, "prev_wave": wave_re[:, -1:, :].detach(), "scam": next_scam}
        return h_out, unc, v0_next, next_state, k, v.float()


class _LatentTokenRefiner(nn.Module):
    """O(1)-state, zero-vocab-GEMM Pass-2 conditional refinement over hard tokens (token-exact duality)."""

    def __init__(self, d_model: int, ref_ffn: int, target_frac: float = 0.25) -> None:
        super().__init__()
        self.d_model = d_model
        self.target_frac = target_frac
        self.step_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.n1 = DiscoveryRMSNorm(d_model)
        self.mix_gate = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.n2 = DiscoveryRMSNorm(d_model)
        self.gu_proj = nn.Linear(d_model, 2 * ref_ffn, bias=False)
        self.dn_proj = nn.Linear(ref_ffn, d_model, bias=False)
        self.register_buffer("ema_tau", torch.tensor(0.5))
        self.register_buffer("ema_inited", torch.tensor(False))

    def forward(
        self,
        x1: Tensor,
        x0: Tensor,
        delta_h1: Tensor,
        uncertainty: Tensor,
    ) -> tuple[Tensor, float, Tensor]:
        bsz, t_len, d_model = x1.shape
        u_f = uncertainty.float()
        if self.training:
            with torch.no_grad():
                q_val = torch.quantile(u_f.detach(), 1.0 - self.target_frac)
                if not bool(self.ema_inited.item()):
                    self.ema_tau.copy_(q_val)
                    self.ema_inited.fill_(True)
                else:
                    self.ema_tau.lerp_(q_val, 0.05)

        tau = self.ema_tau
        hard_mask = u_f > tau
        routed_frac = float(hard_mask.float().mean().detach())
        if not bool(hard_mask.any()):
            return x1, 0.0, x1.sum() * 0.0

        flat_mask = hard_mask.reshape(-1)
        idx_hard = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1)
        x_sel = x1.reshape(-1, d_model)[idx_hard] + self.step_emb.to(x1.dtype).view(1, d_model)
        x0_sel = x0.reshape(-1, d_model)[idx_hard]
        dh1_sel = delta_h1.reshape(-1, d_model)[idx_hard].float()

        anchor_gate = torch.sigmoid(self.mix_gate(self.n1(x_sel)))
        dx_anchor = self.o_proj(anchor_gate * x0_sel)
        x_mid = x_sel + dx_anchor
        g_ff, u_ff = self.gu_proj(self.n2(x_mid)).chunk(2, dim=-1)
        with _no_autocast_ctx(x1.device):
            swi_f = (F.silu(g_ff.float()) * u_ff.float()).clamp(-1e4, 1e4).to(x1.dtype)
        dx_tot = dx_anchor + self.dn_proj(swi_f)

        anticirc = _smooth_anticirc_cosine(dx_tot, dh1_sel)

        soft_w = (torch.sigmoid((u_f - tau) * 6.0) * u_f.clamp(0.0, 1.0)).reshape(-1)[idx_hard].unsqueeze(-1)
        scaled_dx = (dx_tot.float() * soft_w).to(x1.dtype)
        x2_flat = x1.reshape(-1, d_model).clone()
        x2_flat.index_add_(0, idx_hard, scaled_dx)
        return x2_flat.view(bsz, t_len, d_model), routed_frac, anticirc


class G4OmegaModel(nn.Module):
    """G4-Omega: Capacity-balanced G3 + QK-RMSNorm + Gated-VRes + Unembedding-Free LatentTokenRefiner."""

    def __init__(self, cfg: OmegaBackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.census = compute_omega_census(cfg, enforce_strict_4b=True)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([_OmegaHMWRBlock(cfg, i) for i in range(cfg.n_layers)])
        self.norm_f = DiscoveryRMSNorm(cfg.d_model)
        if cfg.use_latent_refiner:
            self.score_norm = DiscoveryRMSNorm(cfg.d_model)
            self.score_proj = nn.Linear(cfg.d_model, 1, bias=True)
            self.refiner = _LatentTokenRefiner(cfg.d_model, cfg.refiner_ffn, cfg.target_routed_frac)
        else:
            self.score_norm = None
            self.score_proj = None
            self.refiner = None
        self._init_weights()

    def _init_weights(self) -> None:
        res_scale = 1.0 / math.sqrt(2.0 * self.cfg.n_layers)
        emb_std = min(0.06, 1.0 / math.sqrt(self.cfg.d_model))
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, mean=0.0, std=emb_std)
            elif isinstance(mod, nn.Linear):
                std = min(0.06, 0.8 / math.sqrt(max(1, mod.weight.shape[1])))
                if name.endswith(("o_proj", "w_down")):
                    std *= res_scale
                nn.init.normal_(mod.weight, mean=0.0, std=std)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.Conv1d):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                if "key_shift" in name and mod.weight.shape[-1] >= 2:
                    with torch.no_grad():
                        mod.weight[:, 0, -1] += 0.7
                        mod.weight[:, 0, -2] += 0.3
        for b in self.blocks:
            if b.depth_router is not None:
                nn.init.normal_(b.depth_router.weight, mean=0.0, std=0.01)
                with torch.no_grad():
                    b.depth_router.bias.zero_()
                    b.depth_router.bias[-1] = 2.5
            with torch.no_grad():
                b.gate_proj.bias.view(b.n_heads, 3)[:, 2].fill_(-3.0)
                if b.v_gate is not None:
                    b.v_gate.weight.zero_()
                    b.v_gate.bias.fill_(1.0)
        if self.refiner is not None:
            with torch.no_grad():
                self.refiner.o_proj.weight.zero_()
                self.refiner.dn_proj.weight.zero_()

    def deployed_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        states: list[dict[str, Any]] | None = None,
        return_states: bool = False,
    ) -> ModelOutput:
        x0 = self.tok_emb(input_ids)
        hidden = x0
        v0: Tensor | None = None
        depth_history = [x0]
        next_states: list[dict[str, Any]] | None = [] if return_states else None
        unc_all: list[Tensor] = []
        last_k = last_v = None

        for i, b in enumerate(self.blocks):
            st_in = states[i] if states is not None else None
            hidden, unc, v0, st_out, last_k, last_v = b(hidden, v0, depth_history, state=st_in)
            depth_history.append(hidden)
            unc_all.append(unc.float())
            if next_states is not None:
                next_states.append(st_out)

        mem_unc = torch.stack(unc_all, dim=0).mean(dim=0)  # [B, T]
        anticirc = hidden.new_zeros(())
        lat_logit: Tensor | None = None
        if self.refiner is not None and self.score_norm is not None and self.score_proj is not None:
            # Zero-vocab-GEMM strictly causal self-calibrated latent difficulty score (works identically for t_len>=1!)
            lat_logit = self.score_proj(self.score_norm(hidden.detach())).squeeze(-1).float()
            mem_prob = torch.sigmoid(torch.log1p(mem_unc.clamp(min=0.0)) - 0.5)
            u_cal = 0.5 * mem_prob + 0.5 * torch.sigmoid(lat_logit)
            hidden, _, anticirc = self.refiner(hidden, x0, (hidden - x0).detach(), u_cal.detach())
            tok_unc = mem_unc * (0.5 + torch.sigmoid(lat_logit))
        else:
            tok_unc = mem_unc

        logits = F.linear(self.norm_f(hidden), self.tok_emb.weight)
        loss = ce_loss = None
        if targets is not None:
            ce_per_tok = F.cross_entropy(
                logits.float().reshape(-1, self.cfg.vocab_size),
                targets.reshape(-1),
                ignore_index=-1,
                reduction="none",
            )
            valid = targets.reshape(-1) >= 0
            ce_loss = ce_per_tok[valid].mean() if bool(valid.any()) else logits.float().sum() * 0.0
            loss = ce_loss + self.cfg.anticirc_weight * anticirc
            if lat_logit is not None and int(valid.sum().item()) >= 4:
                with torch.no_grad():
                    ce_val = ce_per_tok[valid].detach()
                    thresh = torch.quantile(ce_val, 1.0 - self.cfg.target_routed_frac)
                    hard_tgt = (ce_val > thresh).float()
                cal_loss = F.binary_cross_entropy_with_logits(lat_logit.reshape(-1)[valid], hard_tgt)
                loss = loss + 0.05 * cal_loss
        return ModelOutput(
            logits=logits,
            loss=loss,
            ce_loss=ce_loss,
            mean_uncertainty=tok_unc.mean(),
            token_uncertainty=tok_unc,
            states=next_states,
        )

    def forward_step(self, token_id: Tensor, states: list[dict[str, Any]] | None = None) -> ModelOutput:
        if token_id.ndim == 1:
            token_id = token_id.unsqueeze(1)
        return self.forward(token_id, states=states, return_states=True)


def build_omega_model(cfg: OmegaBackboneConfig) -> nn.Module:
    if cfg.variant == OmegaVariant.X8_DUAL_UNEMBED_ATRD:
        return X8DualUnembedModel(cfg)
    return G4OmegaModel(cfg)


def _solve_omega_ffn(base_cfg: OmegaBackboneConfig, target: int, align: int = 16) -> OmegaBackboneConfig:
    best = base_cfg
    best_err = abs(compute_omega_census(base_cfg, enforce_strict_4b=False).total_deployed_parameters - target)
    c0 = compute_omega_census(replace(base_cfg, ffn_hidden=align), enforce_strict_4b=False).total_deployed_parameters
    c1 = compute_omega_census(replace(base_cfg, ffn_hidden=2 * align), enforce_strict_4b=False).total_deployed_parameters
    slope = max(1, c1 - c0)
    approx = max(1, round(align + align * (target - c0) / slope))
    lo = max(align, ((approx // align) - 5) * align)
    hi = ((approx // align) + 6) * align
    for cand_ffn in range(lo, hi + 1, align):
        cand = replace(base_cfg, ffn_hidden=cand_ffn)
        tot = compute_omega_census(cand, enforce_strict_4b=False).total_deployed_parameters
        err = abs(tot - target)
        if err < best_err:
            best_err = err
            best = cand
    compute_omega_census(best, enforce_strict_4b=True)
    return best


def make_omega_suite(tier: str, *, vocab_size: int | None = None) -> dict[OmegaVariant, OmegaBackboneConfig]:
    """Build parameter-matched (within +-0.5% of B0) configs for all OMEGA-EXP-01 arms."""
    from nanolm_discovery.config import compute_parameter_census

    ref_b0 = make_iso_param_suite(tier, vocab_size=vocab_size)[ParadigmKind.B0_TRANSFORMER]
    target = compute_parameter_census(ref_b0).total_deployed_parameters
    align = 2 if tier == "mech" else (64 if tier == "flagship_4b" else 16)
    ref_ffn = max(16, ((ref_b0.d_model // 3) // align) * align) if tier != "mech" else 32
    suite: dict[OmegaVariant, OmegaBackboneConfig] = {}
    specs = (
        (OmegaVariant.X8_DUAL_UNEMBED_ATRD, True, True),
        (OmegaVariant.G4_OMEGA, True, True),
        (OmegaVariant.G4_NO_VRES, False, True),
        (OmegaVariant.G4_NO_SPARSE_REFINER, True, False),
    )
    for var, use_vr, use_ref in specs:
        base = OmegaBackboneConfig(
            name=f"{var.value}_{tier}",
            variant=var,
            vocab_size=ref_b0.vocab_size,
            d_model=ref_b0.d_model,
            n_layers=ref_b0.n_layers,
            n_heads=ref_b0.n_heads,
            n_kv_heads=ref_b0.n_kv_heads,
            head_dim=ref_b0.head_dim,
            ffn_hidden=ref_b0.ffn_hidden,
            refiner_ffn=ref_ffn if use_ref else 16,
            chunk_size=ref_b0.chunk_size,
            scam_slots=ref_b0.scam_slots,
            use_gated_vres=use_vr,
            use_latent_refiner=use_ref,
        )
        suite[var] = _solve_omega_ffn(base, target, align=align)
    return suite
