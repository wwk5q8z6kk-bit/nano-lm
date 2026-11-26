"""Recovered 47M Omega Track A model, with checkpoint-compatible causal routing.

Model structure derives from nanolm_omega/training/train_x9_tracka_cuda.py at
5c830ecf72648dc68ba31de89c5ce10e9b7ccbbf. Only reference PyTorch kernels
are used. Neither a pretrained checkpoint nor a training corpus is bundled.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .routing import cap_prefix, stock_prefix_end


@dataclass
class X9Config:
    vocab: int = 32768
    d: int = 512
    n_head: int = 8
    n_kv: int = 4
    ffn: int = 1376
    layout: str = "DDDADDDADA"
    seq: int = 1024
    rope_theta: float = 10000.0
    conv: int = 4
    chunk: int = 64
    tie: bool = True
    atrd_frac: float = 0.25
    copy_shield_tau: float = 0.55
    voc_lambda: float = 0.22
    anticirc_weight: float = 0.02
    routing_mode: str = "acex9"  # 'acex9' (explore-then-shield) | 'x8' (reimplemented X8 quantile control)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.w, self.eps)


def rope_cache(T: int, hd: int, theta: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd))
    f = torch.outer(torch.arange(T, device=device, dtype=torch.float32), inv)
    return f.cos(), f.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    c = cos[None, None, : x.shape[2]].to(x.dtype)
    s = sin[None, None, : x.shape[2]].to(x.dtype)
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1).flatten(-2)


class ShortConv(nn.Module):
    def __init__(self, ch: int, k: int = 4):
        super().__init__()
        self.k = k
        self.w = nn.Parameter(torch.randn(ch, 1, k) / math.sqrt(k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x.transpose(1, 2), (self.k - 1, 0))
        return F.silu(F.conv1d(padded, self.w, groups=x.shape[-1])).transpose(1, 2)


class QKCanonAttention(nn.Module):
    def __init__(self, c: X9Config):
        super().__init__()
        self.h, self.kv, self.hd = c.n_head, c.n_kv, c.d // c.n_head
        self.q = nn.Linear(c.d, self.h * self.hd, bias=False)
        self.k = nn.Linear(c.d, self.kv * self.hd, bias=False)
        self.v = nn.Linear(c.d, self.kv * self.hd, bias=False)
        self.o = nn.Linear(c.d, self.h * self.hd, bias=False)
        self.qn, self.kn = RMSNorm(self.hd), RMSNorm(self.hd)
        self.qk_conv = ShortConv((self.h + self.kv) * self.hd, c.conv)
        self.v_gate = nn.Linear(c.d, self.h, bias=True)
        nn.init.zeros_(self.v_gate.weight)
        nn.init.constant_(self.v_gate.bias, 1.0)

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor], v0: torch.Tensor | None, ret_kv: bool = False):
        B, T, _ = x.shape
        qk = self.qk_conv(torch.cat([self.q(x), self.k(x)], dim=-1))
        q_raw, k_raw = qk.split([self.h * self.hd, self.kv * self.hd], dim=-1)
        q = self.qn(q_raw.view(B, T, self.h, self.hd)).transpose(1, 2)
        k = self.kn(k_raw.view(B, T, self.kv, self.hd)).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        rep = self.h // self.kv
        k_exp = k.repeat_interleave(rep, 1)
        v_exp = v.repeat_interleave(rep, 1)
        if v0 is None:
            v0 = v_exp
        else:
            alpha = torch.sigmoid(self.v_gate(x)).transpose(1, 2).unsqueeze(-1)
            v_exp = alpha * v_exp + (1.0 - alpha) * v0.to(v_exp.dtype)
        y = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
        out = self.o(y.transpose(1, 2).reshape(B, T, -1))
        return out, v0, ((k_exp, v_exp) if ret_kv else None)


class SharpDeltaMixer(nn.Module):
    """Sharp-V Gated Delta-Rule using the checkpoint reference WY chunked scan."""

    def __init__(self, c: X9Config):
        super().__init__()
        self.h = c.n_head
        self.hd = c.d // c.n_head
        d = c.d
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.conv = ShortConv(2 * d, c.conv)
        self.a = nn.Linear(d, self.h, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.linspace(1.0, 16.0, self.h)))
        self.b = nn.Linear(d, self.h, bias=True)
        self.g = nn.Linear(d, d, bias=False)
        self.on = RMSNorm(self.hd)
        self.o = nn.Linear(d, d, bias=False)
        self.C = c.chunk
        self.v_gate = nn.Linear(d, self.h, bias=True)
        nn.init.zeros_(self.v_gate.weight)
        nn.init.constant_(self.v_gate.bias, 1.0)

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor], v0: torch.Tensor | None):
        B, T, D = x.shape
        H, hd, C = self.h, self.hd, self.C
        qk, v_raw = self.qkv(x).split([2 * D, D], dim=-1)
        q, k = self.conv(qk).split(D, dim=-1)
        q = F.normalize(q.view(B, T, H, hd).float(), dim=-1).transpose(1, 2) * (hd ** -0.5)
        k = F.normalize(k.view(B, T, H, hd).float(), dim=-1).transpose(1, 2)
        v = v_raw.view(B, T, H, hd).float().transpose(1, 2)
        if v0 is None:
            v0 = v
        else:
            alpha = torch.sigmoid(self.v_gate(x).float()).transpose(1, 2).unsqueeze(-1)
            v = alpha * v + (1.0 - alpha) * v0.float()

        loga = (-F.softplus(self.a(x).float()) * self.A_log.exp()).transpose(1, 2)
        beta = torch.sigmoid(self.b(x).float()).transpose(1, 2)

        pad = (-T) % C
        if pad:
            q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
            loga, beta = F.pad(loga, (0, pad)), F.pad(beta, (0, pad))
        N = (T + pad) // C
        q, k, v = (t.view(B, H, N, C, hd) for t in (q, k, v))
        la = loga.view(B, H, N, C).cumsum(-1)
        tri = torch.tril(torch.ones(C, C, dtype=torch.bool, device=x.device))
        decay = (la[..., :, None] - la[..., None, :]).masked_fill(~tri, -1e30).exp()
        bc = beta.view(B, H, N, C)
        kb = k * bc[..., None]
        strict = torch.tril(torch.ones(C, C, dtype=torch.bool, device=x.device), -1)
        Akk = (kb @ k.transpose(-1, -2) * decay).masked_fill(~strict, 0.0)
        eye = torch.eye(C, device=x.device)
        Tm = torch.linalg.solve_triangular(eye + Akk, eye.expand_as(Akk), upper=False)
        w_ = Tm @ (kb * la.exp()[..., None])
        u_ = Tm @ (v * bc[..., None])
        S = x.new_zeros(B, H, hd, hd, dtype=torch.float32)
        ys = []
        for n in range(N):
            qn, kn, la_n, dn = q[:, :, n], k[:, :, n], la[:, :, n], decay[:, :, n]
            u = u_[:, :, n] - w_[:, :, n] @ S
            attn = (qn @ kn.transpose(-1, -2)) * dn
            y = attn @ u + (qn * la_n.exp()[..., None]) @ S
            tail = (la_n[..., -1:] - la_n).exp()[..., None]
            S = la_n[..., -1].exp()[..., None, None] * S + (kn * tail).transpose(-1, -2) @ u
            ys.append(y)
        y = torch.stack(ys, 2).view(B, H, N * C, hd)[:, :, :T]
        y = self.on(y.transpose(1, 2).to(x.dtype)).reshape(B, T, D) * F.silu(self.g(x))
        return self.o(y), v0, None


class X9Block(nn.Module):
    def __init__(self, c: X9Config, kind: str):
        super().__init__()
        self.kind = kind
        self.n1, self.n2 = RMSNorm(c.d), RMSNorm(c.d)
        self.mix = QKCanonAttention(c) if kind == "A" else SharpDeltaMixer(c)
        self.gu = nn.Linear(c.d, 2 * c.ffn, bias=False)
        self.dn = nn.Linear(c.ffn, c.d, bias=False)

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor], v0: torch.Tensor | None, ret_kv: bool = False):
        if self.kind == "A":
            h, v0, kv = self.mix(self.n1(x), rope, v0, ret_kv=ret_kv)
        else:
            h, v0, kv = self.mix(self.n1(x), rope, v0)
        x = x + h
        g, u = self.gu(self.n2(x)).chunk(2, dim=-1)
        return x + self.dn(F.silu(g) * u), v0, kv


class ACEX9Refiner(nn.Module):
    """ACE-X9.2 Explore-Then-Shield Cross-Memory Refiner with Monotonic Margin Guard."""

    def __init__(self, c: X9Config):
        super().__init__()
        self.h = c.n_head
        self.hd = c.d // c.n_head
        self.d = c.d
        self.target_frac = c.atrd_frac
        self.copy_shield_tau = c.copy_shield_tau
        self.voc_lambda = c.voc_lambda
        self.routing_mode = c.routing_mode
        self.eval_policy = "causal"
        self.step_emb = nn.Parameter(torch.randn(1, 1, c.d) * 0.02)
        self.n1 = RMSNorm(c.d)
        self.q = nn.Linear(c.d, c.d, bias=False)
        self.qn = RMSNorm(self.hd)
        self.o = nn.Linear(c.d, c.d, bias=False)
        self.n2 = RMSNorm(c.d)
        ref_ffn = max(256, (c.ffn // 2 // 32) * 32)
        self.gu = nn.Linear(c.d, 2 * ref_ffn, bias=False)
        self.dn = nn.Linear(ref_ffn, c.d, bias=False)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.dn.weight)
        self.register_buffer("ema_u_tau", torch.tensor(0.85))
        self.register_buffer("ema_inited", torch.tensor(False))

    def forward(
        self,
        x1: torch.Tensor,
        delta_h1: torch.Tensor,
        k1_full: torch.Tensor,
        v1_full: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        u_margin: torch.Tensor,
        top8_mass: torch.Tensor,
        jsd_epistemic: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        B, T, D = x1.shape
        margin_gap = 1.0 - u_margin
        copy_shield = margin_gap >= self.copy_shield_tau  # Strictly protect sharp Pass-1 copy predictions!
        if self.routing_mode == "x8":
            copy_shield = torch.zeros_like(copy_shield)

        if self.training or self.routing_mode == "x8":
            # Explore-Then-Shield: quantile load-balancing over non-shielded tokens prevents Pass-2 starvation!
            # (EMA threshold is frozen at eval, exactly as in the original X8 CausalEntropyRefiner.)
            # Only later batches may observe this batch's threshold statistic.
            # The X8 control retains its original training and evaluation semantics.
            threshold = self.ema_u_tau.detach().clone() if self.routing_mode == "acex9" else self.ema_u_tau
            if self.training:
                with torch.no_grad():
                    masked_u = torch.where(copy_shield, torch.zeros_like(u_margin), u_margin)
                    q_val = torch.quantile(masked_u.detach().float(), 1.0 - self.target_frac)
                    if not bool(self.ema_inited.item()):
                        self.ema_u_tau.copy_(q_val)
                        self.ema_inited.fill_(True)
                    else:
                        self.ema_u_tau.lerp_(q_val, 0.05)
            hard_mask = (u_margin > threshold) & (~copy_shield)
        else:
            # Eval/Inference VoC > 0: only route structured epistemic competition, capped at 30% budget!
            voc = 0.70 * (u_margin * top8_mass) + 0.30 * jsd_epistemic - self.voc_lambda
            pos_mask = (voc > 0.0) & (~copy_shield) & (u_margin > self.ema_u_tau * 0.92)
            if self.eval_policy == "causal":
                hard_mask = cap_prefix(pos_mask)
            elif self.eval_policy == "stock-serving":
                hard_mask = stock_prefix_end(voc, pos_mask, copy_shield)
            else:
                raise ValueError(f"Unknown inference route policy {self.eval_policy!r}")

        routed_frac = float(hard_mask.float().mean().detach())
        shield_frac = float(copy_shield.float().mean().detach())
        if not bool(hard_mask.any()):
            return x1, {"routed_frac": 0.0, "copy_shield_frac": round(shield_frac, 4), "anticirc_loss": x1.sum() * 0.0}

        # 32-bucket aligned sparse gather (zero CUDA allocator fragmentation)
        counts = hard_mask.sum(dim=-1)
        raw_max_k = int(counts.max().item())
        max_k = min(T, max(1, ((raw_max_k + 31) // 32) * 32)) if T >= 32 else raw_max_k
        topk_idx = torch.topk(hard_mask.int() * (T + 1) + torch.arange(T, device=x1.device), k=max_k, dim=-1).indices
        topk_idx, _ = torch.sort(topk_idx, dim=-1)
        valid_slot = hard_mask.gather(1, topk_idx)

        idx_d = topk_idx.unsqueeze(-1).expand(B, max_k, D)
        x_sel = x1.gather(1, idx_d) + self.step_emb.to(x1.dtype)

        cos_all, sin_all = rope
        c_sel = cos_all[topk_idx].unsqueeze(1).to(x1.dtype)
        s_sel = sin_all[topk_idx].unsqueeze(1).to(x1.dtype)
        q_raw = self.qn(self.q(self.n1(x_sel)).view(B, max_k, self.h, self.hd)).transpose(1, 2)
        q1, q2 = q_raw[..., ::2], q_raw[..., 1::2]
        q_sel = torch.stack((q1 * c_sel - q2 * s_sel, q1 * s_sel + q2 * c_sel), -1).flatten(-2).to(k1_full.dtype)

        all_pos = torch.arange(T, device=x1.device).view(1, 1, 1, T)
        q_pos = topk_idx.view(B, 1, max_k, 1)
        attn_mask = q_pos >= all_pos

        y_attn = F.scaled_dot_product_attention(q_sel, k1_full, v1_full, attn_mask=attn_mask)
        dx_attn = self.o(y_attn.transpose(1, 2).reshape(B, max_k, D))
        x_mid = x_sel + dx_attn
        g_ffn, u_ffn = self.gu(self.n2(x_mid)).chunk(2, dim=-1)
        dx_total = dx_attn + self.dn(F.silu(g_ffn) * u_ffn)

        dh1_sel = delta_h1.gather(1, idx_d).float()
        cos_sim = F.cosine_similarity(dx_total.float(), dh1_sel, dim=-1, eps=1e-6)
        pen_tok = F.relu(cos_sim - 0.25).pow(2) * valid_slot.float()
        anticirc = pen_tok.sum() / valid_slot.float().sum().clamp_min(1.0)

        gate_sel = u_margin.clamp(0.0, 1.0).gather(1, topk_idx) * valid_slot.to(u_margin.dtype)
        scaled_dx = (dx_total.float() * gate_sel.unsqueeze(-1)).to(x1.dtype)
        x2 = x1.clone()
        x2.scatter_add_(1, idx_d, scaled_dx)
        return x2, {
            "routed_frac": round(routed_frac, 4),
            "copy_shield_frac": round(shield_frac, 4),
            "anticirc_loss": anticirc,
            "hard_mask": hard_mask,
        }


class X9LM(nn.Module):
    """Full Track A `X9_acex9_v2_muon` Language Model."""

    def __init__(self, c: X9Config):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d)
        self.blocks = nn.ModuleList(X9Block(c, k) for k in c.layout)
        self.nf = RMSNorm(c.d)
        self.head = None if c.tie else nn.Linear(c.d, c.vocab, bias=False)
        self.refiner = ACEX9Refiner(c)
        self.first_attn_idx = min(i for i, k in enumerate(c.layout) if k == "A")
        self.last_attn_idx = max(i for i, k in enumerate(c.layout) if k == "A")
        self._init_weights()

    def _init_weights(self) -> None:
        n_eff = len(self.c.layout)
        for name, p in self.named_parameters():
            if any(t in name for t in ("emb", "v_gate", "refiner")):
                continue
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02)
                if name.endswith(("o.weight", "dn.weight")):
                    p.data.mul_(1.0 / math.sqrt(2 * n_eff))
        nn.init.normal_(self.emb.weight, std=0.02)

    def _logits(self, h: torch.Tensor) -> torch.Tensor:
        w_out = self.emb.weight if self.head is None else self.head.weight
        return (self.nf(h) @ w_out.T).float()

    @staticmethod
    def _top2_margin(logits: torch.Tensor) -> torch.Tensor:
        top2 = torch.topk(logits, k=2, dim=-1).values
        lse = torch.logsumexp(logits, dim=-1, keepdim=True)
        p2 = (top2 - lse).exp()
        return p2[..., 0] - p2[..., 1]

    def forward(self, idx: torch.Tensor, tgt: torch.Tensor | None = None, return_aux: bool = False):
        B, T = idx.shape
        x0 = self.emb(idx)
        x = x0
        hd = self.c.d // self.c.n_head
        rope = rope_cache(T, hd, self.c.rope_theta, idx.device)
        v0 = None
        h_mid = None
        k1_anc = v1_anc = None

        for i, blk in enumerate(self.blocks):
            x, v0, kv = blk(x, rope, v0, ret_kv=(i == self.last_attn_idx))
            if i == self.first_attn_idx:
                h_mid = x
            if kv is not None:
                k1_anc, v1_anc = kv

        # Memory-efficient causal Top-8 margin & epistemic JSD computation (no full-vocab softmax tensor!)
        with torch.no_grad():
            lg1 = self._logits(x)
            top8_lg, top8_idx = torch.topk(lg1, k=8, dim=-1)
            lse1 = torch.logsumexp(lg1, dim=-1, keepdim=True)
            top8_p1 = (top8_lg - lse1).exp()
            u_margin = (1.0 - (top8_p1[..., 0] - top8_p1[..., 1])).clamp(0.0, 1.0)
            top8_mass = top8_p1.sum(dim=-1).clamp(0.0, 1.0)

            if not self.training and h_mid is not None:
                lg_m = self._logits(h_mid)
                lse_m = torch.logsumexp(lg_m, dim=-1, keepdim=True)
                mid_on_8 = (lg_m.gather(-1, top8_idx) - lse_m).exp()
                p_d = torch.cat([top8_p1, (1.0 - top8_mass).unsqueeze(-1).clamp_min(1e-7)], dim=-1)
                q_d = torch.cat([mid_on_8.clamp_min(1e-7), (1.0 - mid_on_8.sum(-1)).clamp(0.0, 1.0).unsqueeze(-1).clamp_min(1e-7)], dim=-1)
                p_d = p_d / p_d.sum(-1, keepdim=True)
                q_d = q_d / q_d.sum(-1, keepdim=True)
                m_d = 0.5 * (p_d + q_d)
                jsd = 0.5 * ((p_d * (p_d.clamp_min(1e-7).log() - m_d.clamp_min(1e-7).log())).sum(-1) +
                             (q_d * (q_d.clamp_min(1e-7).log() - m_d.clamp_min(1e-7).log())).sum(-1)) / math.log(2.0)
                jsd_ep = jsd.clamp(0.0, 1.0)
                del lg_m
            else:
                jsd_ep = torch.zeros_like(u_margin)
            if self.training:
                del lg1

        delta_h1 = (x - x0).detach()
        assert k1_anc is not None and v1_anc is not None
        x_cand, ref_aux = self.refiner(x, delta_h1, k1_anc, v1_anc, rope, u_margin, top8_mass, jsd_ep)

        # Inference/Eval Monotonic Margin Non-Degradation Guard:
        # Commit Pass-2 hidden state ONLY at positions where margin separation does not degrade!
        if not self.training and self.c.routing_mode == "acex9" and ref_aux["routed_frac"] > 0.0:
            with torch.no_grad():
                lg_cand = self._logits(x_cand)
                m1 = top8_p1[..., 0] - top8_p1[..., 1]
                m2 = self._top2_margin(lg_cand)
                keep = (m2 >= m1) & ref_aux["hard_mask"]
                lg = torch.where(keep.unsqueeze(-1), lg_cand, lg1)
                ref_aux["routed_frac"] = round(float(keep.float().mean().item()), 4)
        else:
            lg = self._logits(x_cand)

        ref_aux.pop("hard_mask", None)
        if tgt is None:
            eval_aux = {k: (round(float(v.detach().item()), 5) if isinstance(v, torch.Tensor) else v) for k, v in ref_aux.items()}
            return (lg, eval_aux) if return_aux else lg
        ce = F.cross_entropy(lg.view(-1, lg.size(-1)), tgt.reshape(-1), reduction="mean")
        loss = ce + self.c.anticirc_weight * ref_aux["anticirc_loss"]
        ref_aux_clean = {k: (round(float(v.detach().item()), 5) if isinstance(v, torch.Tensor) else v) for k, v in ref_aux.items()}
        return (loss, ref_aux_clean) if return_aux else loss

    def param_report(self) -> dict[str, int]:
        tot = sum(p.numel() for p in self.parameters())
        emb = self.emb.weight.numel() + (self.head.weight.numel() if self.head is not None else 0)
        ref_p = sum(p.numel() for p in self.refiner.parameters())
        base_non_emb = tot - emb - ref_p
        active_non_emb = int(base_non_emb + self.c.atrd_frac * ref_p)
        return {
            "total": tot,
            "embedding": emb,
            "non_embedding": tot - emb,
            "active_non_embedding": active_non_emb,
            "adaptive_refiner_params": ref_p,
            "effective_depth": len(self.c.layout) + 1,
        }
