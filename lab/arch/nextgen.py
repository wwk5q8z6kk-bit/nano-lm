"""Next-generation first-principles architecture candidates (Track A — LAB-ARCH-03).

Derived directly from Wave-1 empirical findings in LAB-ARCH-02:
  Finding 1: 3:1 Gated Delta-Rule + Global Attention (H2_delta) cut composite BPB by -3.04%,
             cut global KV cache by -80%, and raised 512-tok OOD induction accuracy from
             36.0% (T0_dense) to 90.9%.
  Finding 2: Unconditioned scalar Layer-0 Value Residual (T1_vres) cut composite BPB by -4.08%
             (-6.44% on code!) faster than T0_dense, BUT collapsed OOD random-token induction
             accuracy to 8.3% because (a) a single scalar per layer contaminates all KV heads
             uniformly with unshifted static embeddings, and (b) in H3_delta_vres, v0 was only
             wired between Attention layers rather than universally across Delta + Attention.

This module implements three falsifiable mathematical mechanisms:
  1. Universal Per-Head Input-Gated Value Highway (Gated-VRes):
       v^{(l)}_{b,h,t} = alpha_{b,h,t} * v^{(l)}_{b,h,t} + (1 - alpha_{b,h,t}) * v^{(0)}_{b,h,t}
       alpha_{b,h,t}   = sigmoid(w_{l,h}^T x_{b,t} + b_{l,h})
     Shared across BOTH Gated-Delta (H heads) and GQA Attention (projected per head), allowing
     semantic heads to pull clean Layer-0 token identity while induction heads dynamically gate
     contextual vs anchor values per token.
  2. QK-Decoupled Causal Conv + Sharp Value Write (Sharp-Delta):
     Applies ShortConv(k=4) ONLY to (Q, K) so keys capture local lag/n-gram shifts (enabling
     1-layer induction matching) while V remains unblurred for crisp associative recall S_t^T q_t.
  3. Surprise-Gated Recurrent Depth (SGRD):
     Uses the Gated Delta-Rule's own online associative prediction error ||e_t|| (in-context
     information surprise) alongside hidden state h_t to drive per-token halting probabilities
     across recurrent core passes, trained with PonderNet expected output + KL geometric prior.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from lab.arch.model import MLP, RMSNorm, ShortConv, apply_rope, rope_cache

USE_FLA = os.environ.get("NANOLM_FLA", "0") == "1"


@dataclass
class NextGenCfg:
    vocab: int = 32768
    d: int = 512
    n_head: int = 8
    n_kv: int = 4
    ffn: int = 1408
    layout: str = "DDDADDDADA"
    seq: int = 1024
    rope_theta: float = 10000.0
    conv: int = 4
    chunk: int = 64
    tie: bool = True
    # Mechanism toggles (all independently ablatable):
    gated_vres: bool = False          # X1: universal per-head input-gated Layer-0 V highway
    sharp_v: bool = False             # X2: ShortConv on Q,K only; unblurred V write
    qk_canon_attn: bool = False       # X2b: light ShortConv on Q,K in Attention layers too
    loop: tuple[int, int, int, int] | None = None  # (prelude, core, coda, max_r)
    sgrd_prior_p: float = 0.5         # X3: geometric prior halting prob (0 = fixed depth)
    sgrd_kl_weight: float = 0.01


class NextGenAttention(nn.Module):
    def __init__(self, c: NextGenCfg):
        super().__init__()
        self.h, self.kv, self.hd = c.n_head, c.n_kv, c.d // c.n_head
        self.q = nn.Linear(c.d, self.h * self.hd, bias=False)
        self.k = nn.Linear(c.d, self.kv * self.hd, bias=False)
        self.v = nn.Linear(c.d, self.kv * self.hd, bias=False)
        self.o = nn.Linear(c.d, self.h * self.hd, bias=False)
        self.qn, self.kn = RMSNorm(self.hd), RMSNorm(self.hd)
        self.qk_conv = ShortConv((self.h + self.kv) * self.hd, c.conv) if c.qk_canon_attn else None
        self.gated_vres = c.gated_vres
        if self.gated_vres:
            # Per-query-head input-dependent gate after KV repeat so all H heads match v0 (B,H,T,hd)
            self.v_gate = nn.Linear(c.d, self.h, bias=True)
            nn.init.zeros_(self.v_gate.weight)
            nn.init.constant_(self.v_gate.bias, 1.0)  # start at ~0.73 current V, ~0.27 v0

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor], v0: torch.Tensor | None):
        B, T, _ = x.shape
        q_raw = self.q(x)
        k_raw = self.k(x)
        if self.qk_conv is not None:
            qk = self.qk_conv(torch.cat([q_raw, k_raw], dim=-1))
            q_raw, k_raw = qk.split([self.h * self.hd, self.kv * self.hd], dim=-1)
        q = self.qn(q_raw.view(B, T, self.h, self.hd)).transpose(1, 2)
        k = self.kn(k_raw.view(B, T, self.kv, self.hd)).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        rep = self.h // self.kv
        k_exp = k.repeat_interleave(rep, 1)
        v_exp = v.repeat_interleave(rep, 1)  # B, H, T, hd
        if self.gated_vres:
            if v0 is None:
                v0 = v_exp
            else:
                alpha = torch.sigmoid(self.v_gate(x)).transpose(1, 2).unsqueeze(-1)  # B, H, T, 1
                v_exp = alpha * v_exp + (1.0 - alpha) * v0.to(v_exp.dtype)
        y = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, -1)), v0, None


class NextGenDeltaMixer(nn.Module):
    """Gated Delta-Rule with optional Sharp-V (QK-only conv), universal Gated-VRes, and surprise norm output."""

    def __init__(self, c: NextGenCfg):
        super().__init__()
        self.h = c.n_head
        self.hd = c.d // c.n_head
        d = c.d
        self.sharp_v = c.sharp_v
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.conv = ShortConv(2 * d if c.sharp_v else 3 * d, c.conv)
        self.a = nn.Linear(d, self.h, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.linspace(1, 16, self.h)))
        self.b = nn.Linear(d, self.h, bias=True)
        self.g = nn.Linear(d, d, bias=False)
        self.on = RMSNorm(self.hd)
        self.o = nn.Linear(d, d, bias=False)
        self.C = c.chunk
        self.gated_vres = c.gated_vres
        if self.gated_vres:
            self.v_gate = nn.Linear(d, self.h, bias=True)
            nn.init.zeros_(self.v_gate.weight)
            nn.init.constant_(self.v_gate.bias, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        v0: torch.Tensor | None,
        need_surprise: bool = False,
    ):
        B, T, D = x.shape
        H, hd, C = self.h, self.hd, self.C
        qkv = self.qkv(x)
        if self.sharp_v:
            qk, v_raw = qkv.split([2 * D, D], dim=-1)
            q, k = self.conv(qk).split(D, dim=-1)
            v = v_raw  # sharp, unblurred token value write!
        else:
            q, k, v = self.conv(qkv).split(D, dim=-1)

        q = F.normalize(q.view(B, T, H, hd).float(), dim=-1).transpose(1, 2) * (hd ** -0.5)
        k = F.normalize(k.view(B, T, H, hd).float(), dim=-1).transpose(1, 2)
        v = v.view(B, T, H, hd).float().transpose(1, 2)  # B, H, T, hd

        if self.gated_vres:
            if v0 is None:
                v0 = v
            else:
                alpha = torch.sigmoid(self.v_gate(x).float()).transpose(1, 2).unsqueeze(-1)
                v = alpha * v + (1.0 - alpha) * v0.float()

        loga = (-F.softplus(self.a(x).float()) * self.A_log.exp()).transpose(1, 2)  # B, H, T
        beta = torch.sigmoid(self.b(x).float()).transpose(1, 2)  # B, H, T

        if USE_FLA and x.is_cuda and not need_surprise:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            tr = lambda t: t.transpose(1, 2).contiguous().to(torch.bfloat16)
            y_fla, _ = chunk_gated_delta_rule(
                tr(q), tr(k), tr(v),
                loga.transpose(1, 2).contiguous(),
                beta.transpose(1, 2).contiguous().to(torch.bfloat16),
                scale=1.0,
            )
            y = self.on(y_fla.to(x.dtype)).reshape(B, T, D) * F.silu(self.g(x))
            return self.o(y), v0, None

        pad = (-T) % C
        if pad:
            q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
            loga = F.pad(loga, (0, pad))
            beta = F.pad(beta, (0, pad))
        N = (T + pad) // C
        q, k, v = (t.view(B, H, N, C, hd) for t in (q, k, v))
        la = loga.view(B, H, N, C).cumsum(-1)
        tri = torch.tril(torch.ones(C, C, dtype=torch.bool, device=x.device))
        decay = (la[..., :, None] - la[..., None, :]).masked_fill(~tri, -1e30).exp()
        beta_c = beta.view(B, H, N, C)
        kb = k * beta_c[..., None]
        strict = torch.tril(torch.ones(C, C, dtype=torch.bool, device=x.device), -1)
        Akk = (kb @ k.transpose(-1, -2) * decay).masked_fill(~strict, 0)
        eye = torch.eye(C, device=x.device)
        Tm = torch.linalg.solve_triangular(eye + Akk, eye.expand_as(Akk), upper=False)
        w_ = Tm @ (kb * la.exp()[..., None])
        u_ = Tm @ (v * beta_c[..., None])

        S = x.new_zeros(B, H, hd, hd, dtype=torch.float32)
        ys = []
        surprises = [] if need_surprise else None
        for n in range(N):
            qn, kn, la_n, dn = q[:, :, n], k[:, :, n], la[:, :, n], decay[:, :, n]
            u = u_[:, :, n] - w_[:, :, n] @ S  # beta * prediction_error
            if need_surprise:
                # Online associative prediction error norm ||e_t|| averaged over heads -> (B, C)
                err = u / (beta_c[:, :, n].unsqueeze(-1) + 1e-4)
                surprises.append(err.norm(dim=-1).mean(dim=1))
            attn = (qn @ kn.transpose(-1, -2)) * dn
            y = attn @ u + (qn * la_n.exp()[..., None]) @ S
            tail = (la_n[..., -1:] - la_n).exp()[..., None]
            S = la_n[..., -1].exp()[..., None, None] * S + (kn * tail).transpose(-1, -2) @ u
            ys.append(y)

        y = torch.stack(ys, 2).view(B, H, N * C, hd)[:, :, :T]
        surprise_t = torch.cat(surprises, dim=-1)[:, :T] if need_surprise else None  # B, T
        y = self.on(y.transpose(1, 2).to(x.dtype)).reshape(B, T, D) * F.silu(self.g(x))
        return self.o(y), v0, surprise_t


class NextGenBlock(nn.Module):
    def __init__(self, c: NextGenCfg, kind: str):
        super().__init__()
        self.kind = kind
        self.n1, self.n2 = RMSNorm(c.d), RMSNorm(c.d)
        if kind == "A":
            self.mix = NextGenAttention(c)
        elif kind == "D":
            self.mix = NextGenDeltaMixer(c)
        else:
            raise ValueError(f"Unsupported mixer kind: {kind}")
        # Reuse MLP compatible shape
        self.gu = nn.Linear(c.d, 2 * c.ffn, bias=False)
        self.dn = nn.Linear(c.ffn, c.d, bias=False)

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor], v0: torch.Tensor | None, need_surprise: bool = False):
        if self.kind == "D":
            h, v0, surp = self.mix(self.n1(x), rope, v0, need_surprise=need_surprise)
        else:
            h, v0, surp = self.mix(self.n1(x), rope, v0)
        x = x + h
        g, u = self.gu(self.n2(x)).chunk(2, dim=-1)
        return x + self.dn(F.silu(g) * u), v0, surp


class NextGenLM(nn.Module):
    """Unified Next-Gen Discovery Model supporting Gated-VRes, Sharp-Delta, and SGRD."""

    def __init__(self, c: NextGenCfg):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d)
        self.blocks = nn.ModuleList(NextGenBlock(c, k) for k in c.layout)
        self.nf = RMSNorm(c.d)
        self.head = None if c.tie else nn.Linear(c.d, c.vocab, bias=False)
        self.use_sgrd = bool(c.loop and c.sgrd_prior_p > 0.0)
        if c.loop:
            p, co, k, r = c.loop
            assert p + co + k == len(c.layout)
            self.inject = nn.Linear(2 * c.d, c.d, bias=False)
            if self.use_sgrd:
                self.halt_proj = nn.Linear(c.d, 1, bias=True)
                self.halt_gamma = nn.Parameter(torch.tensor(0.5))  # surprise coupling weight
                nn.init.zeros_(self.halt_proj.weight)
                nn.init.constant_(self.halt_proj.bias, -0.5)
        self._init()

    def _init(self):
        n_eff = self.effective_depth()
        for name, p in self.named_parameters():
            if p.dim() == 2 and "emb" not in name and "v_gate" not in name and "halt_proj" not in name:
                nn.init.normal_(p, std=0.02)
                if name.endswith(("o.weight", "dn.weight")):
                    p.data.mul_(1.0 / math.sqrt(2 * n_eff))
        nn.init.normal_(self.emb.weight, std=0.02)
        if self.c.loop:
            nn.init.zeros_(self.inject.weight)
            with torch.no_grad():
                self.inject.weight[:, : self.c.d].copy_(torch.eye(self.c.d))

    def effective_depth(self) -> int:
        if not self.c.loop:
            return len(self.c.layout)
        p, co, k, r = self.c.loop
        return p + co * r + k

    def forward(self, idx: torch.Tensor, tgt: torch.Tensor | None = None, return_aux: bool = False):
        B, T = idx.shape
        x = self.emb(idx)
        hd = self.c.d // self.c.n_head
        rope = rope_cache(T, hd, self.c.rope_theta, idx.device)
        v0 = None
        aux: dict[str, torch.Tensor | float] = {}

        if not self.c.loop:
            for b in self.blocks:
                x, v0, _ = b(x, rope, v0)
        else:
            p, co, k, r = self.c.loop
            for i in range(p):
                x, v0, _ = self.blocks[i](x, rope, v0)
            anchor = x
            if not self.use_sgrd:
                for _ in range(r):
                    x = self.inject(torch.cat([x, anchor], dim=-1))
                    for i in range(p, p + co):
                        x, v0, _ = self.blocks[i](x, rope, v0)
            else:
                # Surprise-Gated Recurrent Depth (SGRD): PonderNet-style halting driven by ||e_t||
                rem = x.new_ones(B, T, dtype=torch.float32)
                h_acc = torch.zeros_like(x, dtype=torch.float32)
                step_probs = []
                last_delta_idx = max((i for i in range(p, p + co) if self.c.layout[i] == "D"), default=-1)
                for step_i in range(r):
                    x = self.inject(torch.cat([x, anchor], dim=-1))
                    last_surp = None
                    for i in range(p, p + co):
                        need_s = (i == last_delta_idx)
                        x, v0, surp = self.blocks[i](x, rope, v0, need_surprise=need_s)
                        if need_s:
                            last_surp = surp
                    if step_i == r - 1:
                        p_step = rem
                        rem = torch.zeros_like(rem)
                    else:
                        s_term = (
                            self.halt_gamma * torch.log(last_surp.clamp_min(1e-4))
                            if last_surp is not None
                            else 0.0
                        )
                        lam_halt = torch.sigmoid(self.halt_proj(x).squeeze(-1).float() + s_term)
                        p_step = rem * lam_halt
                        rem = rem * (1.0 - lam_halt)
                    step_probs.append(p_step)
                    h_acc = h_acc + p_step.unsqueeze(-1) * x.float()

                probs = torch.stack(step_probs, dim=-1)  # B, T, r
                x = h_acc.to(x.dtype)
                # Geometric prior over r truncated steps
                gp = self.c.sgrd_prior_p
                prior = torch.tensor(
                    [(1.0 - gp) ** j * gp if j < r - 1 else (1.0 - gp) ** (r - 1) for j in range(r)],
                    device=idx.device,
                    dtype=torch.float32,
                )
                kl = (probs.clamp_min(1e-7) * (probs.clamp_min(1e-7).log() - prior.log())).sum(dim=-1).mean()
                expected_steps = (probs * torch.arange(1, r + 1, device=idx.device, dtype=torch.float32)).sum(-1).mean()
                aux["sgrd_kl"] = kl
                aux["expected_loops"] = expected_steps.detach()
                aux["halt_gamma"] = self.halt_gamma.detach()

            for i in range(p + co, p + co + k):
                x, v0, _ = self.blocks[i](x, rope, v0)

        h = self.nf(x)
        lg = (h @ self.emb.weight.T if self.head is None else self.head(h)).float()
        if tgt is None:
            return (lg, aux) if return_aux else lg
        ce = F.cross_entropy(lg.view(-1, lg.size(-1)), tgt.reshape(-1), reduction="mean")
        loss = ce + (self.c.sgrd_kl_weight * aux["sgrd_kl"] if "sgrd_kl" in aux else 0.0)
        return (loss, aux) if return_aux else loss

    def param_report(self) -> dict[str, int]:
        tot = sum(p.numel() for p in self.parameters())
        emb = self.emb.weight.numel() + (self.head.weight.numel() if self.head is not None else 0)
        return {
            "total": tot,
            "embedding": emb,
            "non_embedding": tot - emb,
            "effective_depth": self.effective_depth(),
        }
