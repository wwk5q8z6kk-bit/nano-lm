"""Inference definitions from lab/arch/gen5_foundations.py at 10e14108. See PROVENANCE.json."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import torch
from torch import nn
import torch.nn.functional as F
from .common import RMSNorm, ShortConv, apply_rope, rope_cache
from .gen4 import IdentityConv3, apply_data_dependent_rotor
from .kernels import nilpotent_wy_inverse


@dataclass
class Gen5Cfg:
    vocab: int = 32768
    d: int = 256
    n_head: int = 4
    n_kv: int = 2
    ffn: int = 704
    layout: str = "DDDADDDA"
    seq: int = 512
    rope_theta: float = 10000.0
    conv: int = 4
    chunk: int = 64
    tie: bool = True
    # Validated Gen-3 + Gen-4.5 foundation defaults:
    gated_vres: bool = True
    sharp_v: bool = True
    qk_canon_attn: bool = True
    householder_beta: bool = True
    diff_anchor: bool = True
    depth_attnres: bool = True
    canon_ffn: bool = True
    thermo_diff_attn: bool = True
    rational_ffn: bool = True
    # Gen-5 candidate mechanisms (each independently falsifiable):
    coboundary_gauge_delta: bool = False  # M9: O(1)-variance exact-potential gauge rotor
    jaynes_gibbs_attn: bool = False       # M10: Content + log-pos adaptive Gibbs inverse temperature
    dual_logit_bridge: bool = False       # M11: Low-rank covariant-contravariant unembedding bridge
    bridge_rank: int = 64


class Gen5GibbsDiffAttention(nn.Module):
    """Global Differential Attention Anchor with `M6` Log-Partition + `M10` Jaynes Content Temperature."""

    def __init__(self, c: Gen5Cfg, layer_idx: int = 0) -> None:
        super().__init__()
        self.h = c.n_head
        self.kv = c.n_kv
        self.hd = c.d // c.n_head
        self.q = nn.Linear(c.d, self.h * self.hd, bias=False)
        self.k = nn.Linear(c.d, self.kv * self.hd, bias=False)
        self.v = nn.Linear(c.d, self.kv * self.hd, bias=False)
        self.o = nn.Linear(c.d, self.h * self.hd, bias=False)
        self.qn = RMSNorm(self.hd)
        self.kn = RMSNorm(self.hd)
        self.qk_conv = ShortConv((self.h + self.kv) * self.hd, c.conv) if (c.qk_canon_attn and c.conv > 1) else None

        self.gated_vres = c.gated_vres
        if self.gated_vres:
            self.v_gate = nn.Linear(c.d, self.h, bias=True)
            nn.init.zeros_(self.v_gate.weight)
            nn.init.constant_(self.v_gate.bias, 1.0)

        self.diff_anchor = c.diff_anchor
        if self.diff_anchor:
            half_hd = self.hd // 2
            self.lam_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
            self.lam_q1 = nn.Parameter(torch.randn(self.h, half_hd) * 0.1)
            self.lam_k1 = nn.Parameter(torch.randn(self.h, half_hd) * 0.1)
            self.lam_q2 = nn.Parameter(torch.randn(self.h, half_hd) * 0.1)
            self.lam_k2 = nn.Parameter(torch.randn(self.h, half_hd) * 0.1)
            self.sub_norm = RMSNorm(self.hd)

        self.thermo_diff_attn = c.thermo_diff_attn
        if self.thermo_diff_attn:
            self.thermo_gamma = nn.Parameter(torch.full((1, self.h, 1, 1), 0.12))

        self.jaynes_gibbs_attn = c.jaynes_gibbs_attn
        if self.jaynes_gibbs_attn:
            self.gibbs_proj = nn.Linear(c.d, self.h, bias=True)
            nn.init.zeros_(self.gibbs_proj.weight)
            nn.init.zeros_(self.gibbs_proj.bias)

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor], v0: torch.Tensor | None):
        B, T, _ = x.shape
        q_raw = self.q(x)
        k_raw = self.k(x)
        if self.qk_conv is not None:
            qk = self.qk_conv(torch.cat([q_raw, k_raw], dim=-1))
            q_raw, k_raw = qk.split([self.h * self.hd, self.kv * self.hd], dim=-1)

        q = apply_rope(self.qn(q_raw.view(B, T, self.h, self.hd)).transpose(1, 2), *rope)
        k = apply_rope(self.kn(k_raw.view(B, T, self.kv, self.hd)).transpose(1, 2), *rope)
        v = self.v(x).view(B, T, self.kv, self.hd).transpose(1, 2)

        if self.thermo_diff_attn or self.jaynes_gibbs_attn:
            scale_t = torch.ones((B, self.h, T, 1), device=x.device, dtype=torch.float32)
            if self.thermo_diff_attn:
                pos_log = torch.log1p(torch.arange(1, T + 1, device=x.device, dtype=torch.float32)).view(1, 1, T, 1)
                scale_t = scale_t + self.thermo_gamma.float() * pos_log
            if self.jaynes_gibbs_attn:
                # Bounded [-0.5, +0.5] dynamic inverse-temperature modulation (exact 0 at init!)
                gibbs_delta = 0.5 * torch.tanh(self.gibbs_proj(x).float()).transpose(1, 2).unsqueeze(-1)
                scale_t = scale_t + gibbs_delta
            q = q * scale_t.to(q.dtype)

        rep = self.h // self.kv
        k_exp = k.repeat_interleave(rep, dim=1)
        v_exp = v.repeat_interleave(rep, dim=1)

        if self.gated_vres:
            if v0 is None:
                v0 = v_exp
            else:
                alpha = torch.sigmoid(self.v_gate(x)).transpose(1, 2).unsqueeze(-1)
                v_exp = alpha * v_exp + (1.0 - alpha) * v0.to(v_exp.dtype)

        if not self.diff_anchor:
            y = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
        else:
            half = self.hd // 2
            q1, q2 = q.split(half, dim=-1)
            k1, k2 = k_exp.split(half, dim=-1)
            va, vb = v_exp.split(half, dim=-1)
            v_cat = torch.cat([va, vb], dim=1)
            y1_cat = F.scaled_dot_product_attention(torch.cat([q1, q1], dim=1), torch.cat([k1, k1], dim=1), v_cat, is_causal=True)
            y2_cat = F.scaled_dot_product_attention(torch.cat([q2, q2], dim=1), torch.cat([k2, k2], dim=1), v_cat, is_causal=True)
            y1 = torch.cat(y1_cat.chunk(2, dim=1), dim=-1)
            y2 = torch.cat(y2_cat.chunk(2, dim=1), dim=-1)
            lam1 = torch.exp((self.lam_q1.float() * self.lam_k1.float()).sum(dim=-1))
            lam2 = torch.exp((self.lam_q2.float() * self.lam_k2.float()).sum(dim=-1))
            lam = (lam1 - lam2 + self.lam_init).view(1, self.h, 1, 1).to(y1.dtype)
            y = self.sub_norm(y1 - lam * y2) * (1.0 - self.lam_init)

        return self.o(y.transpose(1, 2).reshape(B, T, -1)), v0


class Gen5CoboundaryHouseholderDelta(nn.Module):
    """`M9`: Exact-Potential Coboundary Gauge-Rotor + Householder Delta (`O(1)` variance for all `T`)."""

    def __init__(self, c: Gen5Cfg) -> None:
        super().__init__()
        self.h = c.n_head
        self.hd = c.d // c.n_head
        d = c.d
        self.sharp_v = c.sharp_v
        self.householder_beta = c.householder_beta
        self.coboundary_gauge = c.coboundary_gauge_delta
        self.C = c.chunk

        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.conv = ShortConv(2 * d if c.sharp_v else 3 * d, c.conv) if c.conv > 1 else None
        self.a = nn.Linear(d, self.h, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.linspace(1, 16, self.h)))
        self.b = nn.Linear(d, self.h, bias=True)
        if self.householder_beta:
            nn.init.constant_(self.b.bias, -math.log(3.0))

        if self.coboundary_gauge:
            # Exact 0-form potential Omega_t = (pi/2) * tanh(W_pot x_t): Var(Omega_t) <= (pi/2)^2 for ALL T!
            self.pot_proj = nn.Linear(d, self.h * (self.hd // 2), bias=True)
            nn.init.zeros_(self.pot_proj.weight)
            nn.init.zeros_(self.pot_proj.bias)

        self.g = nn.Linear(d, d, bias=False)
        self.on = RMSNorm(self.hd)
        self.o = nn.Linear(d, d, bias=False)
        self.gated_vres = c.gated_vres
        if self.gated_vres:
            self.v_gate = nn.Linear(d, self.h, bias=True)
            nn.init.zeros_(self.v_gate.weight)
            nn.init.constant_(self.v_gate.bias, 1.0)

    def forward_recurrent_reference(self, x: torch.Tensor, v0: torch.Tensor | None = None):
        """Exact `O(1)`-state physical-frame recurrent reference verifying `S_t = U(Omega_t - Omega_{t-1}) ...`."""
        B, T, D = x.shape
        H, hd = self.h, self.hd
        qkv = self.qkv(x)
        if self.conv is None:
            q, k, v = qkv.split(D, dim=-1)
        elif self.sharp_v:
            qk, v = qkv.split([2 * D, D], dim=-1)
            q, k = self.conv(qk).split(D, dim=-1)
        else:
            q, k, v = self.conv(qkv).split(D, dim=-1)

        q = F.normalize(q.view(B, T, H, hd).float(), dim=-1).transpose(1, 2) * (hd ** -0.5)
        k = F.normalize(k.view(B, T, H, hd).float(), dim=-1).transpose(1, 2)
        v = v.view(B, T, H, hd).float().transpose(1, 2)

        if self.gated_vres:
            if v0 is None:
                v0 = v
            else:
                alpha = torch.sigmoid(self.v_gate(x).float()).transpose(1, 2).unsqueeze(-1)
                v = alpha * v + (1.0 - alpha) * v0.float()

        loga = (-F.softplus(self.a(x).float()) * self.A_log.exp()).transpose(1, 2)
        beta_scale = 2.0 if self.householder_beta else 1.0
        beta = (beta_scale * torch.sigmoid(self.b(x).float())).clamp(max=1.995).transpose(1, 2)

        pot = (
            (0.5 * math.pi) * torch.tanh(self.pot_proj(x).float()).view(B, T, H, hd // 2).transpose(1, 2)
            if self.coboundary_gauge
            else x.new_zeros(B, H, T, hd // 2, dtype=torch.float32)
        )

        # Physical state S_phys [B, H, hd, hd] + previous gauge potential prev_pot [B, H, 1, hd//2]
        S_phys = x.new_zeros(B, H, hd, hd, dtype=torch.float32)
        prev_pot = x.new_zeros(B, H, 1, hd // 2, dtype=torch.float32)
        ys = []
        for t in range(T):
            kt = k[:, :, t : t + 1, :]  # [B, H, 1, hd]
            qt = q[:, :, t : t + 1, :]
            vt = v[:, :, t : t + 1, :]
            at = loga[:, :, t].exp().view(B, H, 1, 1)
            bt = beta[:, :, t].view(B, H, 1, 1)
            cur_pot = pot[:, :, t : t + 1, :]
            d_pot = cur_pot - prev_pot  # Coboundary step rotation!

            # 1) Householder erase/write on S_phys: S_mid = a_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
            proj = kt @ S_phys  # [B, H, 1, hd]
            u = bt * (vt - at * proj)
            S_mid = at * S_phys + kt.transpose(-1, -2) @ u
            # 2) Rotate rows (key space) of S_mid by U(Omega_t - Omega_{t-1})!
            if self.coboundary_gauge:
                # Apply 2x2 rotation along the key axis (-2) by transposing (-1, -2), rotating, transposing back
                S_T = S_mid.transpose(-1, -2)  # [B, H, hd, hd] (last dim = key dim!)
                S_phys = apply_data_dependent_rotor(S_T, d_pot).transpose(-1, -2)
            else:
                S_phys = S_mid
            yt = qt @ S_phys  # [B, H, 1, hd]
            ys.append(yt)
            prev_pot = cur_pot

        y = torch.cat(ys, dim=2)
        y = self.on(y.transpose(1, 2).to(x.dtype)).reshape(B, T, D) * F.silu(self.g(x))
        return self.o(y), v0

    def forward(self, x: torch.Tensor, v0: torch.Tensor | None):
        B, T, D = x.shape
        H, hd, C = self.h, self.hd, self.C
        qkv = self.qkv(x)
        if self.conv is None:
            q, k, v = qkv.split(D, dim=-1)
        elif self.sharp_v:
            qk, v = qkv.split([2 * D, D], dim=-1)
            q, k = self.conv(qk).split(D, dim=-1)
        else:
            q, k, v = self.conv(qkv).split(D, dim=-1)

        q = F.normalize(q.view(B, T, H, hd).float(), dim=-1).transpose(1, 2) * (hd ** -0.5)
        k = F.normalize(k.view(B, T, H, hd).float(), dim=-1).transpose(1, 2)
        v = v.view(B, T, H, hd).float().transpose(1, 2)

        if self.coboundary_gauge:
            # Exact bounded gauge potential Omega_t in (-pi/2, pi/2): ZERO cumulative random-walk drift!
            omega_pot = (0.5 * math.pi) * torch.tanh(self.pot_proj(x).float()).view(B, T, H, hd // 2).transpose(1, 2)
            prev_pot = F.pad(omega_pot[:, :, :-1, :], (0, 0, 1, 0))
            k = apply_data_dependent_rotor(k, -prev_pot)
            q = apply_data_dependent_rotor(q, -omega_pot)

        if self.gated_vres:
            if v0 is None:
                v0 = v
            else:
                alpha = torch.sigmoid(self.v_gate(x).float()).transpose(1, 2).unsqueeze(-1)
                v = alpha * v + (1.0 - alpha) * v0.float()

        loga = (-F.softplus(self.a(x).float()) * self.A_log.exp()).transpose(1, 2)
        beta_scale = 2.0 if self.householder_beta else 1.0
        beta = (beta_scale * torch.sigmoid(self.b(x).float())).clamp(max=1.995).transpose(1, 2)

        if os.environ.get("NANOLM_FLA", "1") == "1" and x.is_cuda:
            try:
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule
                tr = lambda t: t.transpose(1, 2).contiguous().to(torch.bfloat16)
                y_fla, _ = chunk_gated_delta_rule(
                    tr(q), tr(k), tr(v),
                    loga.transpose(1, 2).contiguous(),
                    beta.transpose(1, 2).contiguous().to(torch.bfloat16),
                    scale=1.0,
                )
                y = self.on(y_fla.to(x.dtype)).reshape(B, T, D) * F.silu(self.g(x))
                return self.o(y), v0
            except ImportError:
                pass

        pad = (-T) % C
        if pad:
            q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
            loga = F.pad(loga, (0, pad))
            beta = F.pad(beta, (0, pad))
        N = (T + pad) // C
        qc, kc, vc = (t.view(B, H, N, C, hd) for t in (q, k, v))
        la = loga.view(B, H, N, C).cumsum(-1)
        tri = torch.tril(torch.ones(C, C, dtype=torch.bool, device=x.device))
        strict = torch.tril(torch.ones(C, C, dtype=torch.bool, device=x.device), -1)
        decay = (la[..., :, None] - la[..., None, :]).masked_fill(~tri, -1e30).exp()
        bc = beta.view(B, H, N, C)
        kb = kc * bc[..., None]
        Akk = (kb @ kc.transpose(-1, -2) * decay).masked_fill(~strict, 0.0)
        if x.device.type == "mps" and (C & (C - 1)) == 0:
            Tm = nilpotent_wy_inverse(Akk)
        else:
            eye = torch.eye(C, device=x.device, dtype=torch.float32)
            Tm = torch.linalg.solve_triangular(eye + Akk, eye.expand_as(Akk), upper=False)
        la_exp = la.exp()
        w_ = Tm @ (kb * la_exp[..., None])
        u_ = Tm @ (vc * bc[..., None])

        attn_all = (qc @ kc.transpose(-1, -2)) * decay
        q_scaled_all = qc * la_exp[..., None]
        tail_all = (la[..., -1:] - la).exp()[..., None]
        kt_all = (kc * tail_all).transpose(-1, -2)
        g_all = la[..., -1].exp()[..., None, None]

        S = x.new_zeros(B, H, hd, hd, dtype=torch.float32)
        u_list, s_list = [], []
        for n in range(N):
            s_list.append(S)
            u = u_[:, :, n] - w_[:, :, n] @ S
            u_list.append(u)
            S = g_all[:, :, n] * S + kt_all[:, :, n] @ u

        u_stack = torch.stack(u_list, 2)
        s_stack = torch.stack(s_list, 2)
        y = (attn_all @ u_stack + q_scaled_all @ s_stack).view(B, H, N * C, hd)[:, :, :T]
        y = self.on(y.transpose(1, 2).to(x.dtype)).reshape(B, T, D) * F.silu(self.g(x))
        return self.o(y), v0


class Gen5Block(nn.Module):
    def __init__(self, c: Gen5Cfg, kind: str, layer_idx: int, num_depth_anchors: int = 0) -> None:
        super().__init__()
        self.kind = kind
        self.n1 = RMSNorm(c.d)
        self.n2 = RMSNorm(c.d)
        self.mix = Gen5GibbsDiffAttention(c, layer_idx) if kind == "A" else Gen5CoboundaryHouseholderDelta(c)
        self.ffn_conv = IdentityConv3(c.d) if c.canon_ffn else None
        self.gu = nn.Linear(c.d, 2 * c.ffn, bias=False)
        self.dn = nn.Linear(c.ffn, c.d, bias=False)
        self.rational_alpha = nn.Parameter(torch.zeros(1)) if c.rational_ffn else None

        self.num_depth_anchors = num_depth_anchors
        if num_depth_anchors > 0:
            self.depth_router = nn.Linear(c.d, num_depth_anchors, bias=False)
            self.depth_scale = nn.Parameter(torch.zeros(1))
            nn.init.zeros_(self.depth_router.weight)

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor], v0: torch.Tensor | None, depth_bank: list[torch.Tensor]):
        if self.num_depth_anchors > 0 and depth_bank:
            k_av = min(len(depth_bank), self.num_depth_anchors)
            w = F.softmax(self.depth_router(self.n1(x))[..., :k_av].float(), dim=-1).to(x.dtype)
            ret = (w.unsqueeze(-1) * torch.stack(depth_bank[:k_av], dim=-2)).sum(dim=-2)
            x = x + torch.tanh(self.depth_scale).to(x.dtype) * (ret - x)

        if self.kind == "A":
            h, v0 = self.mix(self.n1(x), rope, v0)
        else:
            h, v0 = self.mix(self.n1(x), v0)
        x = x + h

        h2 = self.n2(x)
        if self.ffn_conv is not None:
            h2 = self.ffn_conv(h2)
        g, u = self.gu(h2).chunk(2, dim=-1)
        act = F.silu(g)
        if self.rational_alpha is not None:
            gf = g.float()
            act = (act.float() + self.rational_alpha.float() * (gf / (1.0 + gf * gf))).to(x.dtype)
        return x + self.dn(act * u), v0


class Gen5LM(nn.Module):
    def __init__(self, c: Gen5Cfg) -> None:
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d)
        anchors, seen = [], 1
        for i, k in enumerate(c.layout):
            anchors.append(seen if (c.depth_attnres and i >= 2) else 0)
            if k == "A":
                seen += 1
        self.blocks = nn.ModuleList(Gen5Block(c, k, i, anchors[i]) for i, k in enumerate(c.layout))
        self.nf = RMSNorm(c.d)
        if c.dual_logit_bridge:
            # Zero-initialized rank-r covariant-to-contravariant metric adapter before tied unembedding!
            self.bridge_down = nn.Linear(c.d, c.bridge_rank, bias=False)
            self.bridge_up = nn.Linear(c.bridge_rank, c.d, bias=False)
        else:
            self.bridge_down = None
            self.bridge_up = None
        self.head = None if c.tie else nn.Linear(c.d, c.vocab, bias=False)
        self._init()

    def _init(self) -> None:
        n_eff = len(self.c.layout)
        skip = (
            "emb", "v_gate", "depth_router", "ffn_conv", "lam_",
            "pot_proj", "gibbs_proj", "thermo_gamma", "rational_alpha", "bridge_up",
        )
        for n, p in self.named_parameters():
            if p.dim() == 2 and not any(t in n for t in skip):
                nn.init.normal_(p, std=0.02)
                if n.endswith(("o.weight", "dn.weight")):
                    p.data.mul_(1.0 / math.sqrt(2 * n_eff))
        nn.init.normal_(self.emb.weight, std=0.02)
        if self.bridge_up is not None:
            nn.init.zeros_(self.bridge_up.weight)

    def param_report(self) -> dict[str, int]:
        tot = sum(p.numel() for p in self.parameters())
        emb = self.emb.weight.numel() + (0 if self.head is None else self.head.weight.numel())
        return {"total": tot, "non_embedding": tot - emb, "embedding": emb, "effective_layers": len(self.c.layout)}

    def forward(self, idx: torch.Tensor, tgt: torch.Tensor | None = None):
        B, T = idx.shape
        x = self.emb(idx)
        rope = rope_cache(T, self.c.d // self.c.n_head, self.c.rope_theta, idx.device)
        v0 = None
        bank = [x] if self.c.depth_attnres else []
        for b in self.blocks:
            x, v0 = b(x, rope, v0, bank)
            if self.c.depth_attnres and b.kind == "A":
                bank.append(x)
        h = self.nf(x)
        if self.bridge_down is not None and self.bridge_up is not None:
            h = h + self.bridge_up(F.silu(self.bridge_down(h)))
        lg = (h @ self.emb.weight.T if self.head is None else self.head(h)).float()
        if tgt is None:
            return lg
        return F.cross_entropy(lg.view(-1, lg.size(-1)), tgt.reshape(-1), ignore_index=-100)
