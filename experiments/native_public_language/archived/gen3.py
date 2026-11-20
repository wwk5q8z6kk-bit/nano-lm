"""Inference definitions from lab/arch/gen3.py at 10e14108. See PROVENANCE.json."""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from .common import RMSNorm, ShortConv, apply_rope, rope_cache

USE_FLA = False


@dataclass
class Gen3Cfg:
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
    # Generation-2 validated baseline features:
    gated_vres: bool = True
    sharp_v: bool = True
    qk_canon_attn: bool = True
    # Generation-3 falsifiable innovations (each independently toggleable):
    householder_beta: bool = False   # M1: beta_t = 2 * sigmoid(b_t) in (0, 2)
    diff_anchor: bool = False        # M2: Differential Global Attention in 'A' layers
    depth_attnres: bool = False      # M3: Laminar macro-stage Depth-AttnRes highway
    canon_ffn: bool = False          # M4: Identity-init 3-tap causal depthwise conv before SwiGLU
    # Optional adaptive recurrent depth (SGRD-v2):
    loop: tuple[int, int, int, int] | None = None
    sgrd_prior_p: float = 0.0
    sgrd_kl_weight: float = 0.01


def _lambda_init(layer_idx: int) -> float:
    return 0.8 - 0.6 * math.exp(-0.3 * layer_idx)


class Gen3Attention(nn.Module):
    """Global Attention Anchor supporting QK-RMSNorm, Gated-VRes, QK-Canon, and Differential Attention."""

    def __init__(self, c: Gen3Cfg, layer_idx: int = 0) -> None:
        super().__init__()
        self.h = c.n_head
        self.kv = c.n_kv
        self.hd = c.d // c.n_head
        assert self.hd % 4 == 0, "head_dim must be divisible by 4 for Diff-GQA + RoPE"

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
            self.lam_init = _lambda_init(layer_idx)
            self.lam_q1 = nn.Parameter(torch.zeros(self.h, half_hd))
            self.lam_k1 = nn.Parameter(torch.zeros(self.h, half_hd))
            self.lam_q2 = nn.Parameter(torch.zeros(self.h, half_hd))
            self.lam_k2 = nn.Parameter(torch.zeros(self.h, half_hd))
            nn.init.normal_(self.lam_q1, std=0.1)
            nn.init.normal_(self.lam_k1, std=0.1)
            nn.init.normal_(self.lam_q2, std=0.1)
            nn.init.normal_(self.lam_k2, std=0.1)
            self.sub_norm = RMSNorm(self.hd)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        v0: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, None]:
        B, T, _ = x.shape
        q_raw = self.q(x)
        k_raw = self.k(x)
        if self.qk_conv is not None:
            qk = self.qk_conv(torch.cat([q_raw, k_raw], dim=-1))
            q_raw, k_raw = qk.split([self.h * self.hd, self.kv * self.hd], dim=-1)

        q = self.qn(q_raw.view(B, T, self.h, self.hd)).transpose(1, 2)
        k = self.kn(k_raw.view(B, T, self.kv, self.hd)).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv, self.hd).transpose(1, 2)

        q = apply_rope(q, *rope)
        k = apply_rope(k, *rope)

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
            # Fused exact identity on MPS/CUDA/CPU: SDPA([q, q], [k, k], [va, vb]) -> [SDPA(q, k, va), SDPA(q, k, vb)]
            y1_cat = F.scaled_dot_product_attention(
                torch.cat([q1, q1], dim=1), torch.cat([k1, k1], dim=1), v_cat, is_causal=True
            )
            y2_cat = F.scaled_dot_product_attention(
                torch.cat([q2, q2], dim=1), torch.cat([k2, k2], dim=1), v_cat, is_causal=True
            )
            y1 = torch.cat(y1_cat.chunk(2, dim=1), dim=-1)
            y2 = torch.cat(y2_cat.chunk(2, dim=1), dim=-1)
            lam1 = torch.exp((self.lam_q1 * self.lam_k1).sum(dim=-1))
            lam2 = torch.exp((self.lam_q2 * self.lam_k2).sum(dim=-1))
            lam = (lam1 - lam2 + self.lam_init).view(1, self.h, 1, 1).to(y1.dtype)
            y = self.sub_norm(y1 - lam * y2) * (1.0 - self.lam_init)

        return self.o(y.transpose(1, 2).reshape(B, T, -1)), v0, None


class Gen3DeltaMixer(nn.Module):
    """Householder-capable Sharp-Delta Mixer with O(1) recurrent state and WY chunk parallelism."""

    def __init__(self, c: Gen3Cfg) -> None:
        super().__init__()
        self.h = c.n_head
        self.hd = c.d // c.n_head
        d = c.d
        self.sharp_v = c.sharp_v
        self.householder_beta = c.householder_beta
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.conv = ShortConv(2 * d if c.sharp_v else 3 * d, c.conv) if c.conv > 1 else None
        self.a = nn.Linear(d, self.h, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.linspace(1, 16, self.h)))
        self.b = nn.Linear(d, self.h, bias=True)
        if self.householder_beta:
            # Initialize bias to -ln(3) so 2 * sigmoid(-ln(3)) = 0.5 matches standard initial step size
            nn.init.constant_(self.b.bias, -math.log(3.0))
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
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        B, T, D = x.shape
        H, hd, C = self.h, self.hd, self.C
        qkv = self.qkv(x)
        if self.conv is None:
            q, k, v = qkv.split(D, dim=-1)
        elif self.sharp_v:
            qk, v_raw = qkv.split([2 * D, D], dim=-1)
            q, k = self.conv(qk).split(D, dim=-1)
            v = v_raw
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
        # Clamp Householder beta strictly inside (0, 1.995) for numerical conditioning at scale
        beta = (beta_scale * torch.sigmoid(self.b(x).float())).clamp(max=1.995).transpose(1, 2)

        if USE_FLA and x.is_cuda and not need_surprise:
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
                return self.o(y), v0, None
            except ImportError:
                pass

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
        eye = torch.eye(C, device=x.device, dtype=torch.float32)
        if x.device.type == "mps" and (C & (C - 1)) == 0:
            from .kernels import nilpotent_wy_inverse
            Tm = nilpotent_wy_inverse(Akk)
        else:
            # Native cuBLAS strsmBatched on CUDA: O(1)-bounded across full Householder beta in (0, 2) & minimal VRAM!
            Tm = torch.linalg.solve_triangular(eye + Akk, eye.expand_as(Akk), upper=False)
        la_exp = la.exp()
        w_ = Tm @ (kb * la_exp[..., None])
        u_ = Tm @ (v * beta_c[..., None])

        attn_all = (q @ k.transpose(-1, -2)) * decay
        q_scaled_all = q * la_exp[..., None]
        tail_all = (la[..., -1:] - la).exp()[..., None]
        kt_all = (k * tail_all).transpose(-1, -2)
        g_all = la[..., -1].exp()[..., None, None]

        S = x.new_zeros(B, H, hd, hd, dtype=torch.float32)
        u_list = []
        s_list = []
        for n in range(N):
            s_list.append(S)
            u = u_[:, :, n] - w_[:, :, n] @ S
            u_list.append(u)
            S = g_all[:, :, n] * S + kt_all[:, :, n] @ u

        u_stack = torch.stack(u_list, 2)
        s_stack = torch.stack(s_list, 2)
        ys_chunks = attn_all @ u_stack + q_scaled_all @ s_stack
        y = ys_chunks.view(B, H, N * C, hd)[:, :, :T]
        if need_surprise:
            err = u_stack / (beta_c.unsqueeze(-1) + 1e-4)
            surprise_t = err.norm(dim=-1).mean(dim=1).view(B, N * C)[:, :T]
        else:
            surprise_t = None
        y = self.on(y.transpose(1, 2).to(x.dtype)).reshape(B, T, D) * F.silu(self.g(x))
        return self.o(y), v0, surprise_t


class IdentityInitCausalConv(nn.Module):
    """Zero-overhead 3-tap depthwise causal conv initialized to exact dirac identity `[0, 0, 1]`."""

    def __init__(self, d: int, k: int = 3) -> None:
        super().__init__()
        self.k = k
        self.w = nn.Parameter(torch.zeros(d, 1, k))
        with torch.no_grad():
            self.w[:, 0, -1] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(
            F.pad(x.transpose(1, 2), (self.k - 1, 0)),
            self.w,
            groups=x.size(-1),
        ).transpose(1, 2)


class Gen3Block(nn.Module):
    def __init__(self, c: Gen3Cfg, kind: str, layer_idx: int, num_depth_anchors: int = 0) -> None:
        super().__init__()
        self.kind = kind
        self.layer_idx = layer_idx
        self.n1 = RMSNorm(c.d)
        self.n2 = RMSNorm(c.d)
        if kind == "A":
            self.mix = Gen3Attention(c, layer_idx=layer_idx)
        elif kind == "D":
            self.mix = Gen3DeltaMixer(c)
        else:
            raise ValueError(f"Unsupported mixer kind: {kind}")

        self.ffn_conv = IdentityInitCausalConv(c.d, k=3) if c.canon_ffn else None
        self.gu = nn.Linear(c.d, 2 * c.ffn, bias=False)
        self.dn = nn.Linear(c.ffn, c.d, bias=False)

        self.num_depth_anchors = num_depth_anchors
        if num_depth_anchors > 0:
            # Zero-initialized laminar depth-attention router: exact identity at step 0!
            self.depth_router = nn.Linear(c.d, num_depth_anchors, bias=False)
            self.depth_scale = nn.Parameter(torch.zeros(1))
            nn.init.zeros_(self.depth_router.weight)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        v0: torch.Tensor | None,
        depth_bank: list[torch.Tensor] | None = None,
        need_surprise: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.num_depth_anchors > 0 and depth_bank:
            k_avail = min(len(depth_bank), self.num_depth_anchors)
            # Per-token selective retrieval over macro-stage depth anchors
            scores = self.depth_router(self.n1(x))[..., :k_avail].float()
            weights = F.softmax(scores, dim=-1).to(x.dtype)  # [B, T, K]
            stacked = torch.stack(depth_bank[:k_avail], dim=-2)  # [B, T, K, D]
            retrieved = (weights.unsqueeze(-1) * stacked).sum(dim=-2)
            x = x + torch.tanh(self.depth_scale).to(x.dtype) * (retrieved - x)

        if self.kind == "D":
            h, v0, surp = self.mix(self.n1(x), rope, v0, need_surprise=need_surprise)
        else:
            h, v0, surp = self.mix(self.n1(x), rope, v0)
        x = x + h

        h2 = self.n2(x)
        if self.ffn_conv is not None:
            h2 = self.ffn_conv(h2)
        g, u = self.gu(h2).chunk(2, dim=-1)
        return x + self.dn(F.silu(g) * u), v0, surp


class Gen3LM(nn.Module):
    """Generation-3 Foundation Model Prototype (`NanoLM-v3`)."""

    def __init__(self, c: Gen3Cfg) -> None:
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d)
        # Macro-stage anchor indices: embedding (stage 0) + after every 'A' block
        anchor_counts: list[int] = []
        seen_anchors = 1  # embedding counts as anchor 0
        for idx, k in enumerate(c.layout):
            anchor_counts.append(seen_anchors if (c.depth_attnres and idx >= 2) else 0)
            if k == "A":
                seen_anchors += 1

        self.blocks = nn.ModuleList(
            Gen3Block(c, k, idx, num_depth_anchors=anchor_counts[idx])
            for idx, k in enumerate(c.layout)
        )
        self.nf = RMSNorm(c.d)
        self.head = None if c.tie else nn.Linear(c.d, c.vocab, bias=False)
        self.use_sgrd = bool(c.loop and c.sgrd_prior_p > 0.0)
        if c.loop:
            p, co, k_coda, _ = c.loop
            assert p + co + k_coda == len(c.layout)
            self.inject = nn.Linear(2 * c.d, c.d, bias=False)
            if self.use_sgrd:
                self.halt_proj = nn.Linear(c.d, 1, bias=True)
                self.halt_gamma = nn.Parameter(torch.tensor(0.5))
                nn.init.zeros_(self.halt_proj.weight)
                nn.init.constant_(self.halt_proj.bias, -0.5)
        self._init()

    def _init(self) -> None:
        n_eff = self.effective_depth()
        skip = ("emb", "v_gate", "halt_proj", "depth_router", "ffn_conv", "lam_")
        for name, p in self.named_parameters():
            if p.dim() == 2 and not any(tag in name for tag in skip):
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

    def param_report(self) -> dict[str, int]:
        tot = sum(p.numel() for p in self.parameters())
        emb = self.emb.weight.numel() + (0 if self.head is None else self.head.weight.numel())
        return {"total": tot, "non_embedding": tot - emb, "embedding": emb, "effective_layers": self.effective_depth()}

    def forward(
        self,
        idx: torch.Tensor,
        tgt: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        B, T = idx.shape
        x = self.emb(idx)
        hd = self.c.d // self.c.n_head
        rope = rope_cache(T, hd, self.c.rope_theta, idx.device)
        v0 = None
        depth_bank: list[torch.Tensor] = [x] if self.c.depth_attnres else []
        aux: dict[str, torch.Tensor | float] = {}

        if not self.c.loop:
            for b in self.blocks:
                x, v0, _ = b(x, rope, v0, depth_bank=depth_bank)
                if self.c.depth_attnres and b.kind == "A":
                    depth_bank.append(x)
        else:
            p, co, k_coda, r = self.c.loop
            for i in range(p):
                x, v0, _ = self.blocks[i](x, rope, v0, depth_bank=depth_bank)
                if self.c.depth_attnres and self.blocks[i].kind == "A":
                    depth_bank.append(x)
            anchor = x
            if not self.use_sgrd:
                for _ in range(r):
                    x = self.inject(torch.cat([x, anchor], dim=-1))
                    for i in range(p, p + co):
                        x, v0, _ = self.blocks[i](x, rope, v0, depth_bank=depth_bank)
            else:
                rem = x.new_ones(B, T, dtype=torch.float32)
                h_acc = torch.zeros_like(x, dtype=torch.float32)
                step_probs = []
                last_delta_idx = max((i for i in range(p, p + co) if self.c.layout[i] == "D"), default=-1)
                for step_i in range(r):
                    x = self.inject(torch.cat([x, anchor], dim=-1))
                    last_surp = None
                    for i in range(p, p + co):
                        need_s = i == last_delta_idx
                        x, v0, surp = self.blocks[i](x, rope, v0, depth_bank=depth_bank, need_surprise=need_s)
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
                probs = torch.stack(step_probs, dim=-1)
                x = h_acc.to(x.dtype)
                gp = self.c.sgrd_prior_p
                prior = torch.tensor(
                    [(1.0 - gp) ** j * gp if j < r - 1 else (1.0 - gp) ** (r - 1) for j in range(r)],
                    device=idx.device,
                    dtype=torch.float32,
                )
                kl = (probs.clamp_min(1e-7) * (probs.clamp_min(1e-7).log() - prior.log())).sum(dim=-1).mean()
                aux["sgrd_kl"] = kl
                aux["expected_loops"] = (probs * torch.arange(1, r + 1, device=idx.device, dtype=torch.float32)).sum(-1).mean().detach()
                aux["halt_gamma"] = self.halt_gamma.detach()

            for i in range(p + co, p + co + k_coda):
                x, v0, _ = self.blocks[i](x, rope, v0, depth_bank=depth_bank)
                if self.c.depth_attnres and self.blocks[i].kind == "A":
                    depth_bank.append(x)

        h = self.nf(x)
        lg = (h @ self.emb.weight.T if self.head is None else self.head(h)).float()
        if tgt is None:
            return (lg, aux) if return_aux else lg
        ce = F.cross_entropy(lg.view(-1, lg.size(-1)), tgt.reshape(-1), ignore_index=-100)
        total_loss = ce + (self.c.sgrd_kl_weight * aux["sgrd_kl"] if "sgrd_kl" in aux else 0.0)
        return (total_loss, ce.detach(), aux) if return_aux else total_loss
