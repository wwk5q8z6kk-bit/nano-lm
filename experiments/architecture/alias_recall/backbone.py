"""NanoLM discovery lab: configurable small-scale LM for controlled architecture experiments.

Every arm shares the same tokenizer, data stream, optimizer budget and evaluation.
Token mixers (one char per layer in ``layout``):
    A  global causal softmax attention (GQA, QK-RMSNorm, RoPE)
    S  sliding-window causal attention
    L  gated linear attention / SSD-style recurrence (scalar data-dependent decay per head,
       short causal conv, L2-normalised q/k, output gate). O(T) time, O(1) decode state.
    D  gated delta-rule recurrence (Gated DeltaNet-style; error-correcting associative memory)

Options:
    value_residual  ResFormer: every layer's V is mixed with layer-0 V via a learned scalar.
    loop            (prelude, core, coda, r): weight-shared core block applied r times.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Cfg:
    vocab: int = 32768
    d: int = 512
    n_head: int = 8
    n_kv: int = 4
    ffn: int = 1408
    layout: str = "AAAAAAAA"
    window: int = 256
    seq: int = 1024
    rope_theta: float = 10000.0
    value_residual: bool = False
    loop: tuple[int, int, int, int] | None = None  # (prelude, core, coda, r) -- layout length = p+c+k
    conv: int = 4
    chunk: int = 64
    tie: bool = True
    extra: dict = field(default_factory=dict)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.w, self.eps)


def rope_cache(T, hd, theta, device):
    inv = 1.0 / theta ** (torch.arange(0, hd, 2, device=device).float() / hd)
    f = torch.outer(torch.arange(T, device=device).float(), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):  # x: B,H,T,hd
    x1, x2 = x[..., ::2], x[..., 1::2]
    c, s = cos[None, None, : x.shape[2]], sin[None, None, : x.shape[2]]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, c: Cfg, window: int | None):
        super().__init__()
        self.h, self.kv, self.hd = c.n_head, c.n_kv, c.d // c.n_head
        self.q = nn.Linear(c.d, self.h * self.hd, bias=False)
        self.k = nn.Linear(c.d, self.kv * self.hd, bias=False)
        self.v = nn.Linear(c.d, self.kv * self.hd, bias=False)
        self.o = nn.Linear(self.h * self.hd, c.d, bias=False)
        self.qn, self.kn = RMSNorm(self.hd), RMSNorm(self.hd)
        self.window = window
        self.vr = c.value_residual
        if self.vr:
            self.lam = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, rope, v0):
        B, T, _ = x.shape
        q = self.qn(self.q(x).view(B, T, self.h, self.hd)).transpose(1, 2)
        k = self.kn(self.k(x).view(B, T, self.kv, self.hd)).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        if self.vr:
            if v0 is None:
                v0 = v
            else:
                v = self.lam * v + (1 - self.lam) * v0
        q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        rep = self.h // self.kv
        k, v_ = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        if self.window is None or self.window >= T:
            y = F.scaled_dot_product_attention(q, k, v_, is_causal=True)
        else:
            i = torch.arange(T, device=x.device)
            m = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < self.window)
            y = F.scaled_dot_product_attention(q, k, v_, attn_mask=m)
        return self.o(y.transpose(1, 2).reshape(B, T, -1)), v0


class ShortConv(nn.Module):
    def __init__(self, ch, k):
        super().__init__()
        self.k = k
        self.w = nn.Parameter(torch.randn(ch, 1, k) / math.sqrt(k))

    def forward(self, x):  # B,T,C
        x = F.pad(x.transpose(1, 2), (self.k - 1, 0))
        return F.silu(F.conv1d(x, self.w, groups=x.shape[1])).transpose(1, 2)


class LinearMixer(nn.Module):
    """Chunked gated linear attention ('L') or gated delta rule ('D').

    State S_t in R^{dk x dv} per head.
      L:  S_t = a_t S_{t-1} + k_t v_t^T                     y_t = S_t^T q_t
      D:  S_t = a_t (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T
    a_t = exp(-softplus(.)*exp(A_log)) in (0,1) is a data-dependent scalar decay per head.
    """

    def __init__(self, c: Cfg, delta: bool):
        super().__init__()
        self.h = c.n_head
        self.hd = c.d // c.n_head
        d = c.d
        self.delta = delta
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.conv = ShortConv(3 * d, c.conv)
        self.a = nn.Linear(d, self.h, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.linspace(1, 16, self.h)))
        self.g = nn.Linear(d, d, bias=False)
        self.on = RMSNorm(self.hd)
        self.o = nn.Linear(d, d, bias=False)
        if delta:
            self.b = nn.Linear(d, self.h, bias=True)
        self.C = c.chunk

    def forward(self, x, rope, v0):
        B, T, D = x.shape
        H, hd, C = self.h, self.hd, self.C
        q, k, v = self.conv(self.qkv(x)).split(D, -1)
        q = F.normalize(q.view(B, T, H, hd).float(), dim=-1).transpose(1, 2) * hd ** -0.5
        k = F.normalize(k.view(B, T, H, hd).float(), dim=-1).transpose(1, 2)
        v = v.view(B, T, H, hd).float().transpose(1, 2)
        loga = (-F.softplus(self.a(x).float()) * self.A_log.exp()).transpose(1, 2)  # B,H,T
        if self.delta:
            beta = torch.sigmoid(self.b(x).float()).transpose(1, 2)  # B,H,T
        if USE_FLA and x.is_cuda:
            return self._fla(x, q, k, v, loga, beta if self.delta else None, B, T, D)[0], v0
        pad = (-T) % C
        if pad:
            q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
            loga = F.pad(loga, (0, pad))
            if self.delta:
                beta = F.pad(beta, (0, pad))
        N = (T + pad) // C
        q, k, v = (t.view(B, H, N, C, hd) for t in (q, k, v))
        la = loga.view(B, H, N, C).cumsum(-1)  # cumulative log-decay within chunk
        tri = torch.tril(torch.ones(C, C, dtype=torch.bool, device=x.device))
        decay = (la[..., :, None] - la[..., None, :]).masked_fill(~tri, -1e30).exp()  # B,H,N,C,C
        if self.delta:
            beta = beta.view(B, H, N, C)
            kb = k * beta[..., None]
            # WY representation: solve (I + tril(beta K K^T * decay, -1)) U = beta * V_eff
            strict = torch.tril(torch.ones(C, C, dtype=torch.bool, device=x.device), -1)
            Akk = (kb @ k.transpose(-1, -2) * decay).masked_fill(~strict, 0)
            Tm = torch.linalg.solve_triangular(
                torch.eye(C, device=x.device) + Akk, torch.eye(C, device=x.device).expand_as(Akk),
                upper=False)
            w_ = Tm @ (kb * la.exp()[..., None])  # decay-weighted keys
            u_ = Tm @ (v * beta[..., None])
        S = x.new_zeros(B, H, hd, hd, dtype=torch.float32)
        ys = []
        for n in range(N):
            qn, kn, la_n, dn = q[:, :, n], k[:, :, n], la[:, :, n], decay[:, :, n]
            if self.delta:
                u = u_[:, :, n] - w_[:, :, n] @ S  # pseudo-values after correcting old memory
                attn = (qn @ kn.transpose(-1, -2)) * dn
                y = attn @ u + (qn * la_n.exp()[..., None]) @ S
                tail = (la_n[..., -1:] - la_n).exp()[..., None]
                S = la_n[..., -1].exp()[..., None, None] * S + (kn * tail).transpose(-1, -2) @ u
            else:
                vn = v[:, :, n]
                attn = (qn @ kn.transpose(-1, -2)) * dn
                y = attn @ vn + (qn * la_n.exp()[..., None]) @ S
                tail = (la_n[..., -1:] - la_n).exp()[..., None]
                S = la_n[..., -1].exp()[..., None, None] * S + (kn * tail).transpose(-1, -2) @ vn
            ys.append(y)
        y = torch.stack(ys, 2).view(B, H, N * C, hd)[:, :, :T]
        y = self.on(y.transpose(1, 2).to(x.dtype))  # B,T,H,hd
        y = y.reshape(B, T, D) * F.silu(self.g(x))
        return self.o(y), v0


def _fla_forward(self, x, q, k, v, loga, beta, B, T, D):
    """Same math via flash-linear-attention Triton kernels (CUDA only). Inputs are B,H,T,hd fp32."""
    tr = lambda t: t.transpose(1, 2).contiguous().to(torch.bfloat16)  # -> B,T,H,hd
    g = loga.transpose(1, 2).contiguous()  # B,T,H log-decay (fp32)
    if self.delta:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        y, _ = chunk_gated_delta_rule(tr(q), tr(k), tr(v), g, beta.transpose(1, 2).contiguous().to(torch.bfloat16),
                                      scale=1.0)
    else:
        from fla.ops.simple_gla import chunk_simple_gla
        y, _ = chunk_simple_gla(tr(q), tr(k), tr(v), g, scale=1.0)
    y = self.on(y.to(x.dtype))  # B,T,H,hd
    y = y.reshape(B, T, D) * F.silu(self.g(x))
    return self.o(y), None


LinearMixer._fla = _fla_forward
import os as _os  # noqa: E402

USE_FLA = _os.environ.get("NANOLM_FLA", "0") == "1"


class MLP(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.gu = nn.Linear(c.d, 2 * c.ffn, bias=False)
        self.dn = nn.Linear(c.ffn, c.d, bias=False)

    def forward(self, x):
        g, u = self.gu(x).chunk(2, -1)
        return self.dn(F.silu(g) * u)


class Block(nn.Module):
    def __init__(self, c: Cfg, kind: str):
        super().__init__()
        self.kind = kind
        self.n1, self.n2 = RMSNorm(c.d), RMSNorm(c.d)
        if kind == "A":
            self.mix = Attention(c, None)
        elif kind == "S":
            self.mix = Attention(c, c.window)
        elif kind in "LD":
            self.mix = LinearMixer(c, delta=kind == "D")
        else:
            raise ValueError(kind)
        self.mlp = MLP(c)

    def forward(self, x, rope, v0):
        h, v0 = self.mix(self.n1(x), rope, v0)
        x = x + h
        return x + self.mlp(self.n2(x)), v0


class LM(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d)
        self.blocks = nn.ModuleList(Block(c, k) for k in c.layout)
        self.nf = RMSNorm(c.d)
        self.head = None if c.tie else nn.Linear(c.d, c.vocab, bias=False)
        if c.loop:
            p, co, k, r = c.loop
            assert p + co + k == len(c.layout)
            self.inject = nn.Linear(2 * c.d, c.d, bias=False)
        self._init()

    def _init(self):
        n_eff = self.effective_depth()
        for name, p in self.named_parameters():
            if p.dim() == 2 and "emb" not in name:
                nn.init.normal_(p, std=0.02)
                if name.endswith(("o.weight", "dn.weight")):
                    p.data.mul_(1 / math.sqrt(2 * n_eff))
        nn.init.normal_(self.emb.weight, std=0.02)
        if self.c.loop:
            nn.init.zeros_(self.inject.weight)
            with torch.no_grad():
                self.inject.weight[:, : self.c.d].copy_(torch.eye(self.c.d))

    def effective_depth(self):
        if not self.c.loop:
            return len(self.c.layout)
        p, co, k, r = self.c.loop
        return p + co * r + k

    def schedule(self):
        if not self.c.loop:
            return list(range(len(self.blocks)))
        p, co, k, r = self.c.loop
        return list(range(p)) + list(range(p, p + co)) * r + list(range(p + co, p + co + k))

    def hidden(self, idx, loops: int | None = None):
        B, T = idx.shape
        x = self.emb(idx)
        hd = self.c.d // self.c.n_head
        rope = rope_cache(T, hd, self.c.rope_theta, idx.device)
        v0 = None
        if not self.c.loop:
            for b in self.blocks:
                x, v0 = b(x, rope, v0)
        else:
            p, co, k, r = self.c.loop
            r = loops or r
            for i in range(p):
                x, v0 = self.blocks[i](x, rope, v0)
            e = x
            for _ in range(r):
                x = self.inject(torch.cat([x, e], -1))
                for i in range(p, p + co):
                    x, v0 = self.blocks[i](x, rope, v0)
            for i in range(p + co, p + co + k):
                x, v0 = self.blocks[i](x, rope, v0)
        return self.nf(x)

    def logits(self, h):
        return h @ self.emb.weight.T if self.head is None else self.head(h)

    def forward(self, idx, tgt=None, loops=None):
        h = self.hidden(idx, loops)
        lg = self.logits(h).float()
        if tgt is None:
            return lg
        return F.cross_entropy(lg.view(-1, lg.size(-1)), tgt.reshape(-1), reduction="mean")

    def param_report(self):
        tot = sum(p.numel() for p in self.parameters())
        emb = self.emb.weight.numel() + (self.head.weight.numel() if self.head is not None else 0)
        non_emb = tot - emb
        # active non-embedding *applications* per token (looped blocks counted r times)
        sched = self.schedule()
        per_block = [sum(p.numel() for p in b.parameters()) for b in self.blocks]
        applied = sum(per_block[i] for i in sched)
        return {"total": tot, "embedding": emb, "non_embedding": non_emb,
                "non_embedding_applied_per_token": applied, "effective_depth": len(sched)}
