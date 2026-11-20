"""GQA attention control from the NanoLM discovery lab at 3314170a (MIT)."""
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
        if kind != "A":
            raise ValueError("This experiment uses full self-attention")
        self.mix = Attention(c, None)
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
