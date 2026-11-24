"""Sequence baselines over the flat event stream. One position per event for every arm."""
from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F


def rope(x, base=10000.0):
    B, H, T, D = x.shape
    half = D // 2
    freq = base ** (-torch.arange(half, device=x.device).float() / half)
    ang = torch.arange(T, device=x.device).float()[:, None] * freq[None]
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        return x + self.ff(self.n2(x))


class EventTransformer(nn.Module):
    """Dense causal Transformer. loops > 1 re-applies the same stack (weight-tied iteration)."""
    def __init__(self, symbols, d=64, heads=4, layers=4, loops=1):
        super().__init__()
        self.sym, self.op = nn.Embedding(symbols + 1, d), nn.Embedding(5, d)
        self.inp = nn.Linear(3 * d, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.loops, self.norm = loops, nn.LayerNorm(d)
        self.head = nn.Linear(d, symbols + 1)

    def forward(self, ev):
        """ev [B,T,5] -> logits [B,T,S+1]"""
        x = self.inp(torch.cat([self.op(ev[..., 0]), self.sym(ev[..., 1]), self.sym(ev[..., 2])], -1))
        for _ in range(self.loops):
            for b in self.blocks: x = b(x)
        return self.head(self.norm(x))
