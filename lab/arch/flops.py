"""Analytic training-FLOP accounting per token (forward+backward = 3x forward).

F_token = 6 * (applied non-embedding matmul params)      # weights incl. looped re-application
        + 6 * d * V                                       # tied LM head (embedding lookup is free)
        + sum_layers 3 * mixer_context_flops_fwd          # attention scores / recurrent state ops

mixer_context_flops_fwd per token:
    A (global, causal):   4 * d * T/2            (QK^T and AV over on average T/2 keys)
    S (window W):         4 * d * W_eff,  W_eff = W - W^2/(2T)  (average visible keys)
    L (gated linear):     4 * d * C/2 + 4 * d * hd        (intra-chunk + state read/write)
    D (gated delta):      L + 2 * d * hd + 2 * d * C      (correction w@S + WY solve, amortised)
Approximations are stated so they can be audited; they are used only for iso-FLOP matching.
"""

from __future__ import annotations

from lab.arch.model import LM


def mixer_fwd(kind: str, d: int, hd: int, T: int, W: int, C: int) -> float:
    if kind == "A":
        return 4 * d * T / 2
    if kind == "S":
        w = min(W, T)
        return 4 * d * (w - w * w / (2 * T))
    if kind == "L":
        return 4 * d * C / 2 + 4 * d * hd
    if kind == "D":
        return 4 * d * C / 2 + 4 * d * hd + 2 * d * hd + 2 * d * C
    raise ValueError(kind)


def train_flops_per_token(m: LM, T: int | None = None) -> dict:
    c = m.c
    T = T or c.seq
    hd = c.d // c.n_head
    rep = m.param_report()
    # matmul params exclude norms/conv/scalars (tiny); counting them is a <0.5% effect
    weights = 6 * rep["non_embedding_applied_per_token"]
    head = 6 * c.d * c.vocab
    mix = sum(3 * mixer_fwd(c.layout[i], c.d, hd, T, c.window, c.chunk) for i in m.schedule())
    return {"weights": weights, "lm_head": head, "mixer_context": mix, "total": weights + head + mix}
