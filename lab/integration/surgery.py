"""Controlled Architecture Surgery & Progressive Hybridization (`LAB-INT-01`: Track A -> Track B).

Implements and empirically compares three mechanisms for transferring Track A discoveries
(`Gated-VRes` and `Sharp-Delta` 3:1 hybrid mixers) onto an already-pretrained Dense GQA Transformer
while preserving original weights untouched:

  1. `ZeroInitGatedVResAttention`: Exact function-preserving graft of Track A's per-head input-gated
     Layer-0 Value Highway:
         v^{(l)}_{b,h,t} <- v^{(l)}_{b,h,t} + 0.5 * tanh(gate(x_{b,t})) * (v^{(0)}_{b,h,t} - v^{(l)}_{b,h,t})
     With `gate` zero-initialized, step-0 logits are bit-for-bit identical to the pretrained model
     (KL(P_pre || P_graft) == 0.0), guaranteeing zero capability regression before adaptation.

  2. `warmstart_attn_to_sharp_delta`: Transfers pretrained GQA `(q, k, v, o)` projections into a
     `NextGenDeltaMixer(sharp_v=True)` (expanding grouped KV heads via `repeat_interleave` and
     setting `ShortConv` to causal lag-0 impulse) for `-75%` to `-80%` KV-cache reduction.

  3. `distill_hybrid_progressive`: Two-stage RAD/MambaInLlama-style progressive conversion:
       Stage 1: Layerwise mixer-output MSE alignment ||y_delta^{(l)} - y_attn^{(l)}||_2^2
       Stage 2: End-to-end Proper Top-K + Tail-Bucket Logit KD from the frozen pretrained teacher.
"""

from __future__ import annotations

import copy
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from lab.arch.model import apply_rope
from lab.arch.nextgen import NextGenAttention, NextGenCfg, NextGenDeltaMixer, NextGenLM
from lab.trackb.posttrain_pipeline import extract_topk_teacher_targets, proper_topk_kd_loss


class ZeroInitGatedVResAttention(nn.Module):
    """Wraps a pretrained NextGenAttention with an exact step-0 identity Layer-0 Value Highway."""

    def __init__(self, base_attn: NextGenAttention):
        super().__init__()
        self.h = base_attn.h
        self.kv = base_attn.kv
        self.hd = base_attn.hd
        self.q = copy.deepcopy(base_attn.q)
        self.k = copy.deepcopy(base_attn.k)
        self.v = copy.deepcopy(base_attn.v)
        self.o = copy.deepcopy(base_attn.o)
        self.qn = copy.deepcopy(base_attn.qn)
        self.kn = copy.deepcopy(base_attn.kn)
        self.qk_conv = copy.deepcopy(base_attn.qk_conv) if base_attn.qk_conv is not None else None
        # Exact zero-initialized residual gate: at step 0, tanh(0) = 0 -> exact pretrained forward!
        d_model = self.q.in_features
        self.highway_gate = nn.Linear(d_model, self.h, bias=True)
        nn.init.zeros_(self.highway_gate.weight)
        nn.init.zeros_(self.highway_gate.bias)

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
        v_exp = v.repeat_interleave(rep, 1)
        if v0 is None:
            v0 = v_exp
        else:
            lam = 0.5 * torch.tanh(self.highway_gate(x)).transpose(1, 2).unsqueeze(-1)  # in (-0.5, 0.5), 0 at init
            v_exp = v_exp + lam.to(v_exp.dtype) * (v0.to(v_exp.dtype) - v_exp)
        y = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, -1)), v0, None


def graft_zero_init_gated_vres(pretrained_dense: NextGenLM) -> NextGenLM:
    """Return a deep-copied model with ZeroInitGatedVResAttention grafted onto all Attention blocks."""
    m = copy.deepcopy(pretrained_dense)
    for b in m.blocks:
        if b.kind == "A" and isinstance(b.mix, NextGenAttention):
            b.mix = ZeroInitGatedVResAttention(b.mix).to(
                device=next(pretrained_dense.parameters()).device,
                dtype=next(pretrained_dense.parameters()).dtype,
            )
    return m


@torch.no_grad()
def warmstart_attn_to_sharp_delta(attn: NextGenAttention, cfg: NextGenCfg) -> NextGenDeltaMixer:
    """Construct a Sharp-Delta mixer warm-started from pretrained GQA Q, K, V, O weights."""
    c_delta = copy.deepcopy(cfg)
    c_delta.sharp_v = True
    delta = NextGenDeltaMixer(c_delta).to(device=attn.q.weight.device, dtype=attn.q.weight.dtype)

    d = cfg.d
    rep = attn.h // attn.kv
    # Expand K and V output rows from (kv * hd, d) -> (h * hd, d) matching GQA repeat_interleave
    k_w = attn.k.weight.view(attn.kv, attn.hd, d).repeat_interleave(rep, dim=0).reshape(d, d)
    v_w = attn.v.weight.view(attn.kv, attn.hd, d).repeat_interleave(rep, dim=0).reshape(d, d)
    delta.qkv.weight.copy_(torch.cat([attn.q.weight, k_w, v_w], dim=0))
    delta.o.weight.copy_(attn.o.weight)

    # Set depthwise causal ShortConv to pass current-step features cleanly at init
    delta.conv.w.zero_()
    delta.conv.w[:, 0, -1] = 1.0
    # Start output gate linear projection small/balanced so silu(g(x)) doesn't explode
    nn.init.normal_(delta.g.weight, std=0.02)
    return delta


def build_surgical_hybrid(
    pretrained_dense: NextGenLM,
    target_layout: str,
    warmstart_qkv: bool = True,
) -> NextGenLM:
    """Convert selected 'A' layers into 'D' (Sharp-Delta) layers without mutating `pretrained_dense`."""
    assert len(target_layout) == len(pretrained_dense.blocks)
    m = copy.deepcopy(pretrained_dense)
    m.c.layout = target_layout
    m.c.sharp_v = True
    dev = next(pretrained_dense.parameters()).device
    dt = next(pretrained_dense.parameters()).dtype
    for idx, (blk, kind) in enumerate(zip(m.blocks, target_layout)):
        if kind == "D" and blk.kind == "A":
            blk.kind = "D"
            if warmstart_qkv:
                blk.mix = warmstart_attn_to_sharp_delta(pretrained_dense.blocks[idx].mix, m.c)
            else:
                c_d = copy.deepcopy(m.c)
                c_d.sharp_v = True
                blk.mix = NextGenDeltaMixer(c_d).to(device=dev, dtype=dt)
    return m


def capture_mixer_outputs(model: NextGenLM, idx: torch.Tensor) -> list[torch.Tensor]:
    """Run forward pass capturing each block's mixer output y^{(l)} for layerwise alignment."""
    from lab.arch.model import rope_cache

    B, T = idx.shape
    x = model.emb(idx)
    hd = model.c.d // model.c.n_head
    rope = rope_cache(T, hd, model.c.rope_theta, idx.device)
    v0 = None
    outs: list[torch.Tensor] = []
    for b in model.blocks:
        if b.kind == "D":
            h, v0, _ = b.mix(b.n1(x), rope, v0, need_surprise=False)
        else:
            h, v0, _ = b.mix(b.n1(x), rope, v0)
        outs.append(h)
        x = x + h
        g, u = b.gu(b.n2(x)).chunk(2, dim=-1)
        x = x + b.dn(F.silu(g) * u)
    return outs


def stage1_align_mixers(
    teacher_dense: NextGenLM,
    student_hybrid: NextGenLM,
    batches: Sequence[torch.Tensor],
    lr: float = 2e-3,
) -> list[float]:
    """Stage 1: Freeze non-converted parameters; train only 'D' mixers to match teacher mixer outputs."""
    teacher_dense.eval()
    student_hybrid.train()
    for p in student_hybrid.parameters():
        p.requires_grad_(False)
    converted_indices = [i for i, b in enumerate(student_hybrid.blocks) if b.kind == "D"]
    for i in converted_indices:
        for p in student_hybrid.blocks[i].mix.parameters():
            p.requires_grad_(True)

    opt = torch.optim.AdamW([p for p in student_hybrid.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    losses: list[float] = []
    from lab.arch.model import rope_cache

    for xb in batches:
        B, T = xb.shape
        hd = teacher_dense.c.d // teacher_dense.c.n_head
        rope = rope_cache(T, hd, teacher_dense.c.rope_theta, xb.device)
        with torch.no_grad():
            x_t = teacher_dense.emb(xb)
            v0_t = None
            t_in_normed = []
            t_mix_out = []
            for b in teacher_dense.blocks:
                n1x = b.n1(x_t)
                h_t, v0_t, _ = b.mix(n1x, rope, v0_t)
                t_in_normed.append(n1x.detach())
                t_mix_out.append(h_t.detach())
                x_t = x_t + h_t
                g, u = b.gu(b.n2(x_t)).chunk(2, dim=-1)
                x_t = x_t + b.dn(F.silu(g) * u)

        mse_total = xb.new_zeros((), dtype=torch.float32)
        for i in converted_indices:
            h_s, _, _ = student_hybrid.blocks[i].mix(t_in_normed[i], rope, None, need_surprise=False)
            mse_total = mse_total + F.mse_loss(h_s.float(), t_mix_out[i].float())
        mse_total = mse_total / max(1, len(converted_indices))
        opt.zero_grad(set_to_none=True)
        mse_total.backward()
        torch.nn.utils.clip_grad_norm_([p for p in student_hybrid.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append(float(mse_total.detach()))
    return losses


def stage2_distill_end_to_end(
    teacher_dense: NextGenLM,
    student: NextGenLM,
    batches_xy: Sequence[tuple[torch.Tensor, torch.Tensor]],
    lr: float = 5e-4,
    alpha_kd: float = 0.85,
    topk: int = 64,
    temperature: float = 1.5,
) -> list[float]:
    """Stage 2: Unfreeze student and distill end-to-end with Proper Top-K + Tail-Bucket Logit KD."""
    teacher_dense.eval()
    student.train()
    for p in student.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    kls: list[float] = []
    for xb, yb in batches_xy:
        with torch.no_grad():
            t_logits = teacher_dense(xb)
            tk_i, tk_lp = extract_topk_teacher_targets(t_logits, k=topk, temperature=temperature)
        s_logits = student(xb)
        out = proper_topk_kd_loss(
            s_logits, tk_i, tk_lp, yb, temperature=temperature, alpha_kd=alpha_kd
        )
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        kls.append(float(out["kd_kl"]))
    return kls
