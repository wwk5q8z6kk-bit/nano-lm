from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.cycle1.exp_recurrent_operator_01 import _gen_d4_batch, _gen_d4_commutator_probe
from research.cycle1.primitives import (
    CausalAttentionMixer,
    CausalCopyPointerHead,
    HDCDMixer,
    RMSNorm,
    SwiGLUFFN,
    _apply_rope,
    _cayley_woodbury_factors,
    apply_skew_cayley_right,
    count_trainable_params,
    make_adamw_no_embed_decay,
    seed_everything,
    summarize_seeds,
)

torch.set_num_threads(1)


def _git_sha(repo_dir: Path) -> str:
    res = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return res.stdout.strip() or "unknown"


class StepCausalAttentionMixer(CausalAttentionMixer):
    """CausalAttentionMixer extended with exact RoPE + k_conv incremental KV-cache step()."""

    def step(
        self,
        x_t: torch.Tensor,
        state: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        k_ring, k_hist, v_hist = state if state is not None else (None, None, None)
        bs = x_t.shape[0]
        pos = 0 if k_hist is None else k_hist.shape[2]

        q_t = self.q_norm(self.q_proj(x_t).view(bs, 1, self.n_heads, self.head_dim))
        k_conv_t, next_k_ring = self.k_conv.step(self.k_proj(x_t), k_ring)
        k_t = self.k_norm(k_conv_t.view(bs, 1, self.n_heads, self.head_dim))

        q_t = _apply_rope(q_t, offset=pos).transpose(1, 2)  # [B, H, 1, d_h]
        k_t = _apply_rope(k_t, offset=pos).transpose(1, 2)  # [B, H, 1, d_h]
        v_t = self.v_proj(x_t).view(bs, 1, self.n_heads, self.head_dim).transpose(1, 2)

        k_cat = k_t if k_hist is None else torch.cat([k_hist, k_t], dim=2)
        v_cat = v_t if v_hist is None else torch.cat([v_hist, v_t], dim=2)
        scores = torch.matmul(q_t, k_cat.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_cat).transpose(1, 2).contiguous().view(bs, self.d_model)
        return self.o_proj(out), (next_k_ring, k_cat, v_cat)


class HDCDWriteShieldMixer(HDCDMixer):
    """HDCDMixer with closed-init slow-guild write/erase biases, decay floor >= 0.90, and L1 exposure."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        conv_kernel: int = 3,
        bw_slow_bias: float = -2.5,
    ) -> None:
        super().__init__(d_model, n_heads=n_heads, conv_kernel=conv_kernel)
        with torch.no_grad():
            self.bw_proj.bias[1 : n_heads - 1].fill_(bw_slow_bias)
            self.be_proj.bias[1 : n_heads - 1].fill_(bw_slow_bias)
        self.last_bw_slow_l1: torch.Tensor | None = None

    def forward_with_state(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None] | None = None,
        step_mode: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]:
        b, t, d = x.shape
        h, dh = self.n_heads, self.head_dim
        if state is None:
            s = x.new_zeros(b, h, dh, dh)
            q_ring, k_ring = None, None
        else:
            s, q_ring, k_ring = state[0].clone(), state[1], state[2]

        if step_mode:
            xq, q_ring = self.q_conv.step(self.q_proj(x[:, 0]), q_ring)
            xk, k_ring = self.k_conv.step(self.k_proj(x[:, 0]), k_ring)
            xq, xk = xq.unsqueeze(1), xk.unsqueeze(1)
        else:
            xq = self.q_conv(self.q_proj(x))
            xk = self.k_conv(self.k_proj(x))

        q = F.normalize(xq.view(b, t, h, dh), dim=-1, eps=1e-6)
        k = F.normalize(xk.view(b, t, h, dh), dim=-1, eps=1e-6)
        self.last_keys = k.detach()
        v = self.v_proj(x).view(b, t, h, dh)

        ra = torch.tanh(self.ra_proj(x).view(b, t, self.rot_heads, dh))
        rb = torch.tanh(self.rb_proj(x).view(b, t, self.rot_heads, dh))
        inv2, u1, u2 = _cayley_woodbury_factors(ra, rb)
        u_hh = F.normalize(self.u_proj(x).view(b, t, self.rot_heads, dh), dim=-1, eps=1e-6)
        rho_hh = 2.0 * torch.sigmoid(self.r_gate(x))  # [B, T, 1]

        a_raw = self.a_proj(x)
        alpha = torch.cat(
            [
                torch.sigmoid(a_raw[..., :1]),
                0.90 + 0.099 * torch.sigmoid(a_raw[..., 1 : h - 1]),
                torch.sigmoid(a_raw[..., h - 1 :]),
            ],
            dim=-1,
        )
        be = 2.0 * torch.sigmoid(self.be_proj(x))
        bw = 2.0 * torch.sigmoid(self.bw_proj(x))
        self.last_bw_slow_l1 = bw[..., 1 : h - 1].mean()
        gate = torch.sigmoid(self.g_proj(x)).view(b, t, h, dh)

        outs = []
        for i in range(t):
            ki, vi, qi = k[:, i], v[:, i], q[:, i]
            ai = alpha[:, i].view(b, h, 1, 1)
            bei = be[:, i].view(b, h, 1)
            bwi = bw[:, i].view(b, h, 1)

            s_fast_slow = s[:, :-1]
            s_cay = apply_skew_cayley_right(
                s[:, -1:], ra[:, i], rb[:, i], inv2[:, i], u1[:, i], u2[:, i]
            )
            ui = u_hh[:, i]
            rhoi = rho_hh[:, i].view(b, 1, 1)
            s_u = torch.einsum("bhij,bhj->bhi", s_cay, ui)
            s_g2 = s_cay - torch.einsum("bhi,bhj->bhij", s_u, rhoi * ui)
            s_rot = torch.cat([s_fast_slow, s_g2], dim=1)

            k_s = torch.einsum("bhi,bhij->bhj", ki, s_rot)
            s = ai * (s_rot - torch.einsum("bhi,bhj->bhij", ki, bei * k_s)) + torch.einsum("bhi,bhj->bhij", ki, bwi * vi)
            outs.append(torch.einsum("bhi,bhij->bhj", qi, s))
        y = torch.stack(outs, dim=1)
        y = self.o_norm(y) * gate
        return self.o_proj(y.reshape(b, t, d)), (s, q_ring, k_ring)


class HybridBlock(nn.Module):
    """Pre-norm block hosting either 'attn' (StepCausalAttentionMixer), 'hdcd', or 'hdcd_writeshield'."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, mixer_kind: str) -> None:
        super().__init__()
        self.mixer_kind = mixer_kind
        self.norm1 = RMSNorm(d_model)
        if mixer_kind == "attn":
            self.mixer: nn.Module = StepCausalAttentionMixer(d_model, n_heads, conv_kernel=3)
        elif mixer_kind == "hdcd":
            self.mixer = HDCDMixer(d_model, n_heads, conv_kernel=3)
        elif mixer_kind == "hdcd_writeshield":
            self.mixer = HDCDWriteShieldMixer(d_model, n_heads, conv_kernel=3)
        else:
            raise ValueError(f"Unknown mixer_kind: {mixer_kind}")
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

    def step(self, x_t: torch.Tensor, state: Any = None) -> tuple[torch.Tensor, Any]:
        m_out, next_state = self.mixer.step(self.norm1(x_t), state)
        x_t = x_t + m_out
        x_t = x_t + self.ffn(self.norm2(x_t))
        return x_t, next_state


class HybridSeqModel(nn.Module):
    """Per-layer heterogeneous sequence model supporting exact forward and incremental step decoding."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 48,
        n_heads: int = 4,
        d_ff: int = 96,
        mixer_kinds: list[str] | tuple[str, ...] = ("hdcd", "attn"),
        pointer_mode: str = "none",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mixer_kinds = list(mixer_kinds)
        self.pointer_mode = pointer_mode
        self.embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embed.weight, std=0.1)
        self.blocks = nn.ModuleList(
            [HybridBlock(d_model, n_heads, d_ff, mk) for mk in self.mixer_kinds]
        )
        self.norm_f = RMSNorm(d_model)
        if pointer_mode == "standard":
            self.pointer: CausalCopyPointerHead | None = CausalCopyPointerHead(d_model, decouple_backbone=False)
        else:
            self.pointer = None

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm_f(h)
        vocab_logits = F.linear(h, self.embed.weight)
        if self.pointer is None:
            log_probs = F.log_softmax(vocab_logits, dim=-1)
            if return_aux:
                return log_probs, {"vocab_logits": vocab_logits, "log_probs": log_probs}
            return log_probs
        ptr_out = self.pointer(h, x, vocab_logits)
        if return_aux:
            return ptr_out["log_probs"], ptr_out
        return ptr_out["log_probs"]

    def step(
        self,
        tok_t: torch.Tensor,
        state: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if state is None:
            layer_states: list[Any] = [None] * len(self.blocks)
            h_hist: torch.Tensor | None = None
            tok_hist: torch.Tensor | None = None
        else:
            layer_states = list(state["layers"])
            h_hist = state.get("h_hist")
            tok_hist = state.get("tok_hist")

        h_t = self.embed(tok_t)
        new_layer_states: list[Any] = []
        for blk, lst in zip(self.blocks, layer_states):
            h_t, nxt = blk.step(h_t, lst)
            new_layer_states.append(nxt)
        h_norm = self.norm_f(h_t)
        vocab_logits = F.linear(h_norm, self.embed.weight)

        if self.pointer is None:
            log_probs = F.log_softmax(vocab_logits, dim=-1)
            new_state = {"layers": new_layer_states}
            return log_probs, new_state

        h_cat = h_norm.unsqueeze(1) if h_hist is None else torch.cat([h_hist, h_norm.unsqueeze(1)], dim=1)
        tok_cat = tok_t.unsqueeze(1) if tok_hist is None else torch.cat([tok_hist, tok_t.unsqueeze(1)], dim=1)
        # Recompute causal pointer over exact history up to step t for full parity with forward()
        full_vocab_logits = F.linear(h_cat, self.embed.weight)
        ptr_out = self.pointer(h_cat, tok_cat, full_vocab_logits)
        return ptr_out["log_probs"][:, -1, :], {"layers": new_layer_states, "h_hist": h_cat, "tok_hist": tok_cat}

    def bw_slow_l1_loss(self) -> torch.Tensor:
        terms = [
            blk.mixer.last_bw_slow_l1
            for blk in self.blocks
            if isinstance(blk.mixer, HDCDWriteShieldMixer) and blk.mixer.last_bw_slow_l1 is not None
        ]
        if not terms:
            return torch.tensor(0.0)
        return sum(terms) / len(terms)


def _match_hybrid_d_ff(
    target_non_emb_params: int,
    vocab_size: int,
    d_model: int,
    n_heads: int,
    mixer_kinds: list[str],
    pointer_mode: str,
) -> int:
    probe = HybridSeqModel(vocab_size, d_model, n_heads, d_ff=16, mixer_kinds=mixer_kinds, pointer_mode=pointer_mode)
    probe_non_emb = count_trainable_params(probe) - probe.embed.weight.numel()
    per_dff = len(mixer_kinds) * 3 * d_model
    d_ff = max(16, round(16 + (target_non_emb_params - probe_non_emb) / per_dff))
    return max(16, int(round(d_ff / 4) * 4))


def _gen_mixed_extended_batch(
    rng: np.random.Generator,
    batch_size: int,
    val_lo: int,
    val_hi: int,
    n_keys: int = 4,
    n_overwrites: int = 2,
    syntax_len: int = 12,
    n_queries: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extended Mixed_FastSyntax_SlowMemory in vocab=44 allowing disjoint seen (10..21) and unseen (36..43) KV values."""
    sep, pad = 34, 35
    ev_len = 2 * (n_keys + n_overwrites)
    total_len = ev_len + syntax_len + 1 + 2 * n_queries
    xs = np.full((batch_size, total_len), pad, dtype=np.int64)
    ys = np.full((batch_size, total_len), -100, dtype=np.int64)
    syn_mask = np.zeros((batch_size, total_len), dtype=bool)
    kv_mask = np.zeros((batch_size, total_len), dtype=bool)

    for b in range(batch_size):
        keys = rng.choice(10, size=n_keys, replace=False)
        vals = rng.integers(val_lo, val_hi, size=n_keys)
        cur = {int(k): int(v) for k, v in zip(keys, vals)}
        pos = 0
        for k, v in zip(keys, vals):
            xs[b, pos] = int(k)
            xs[b, pos + 1] = int(v)
            pos += 2
        for ok in rng.choice(keys, size=n_overwrites, replace=True):
            nv = int(rng.integers(val_lo, val_hi))
            while nv == cur[int(ok)]:
                nv = int(rng.integers(val_lo, val_hi))
            cur[int(ok)] = nv
            xs[b, pos] = int(ok)
            xs[b, pos + 1] = nv
            pos += 2

        a = int(rng.integers(0, 12))
        b_tok = int(rng.integers(0, 12))
        xs[b, pos] = 22 + a
        xs[b, pos + 1] = 22 + b_tok
        for s_i in range(2, syntax_len):
            nxt = (a + 2 * b_tok) % 12
            xs[b, pos + s_i] = 22 + nxt
            ys[b, pos + s_i - 1] = 22 + nxt
            syn_mask[b, pos + s_i - 1] = True
            a, b_tok = b_tok, nxt
        pos += syntax_len
        xs[b, pos] = sep
        pos += 1

        for qk in rng.choice(keys, size=n_queries, replace=False):
            ans = cur[int(qk)]
            xs[b, pos] = int(qk)
            xs[b, pos + 1] = ans
            ys[b, pos] = ans
            kv_mask[b, pos] = True
            pos += 2

    return torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(syn_mask), torch.from_numpy(kv_mask)


def _train_eval_arm_seed(
    seed: int,
    arm_name: str,
    mixer_kinds: list[str],
    pointer_mode: str,
    bw_l1_weight: float,
    ref_non_emb_mixed: int,
    ref_non_emb_d4: int,
    steps_mixed: int = 260,
    steps_d4: int = 240,
) -> dict[str, float]:
    # --- Task 1: Extended Mixed_FastSyntax_SlowMemory (vocab=44) ---
    seed_everything(seed)
    rng_m = np.random.default_rng(60_000 + seed)
    dff_m = _match_hybrid_d_ff(ref_non_emb_mixed, 44, 48, 4, mixer_kinds, pointer_mode)
    m_mixed = HybridSeqModel(vocab_size=44, d_model=48, n_heads=4, d_ff=dff_m, mixer_kinds=mixer_kinds, pointer_mode=pointer_mode)
    opt_m = make_adamw_no_embed_decay(m_mixed, lr=4e-3, weight_decay=0.01)

    m_mixed.train()
    for _ in range(steps_mixed):
        xb, yb, _, _ = _gen_mixed_extended_batch(rng_m, batch_size=32, val_lo=10, val_hi=22)
        log_probs = m_mixed(xb)
        loss = F.nll_loss(log_probs.reshape(-1, 44), yb.reshape(-1), ignore_index=-100)
        if bw_l1_weight > 0.0:
            loss = loss + bw_l1_weight * m_mixed.bw_slow_l1_loss()
        opt_m.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m_mixed.parameters(), 1.0)
        opt_m.step()

    m_mixed.eval()
    eval_rng = np.random.default_rng(65_000 + seed)
    with torch.no_grad():
        xs_seen, ys_seen, syn_m, kv_m = _gen_mixed_extended_batch(eval_rng, batch_size=128, val_lo=10, val_hi=22)
        pred_seen = m_mixed(xs_seen).argmax(dim=-1)
        mixed_syntax_acc = float((pred_seen[syn_m] == ys_seen[syn_m]).float().mean().item())
        mixed_distant_kv_acc = float((pred_seen[kv_m] == ys_seen[kv_m]).float().mean().item())

        xs_unseen, ys_unseen, _, kv_u = _gen_mixed_extended_batch(eval_rng, batch_size=128, val_lo=36, val_hi=44)
        pred_unseen = m_mixed(xs_unseen).argmax(dim=-1)
        unseen_distant_kv_acc = float((pred_unseen[kv_u] == ys_unseen[kv_u]).float().mean().item())

    # --- Task 2: NonAbelian_D4_WordTrack (vocab=16) ---
    seed_everything(seed + 100)
    rng_d = np.random.default_rng(70_000 + seed)
    dff_d = _match_hybrid_d_ff(ref_non_emb_d4, 16, 48, 4, mixer_kinds, pointer_mode)
    m_d4 = HybridSeqModel(vocab_size=16, d_model=48, n_heads=4, d_ff=dff_d, mixer_kinds=mixer_kinds, pointer_mode=pointer_mode)
    opt_d = make_adamw_no_embed_decay(m_d4, lr=4e-3, weight_decay=0.01)

    m_d4.train()
    for _ in range(steps_d4):
        xd, yd = _gen_d4_batch(rng_d, batch_size=32, seq_ops=10, noop_prob=0.25)
        log_probs = m_d4(xd)
        loss = F.nll_loss(log_probs.reshape(-1, 16), yd.reshape(-1), ignore_index=-100)
        if bw_l1_weight > 0.0:
            loss = loss + bw_l1_weight * m_d4.bw_slow_l1_loss()
        opt_d.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m_d4.parameters(), 1.0)
        opt_d.step()

    m_d4.eval()
    d4_eval_rng = np.random.default_rng(75_000 + seed)
    with torch.no_grad():
        xd_id, yd_id = _gen_d4_batch(d4_eval_rng, batch_size=128, seq_ops=10, noop_prob=0.25)
        p_id = m_d4(xd_id)[:, -1, :].argmax(dim=-1)
        d4_id_10 = float((p_id == yd_id[:, -1]).float().mean().item())

        xd_ood, yd_ood = _gen_d4_batch(d4_eval_rng, batch_size=128, seq_ops=30, noop_prob=0.50)
        p_ood = m_d4(xd_ood)[:, -1, :].argmax(dim=-1)
        d4_ood_30_noisy = float((p_ood == yd_ood[:, -1]).float().mean().item())

        xc, yc = _gen_d4_commutator_probe(seq_ops=10)
        pc = m_d4(xc)[:, -1, :].argmax(dim=-1)
        comm_acc = float((pc == yc).float().mean().item())

    return {
        "mixed_syntax_acc": mixed_syntax_acc,
        "mixed_distant_kv_acc": mixed_distant_kv_acc,
        "unseen_distant_kv_acc": unseen_distant_kv_acc,
        "d4_id_10": d4_id_10,
        "d4_ood_30_noisy": d4_ood_30_noisy,
        "commutator_acc": comm_acc,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-seed0", action="store_true", help="Run seed 0 only for timing & sanity check without writing final artifact.")
    args = parser.parse_args()

    t0 = time.perf_counter()
    seeds = [0] if args.smoke_seed0 else [0, 1, 2]
    wt_root = Path(__file__).resolve().parents[2]

    ref_m = HybridSeqModel(vocab_size=44, d_model=48, n_heads=4, d_ff=96, mixer_kinds=["attn", "attn"], pointer_mode="none")
    ref_non_emb_mixed = count_trainable_params(ref_m) - ref_m.embed.weight.numel()

    ref_d = HybridSeqModel(vocab_size=16, d_model=48, n_heads=4, d_ff=96, mixer_kinds=["attn", "attn"], pointer_mode="none")
    ref_non_emb_d4 = count_trainable_params(ref_d) - ref_d.embed.weight.numel()

    arm_specs = [
        ("H0_Pure_CausalAttn", ["attn", "attn"], "none", 0.0),
        ("H1_Pure_HDCD", ["hdcd", "hdcd"], "none", 0.0),
        ("H2_HDCD_WriteShield", ["hdcd_writeshield", "hdcd_writeshield"], "none", 1e-3),
        ("H3_Hybrid_HDCD_Attn", ["hdcd", "attn"], "none", 0.0),
        ("H4_HDCD_StandardPtr", ["hdcd", "hdcd"], "standard", 0.0),
    ]
    arms = [a[0] for a in arm_specs]
    raw: dict[str, list[dict[str, float]]] = {a: [] for a in arms}
    arm_meta: dict[str, Any] = {}

    for arm_name, m_kinds, p_mode, _ in arm_specs:
        dff_m = _match_hybrid_d_ff(ref_non_emb_mixed, 44, 48, 4, m_kinds, p_mode)
        probe = HybridSeqModel(44, 48, 4, dff_m, m_kinds, p_mode)
        non_emb = count_trainable_params(probe) - probe.embed.weight.numel()
        arm_meta[arm_name] = {
            "trainable_params": count_trainable_params(probe),
            "non_embedding_params": non_emb,
            "non_emb_rel_diff_pct": round(100.0 * abs(non_emb - ref_non_emb_mixed) / ref_non_emb_mixed, 2),
            "d_ff_mixed": dff_m,
            "mixer_kinds": m_kinds,
            "pointer_mode": p_mode,
        }

    for seed in seeds:
        t_s = time.perf_counter()
        for arm_name, m_kinds, p_mode, bw_l1 in arm_specs:
            res = _train_eval_arm_seed(
                seed, arm_name, m_kinds, p_mode, bw_l1, ref_non_emb_mixed, ref_non_emb_d4
            )
            raw[arm_name].append(res)
        print(f"[HYBRID-INT] seed {seed} done in {time.perf_counter() - t_s:.2f}s", flush=True)

    m_keys = list(raw[arms[0]][0].keys())
    summary = {arm: {k: summarize_seeds([r[k] for r in rows]) for k in m_keys} for arm, rows in raw.items()}

    for arm in arms:
        s = summary[arm]
        print(
            f"[HYBRID-INT] {arm:22s} | Syn={s['mixed_syntax_acc']['mean']:.3f} DistKV={s['mixed_distant_kv_acc']['mean']:.3f} "
            f"UnseenKV={s['unseen_distant_kv_acc']['mean']:.3f} | "
            f"D4_10={s['d4_id_10']['mean']:.3f} D4_30Nz={s['d4_ood_30_noisy']['mean']:.3f} Comm={s['commutator_acc']['mean']:.3f}",
            flush=True,
        )

    if args.smoke_seed0:
        print(f"[SMOKE-SEED0] Total elapsed={time.perf_counter() - t0:.2f}s | ParamMeta={arm_meta}")
        return

    h0_dist = summary["H0_Pure_CausalAttn"]["mixed_distant_kv_acc"]["mean"]
    h1_dist = summary["H1_Pure_HDCD"]["mixed_distant_kv_acc"]["mean"]
    h2_dist = summary["H2_HDCD_WriteShield"]["mixed_distant_kv_acc"]["mean"]
    h3_dist = summary["H3_Hybrid_HDCD_Attn"]["mixed_distant_kv_acc"]["mean"]
    h4_unseen = summary["H4_HDCD_StandardPtr"]["unseen_distant_kv_acc"]["mean"]
    h3_d4 = summary["H3_Hybrid_HDCD_Attn"]["d4_ood_30_noisy"]["mean"]
    h0_d4 = summary["H0_Pure_CausalAttn"]["d4_ood_30_noisy"]["mean"]

    if h3_dist > h1_dist + 0.40 and h3_d4 > h0_d4 + 0.15:
        if h2_dist > h1_dist + 0.25:
            decision = "PROMOTE_HYBRID_HDCD_ATTN_AND_PARTIAL_WRITESHIELD"
        else:
            decision = "PROMOTE_HYBRID_HDCD_ATTN_FALSIFY_O1_WRITESHIELD"
    elif h2_dist > h1_dist + 0.40:
        decision = "PROMOTE_O1_HDCD_WRITESHIELD"
    else:
        decision = "CHECK_INTERFERENCE_TRADEOFFS"

    elapsed = round(time.perf_counter() - t0, 3)
    payload = {
        "schema": "nanolm.cycle2.experiment.v1",
        "experiment_id": "HYBRID-INTERFERENCE-01",
        "research_question": (
            "Can closed-init slow-guild write shielding + decay floor (H2_HDCD_WriteShield), per-layer "
            "HDCD+Attention hybridization (H3_Hybrid_HDCD_Attn), or an explicit copy pointer (H4_HDCD_StandardPtr) "
            "resolve Cycle 1's Mixed_FastSyntax_SlowMemory collapse while preserving non-Abelian D4 state tracking?"
        ),
        "hypothesis": (
            "1) Pure HDCD (H1) preserves D4 well but suffers fast-syntax interference into slow KV memory; "
            "2) Pure Attention (H0) tracks Mixed KV strongly without O(1) non-Abelian group state OOD generalization; "
            "3) Write-shielding (H2) partially/fully restores O(1) KV shielding; "
            "4) H3_Hybrid_HDCD_Attn achieves strong performance on BOTH Mixed_FastSyntax_SlowMemory and NonAbelian D4; "
            "5) Standard copy pointer (H4) lifts unseen open-vocab distant KV retrieval when backbone carries KV slots."
        ),
        "competing_explanation": (
            "1) O(1) write shielding may still struggle to coordinate key/value gating across random syntax tokens; "
            "2) Adding Layer-1 Attention in H3 might interfere with Layer-0 HDCD D4 tracking or fail unseen value tokens without a copy pointer."
        ),
        "code_revision": _git_sha(wt_root),
        "dataset_provenance": (
            "Extended Mixed_FastSyntax_SlowMemory (vocab=44: seen vals 10..21, held-out unseen vals 36..43, syntax_len=12) "
            "+ NonAbelian_D4_WordTrack (vocab=16: D4_10, D4_30 noisy, commutator probe) across seeds [0, 1, 2]"
        ),
        "arms": arm_meta,
        "metrics": m_keys,
        "success_criteria": "Iso-param (+-3%) comparison completes cleanly; H3 or H2 beats H1 on mixed_distant_kv_acc by >= +0.40 while keeping D4 OOD > H0",
        "failure_criteria": "H3 breaks D4 tracking or fails to recover mixed_distant_kv_acc, AND H2 fails to shield slow memory",
        "seeds": seeds,
        "raw_per_seed": raw,
        "results": {"summary": summary},
        "resource_consumption": {"paid_compute_usd": 0.0, "device": "cpu_float32", "wall_clock_seconds": elapsed},
        "independent_verification": {
            "h3_vs_h1_mixed_dist_kv_gain": round(h3_dist - h1_dist, 4),
            "h2_vs_h1_mixed_dist_kv_gain": round(h2_dist - h1_dist, 4),
            "h3_vs_h0_d4_ood30_gain": round(h3_d4 - h0_d4, 4),
            "h4_vs_h1_unseen_kv_gain": round(h4_unseen - summary["H1_Pure_HDCD"]["unseen_distant_kv_acc"]["mean"], 4),
            "h0_dist_kv_sanity": round(h0_dist, 4),
        },
        "final_decision": decision,
    }
    out = wt_root / "artifacts" / "cycle2" / "HYBRID-INTERFERENCE-01.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"HYBRID-INTERFERENCE-01 complete in {elapsed}s -> {out} | Decision: {decision}")


if __name__ == "__main__":
    main()
