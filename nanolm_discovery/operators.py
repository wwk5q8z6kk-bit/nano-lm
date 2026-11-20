"""Hardware-fast exact dual operators and synthesis blocks for NanoLM Discovery."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DiscoveryRMSNorm(nn.Module):
    """Precision-safe RMSNorm casting internal variance to float32 to prevent MPS warnings."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x_fp32 = x.float()
        normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.to(x.dtype) * self.weight.to(x.dtype)


def apply_rope(x: Tensor, seq_offset: int = 0, theta: float = 500_000.0) -> Tensor:
    """Apply rotary position embeddings to `[B, H, T, D]` tensor."""
    _, _, t_len, head_dim = x.shape
    pos = torch.arange(seq_offset, seq_offset + t_len, device=x.device, dtype=torch.float32)
    half = head_dim // 2
    freq_idx = torch.arange(half, device=x.device, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (2.0 * freq_idx / float(head_dim)))
    angles = torch.outer(pos, inv_freq)[None, None, :, :]
    cos = angles.cos().to(x.dtype)
    sin = angles.sin().to(x.dtype)
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1).flatten(-2)


def _no_autocast_ctx(device: torch.device):
    dev_type = device.type if device.type in ("mps", "cuda", "cpu") else "cpu"
    return torch.autocast(device_type=dev_type, enabled=False)


class CausalKeyShift(nn.Module):
    """Asymmetric 4-tap causal depthwise key-shift conv on addressing pathway (4*d params)."""

    def __init__(self, d_model: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, groups=d_model, bias=False)

    def forward(self, x: Tensor, ring: Tensor | None = None) -> tuple[Tensor, Tensor]:
        bsz, t_len, d_model = x.shape
        x_t = x.transpose(1, 2)  # [B, D, T]
        pad_len = self.kernel_size - 1
        if ring is None:
            ring_buf = torch.zeros(bsz, d_model, pad_len, device=x.device, dtype=x.dtype)
        else:
            ring_buf = ring.to(x.dtype)
        if t_len == 1:
            win = torch.cat([ring_buf, x_t], dim=-1)  # [B, D, 4]
            w = self.conv.weight.squeeze(1).to(x.dtype)
            out = F.silu((win * w.unsqueeze(0)).sum(dim=-1)).unsqueeze(1)
            next_ring = win[:, :, 1:].detach().clone()
            return out, next_ring
        sig_full = torch.cat([ring_buf, x_t], dim=-1)
        out = F.silu(self.conv(sig_full)).transpose(1, 2)
        next_ring = sig_full[:, :, -pad_len:].detach().clone()
        return out, next_ring


def _positive_feature_map(x: Tensor) -> Tensor:
    """Positive exponential-family feature map phi(x) = elu(x) + 1 with L2 stabilizer."""
    phi = F.elu(x.float()) + 1.0
    norm = phi.pow(2).sum(dim=-1, keepdim=True).add(1e-6).rsqrt() * (x.shape[-1] ** 0.5)
    return phi * norm


def wy_metriplectic_chunk_scan(
    w: Tensor,
    v: Tensor,
    q: Tensor,
    beta: Tensor,
    log_gamma: Tensor,
    init_state: Tensor | None = None,
    chunk_size: int = 32,
) -> tuple[Tensor, Tensor]:
    """Exact nilpotent WY solver for scalar-decayed rank-1 updates:

        S_m = exp(log_gamma_m) * (I - beta_m w_m w_m^T) S_{m-1} + beta_m w_m v_m^T
        y_m = S_m^T q_m

    Accepts `w, v, q: [B, H, M, D]`, `beta: [B, H, M, 1]`, `log_gamma: [B, H, M] <= 0`.
    All internal WY solves and state accumulations run in float32 with autocast disabled.
    """
    with _no_autocast_ctx(w.device):
        bsz, n_heads, m_len, head_dim = w.shape
        orig_dtype = q.dtype
        rem = m_len % chunk_size
        if rem != 0:
            pad = chunk_size - rem
            w = F.pad(w, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
            q = F.pad(q, (0, 0, 0, pad))
            beta = F.pad(beta, (0, 0, 0, pad))
            log_gamma = F.pad(log_gamma, (0, pad))
        m_padded = w.shape[2]
        n_chunks = m_padded // chunk_size

        w_f = w.float().view(bsz, n_heads, n_chunks, chunk_size, head_dim)
        v_f = v.float().view(bsz, n_heads, n_chunks, chunk_size, head_dim)
        q_f = q.float().view(bsz, n_heads, n_chunks, chunk_size, head_dim)
        b_f = beta.float().view(bsz, n_heads, n_chunks, chunk_size, 1)
        lg_f = log_gamma.float().clamp(max=0.0).view(bsz, n_heads, n_chunks, chunk_size)

        csum = torch.cumsum(lg_f, dim=-1)
        tril_0 = torch.tril(torch.ones(chunk_size, chunk_size, device=w.device, dtype=torch.bool), 0)
        tril_m1 = torch.tril(torch.ones(chunk_size, chunk_size, device=w.device, dtype=torch.bool), -1)
        g_mat = torch.exp((csum.unsqueeze(-1) - csum.unsqueeze(-2)).clamp(max=0.0)) * tril_0

        wb = w_f * b_f
        lower = -torch.matmul(wb, w_f.transpose(-1, -2)) * (g_mat * tril_m1)
        eye_c = torch.eye(chunk_size, device=w.device, dtype=torch.float32)
        inv = eye_c + lower
        l_pow = torch.matmul(lower, lower)
        n_sq = (chunk_size - 1).bit_length()
        for _ in range(max(0, n_sq - 1)):
            inv = torch.matmul(inv, eye_c + l_pow)
            l_pow = torch.matmul(l_pow, l_pow)

        g0 = torch.exp(csum).unsqueeze(-1)
        g_end = torch.exp((csum[..., -1:] - csum).clamp(max=0.0)).unsqueeze(-1)
        u_wy = torch.matmul(inv, b_f * v_f)
        w_wy = torch.matmul(inv, wb * g0)
        qw = torch.matmul(q_f, w_f.transpose(-1, -2)) * g_mat

        if init_state is None:
            state = torch.zeros(bsz, n_heads, head_dim, head_dim, device=w.device, dtype=torch.float32)
        else:
            state = init_state.float()

        outs: list[Tensor] = []
        for c_idx in range(n_chunks):
            z_chunk = u_wy[:, :, c_idx] - torch.matmul(w_wy[:, :, c_idx], state)
            y_chunk = torch.matmul(g0[:, :, c_idx] * q_f[:, :, c_idx], state) + torch.matmul(
                qw[:, :, c_idx], z_chunk
            )
            outs.append(y_chunk)
            g_tot = torch.exp(csum[:, :, c_idx, -1]).unsqueeze(-1).unsqueeze(-1)
            state = g_tot * state + torch.matmul(
                (w_f[:, :, c_idx] * g_end[:, :, c_idx]).transpose(-1, -2), z_chunk
            )

        out = torch.stack(outs, dim=2).view(bsz, n_heads, m_padded, head_dim)[:, :, :m_len, :]
        return out.to(orig_dtype), state


def nebn_dual_filter(
    q_raw: Tensor,
    k_raw: Tensor,
    v: Tensor,
    log_gamma: Tensor,
    tau: Tensor,
    prior_prec: Tensor,
    init_state: dict[str, Tensor] | None = None,
    chunk_size: int = 128,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Chunk-parallel exponential-family Bayesian natural-parameter filter.

    Exact O(T * chunk_size) dual to O(1)-state recurrent filtering `(H, P, ell)`.
    Returns `(mu, uncertainty, final_state)`.
    """
    with _no_autocast_ctx(q_raw.device):
        bsz, n_heads, t_len, head_dim = q_raw.shape
        orig_dtype = v.dtype
        phi_q = _positive_feature_map(q_raw)
        phi_k = _positive_feature_map(k_raw)
        k_tilde = phi_k * tau.float().unsqueeze(-1)
        v_f = v.float()
        lg_f = log_gamma.float().clamp(max=0.0)
        lam0 = prior_prec.float().view(1, n_heads, 1)

        if init_state is None:
            h_state = torch.zeros(bsz, n_heads, head_dim, head_dim, device=q_raw.device, dtype=torch.float32)
            p_state = torch.zeros(bsz, n_heads, head_dim, device=q_raw.device, dtype=torch.float32)
            ell_state = torch.zeros(bsz, n_heads, device=q_raw.device, dtype=torch.float32)
        else:
            h_state = init_state["h"].float()
            p_state = init_state["p"].float()
            ell_state = init_state["ell"].float()

        mu_chunks: list[Tensor] = []
        unc_chunks: list[Tensor] = []
        for start in range(0, t_len, chunk_size):
            end = min(t_len, start + chunk_size)
            c_len = end - start
            qc = phi_q[:, :, start:end]
            kc = k_tilde[:, :, start:end]
            vc = v_f[:, :, start:end]
            lgc = lg_f[:, :, start:end]

            csum = torch.cumsum(lgc, dim=-1)
            tril = torch.tril(torch.ones(c_len, c_len, device=q_raw.device, dtype=torch.bool))
            g_mat = torch.exp((csum.unsqueeze(-1) - csum.unsqueeze(-2)).clamp(max=0.0)) * tril
            scores = torch.matmul(qc, kc.transpose(-1, -2)) * g_mat

            num_intra = torch.matmul(scores, vc)
            den_intra = scores.sum(dim=-1)

            alpha = torch.exp(csum)
            num_carry = alpha.unsqueeze(-1) * torch.matmul(qc, h_state)
            den_carry = alpha * torch.einsum("bhcd,bhd->bhc", qc, p_state)
            prior_term = lam0 * torch.exp((ell_state.unsqueeze(-1) + csum).clamp(max=0.0))

            prec = den_intra + den_carry + prior_term + 1e-4
            mu_c = (num_intra + num_carry) / prec.unsqueeze(-1)
            unc_c = 1.0 / prec
            mu_chunks.append(mu_c)
            unc_chunks.append(unc_c)

            alpha_end = torch.exp(csum[:, :, -1])
            d_end = torch.exp((csum[:, :, -1:] - csum).clamp(max=0.0)).unsqueeze(-1)
            k_weighted = kc * d_end
            h_state = alpha_end.unsqueeze(-1).unsqueeze(-1) * h_state + torch.matmul(
                k_weighted.transpose(-1, -2), vc
            )
            p_state = alpha_end.unsqueeze(-1) * p_state + k_weighted.sum(dim=2)
            ell_state = ell_state + csum[:, :, -1]

        mu = torch.cat(mu_chunks, dim=2).to(orig_dtype)
        unc = torch.cat(unc_chunks, dim=2).to(orig_dtype)
        return mu, unc, {"h": h_state, "p": p_state, "ell": ell_state}


def nebn_precision_trace(
    q_raw: Tensor,
    k_raw: Tensor,
    log_gamma: Tensor,
    tau: Tensor,
    prior_prec: Tensor,
    init_state: dict[str, Tensor] | None = None,
    chunk_size: int = 128,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Lightweight O(T * chunk_size) positive precision tracker returning scalar uncertainty u_t."""
    with _no_autocast_ctx(q_raw.device):
        bsz, n_heads, t_len, head_dim = q_raw.shape
        phi_q = _positive_feature_map(q_raw)
        phi_k = _positive_feature_map(k_raw)
        k_tilde = phi_k * tau.float().unsqueeze(-1)
        lg_f = log_gamma.float().clamp(max=0.0)
        lam0 = prior_prec.float().view(1, n_heads, 1)

        if init_state is None:
            p_state = torch.zeros(bsz, n_heads, head_dim, device=q_raw.device, dtype=torch.float32)
            ell_state = torch.zeros(bsz, n_heads, device=q_raw.device, dtype=torch.float32)
        else:
            p_state = init_state["p"].float()
            ell_state = init_state["ell"].float()

        unc_chunks: list[Tensor] = []
        for start in range(0, t_len, chunk_size):
            end = min(t_len, start + chunk_size)
            c_len = end - start
            qc = phi_q[:, :, start:end]
            kc = k_tilde[:, :, start:end]
            lgc = lg_f[:, :, start:end]

            csum = torch.cumsum(lgc, dim=-1)
            tril = torch.tril(torch.ones(c_len, c_len, device=q_raw.device, dtype=torch.bool))
            g_mat = torch.exp((csum.unsqueeze(-1) - csum.unsqueeze(-2)).clamp(max=0.0)) * tril
            scores = torch.matmul(qc, kc.transpose(-1, -2)) * g_mat
            den_intra = scores.sum(dim=-1)
            alpha = torch.exp(csum)
            den_carry = alpha * torch.einsum("bhcd,bhd->bhc", qc, p_state)
            prior_term = lam0 * torch.exp((ell_state.unsqueeze(-1) + csum).clamp(max=0.0))
            prec = den_intra + den_carry + prior_term + 1e-4
            unc_chunks.append(1.0 / prec)

            alpha_end = torch.exp(csum[:, :, -1])
            d_end = torch.exp((csum[:, :, -1:] - csum).clamp(max=0.0)).unsqueeze(-1)
            p_state = alpha_end.unsqueeze(-1) * p_state + (kc * d_end).sum(dim=2)
            ell_state = ell_state + csum[:, :, -1]

        return torch.cat(unc_chunks, dim=2), {"p": p_state, "ell": ell_state}


def scam_causal_topk_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    score: Tensor,
    k_slots: int = 32,
    chunk_size: int = 256,
    mem_state: dict[str, Tensor] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Fast strictly causal surprise-addressed fixed-capacity cache lookup.

    Exact match to token-by-token min-score eviction `forward_step()` with zero future leakage.
    """
    bsz, t_len, d_scam = q.shape
    neg_inf = -1e9
    scale = d_scam ** -0.5

    if mem_state is None:
        mem_k = torch.zeros(bsz, k_slots, d_scam, device=q.device, dtype=q.dtype)
        mem_v = torch.zeros(bsz, k_slots, d_scam, device=q.device, dtype=q.dtype)
        mem_s = torch.full((bsz, k_slots), neg_inf, device=q.device, dtype=torch.float32)
        step_offset = 0
    else:
        mem_k = mem_state["k"]
        mem_v = mem_state["v"]
        mem_s = mem_state["s"].float()
        step_offset = int(mem_state["step"])

    if k_slots == 0:
        return torch.zeros_like(q), {"k": mem_k, "v": mem_v, "s": mem_s, "step": step_offset + t_len}

    outs: list[Tensor] = []
    for start in range(0, t_len, chunk_size):
        end = min(t_len, start + chunk_size)
        c_len = end - start
        qc = q[:, start:end]
        kc = k[:, start:end]
        vc = v[:, start:end]
        pos_idx = torch.arange(step_offset + start, step_offset + end, device=q.device, dtype=torch.float32)
        sc = score[:, start:end].float() + pos_idx.unsqueeze(0) * 1e-6

        if step_offset == 0 and start == 0:
            # First chunk optimization: no prior slots needed
            n_cand = c_len
            k_eff = min(k_slots, c_len)
            cand_k, cand_v, cand_s = kc, vc, sc
            causal = torch.tril(torch.ones(c_len, c_len, device=q.device, dtype=torch.bool))
        else:
            n_cand = k_slots + c_len
            k_eff = k_slots
            cand_k = torch.cat([mem_k, kc], dim=1)
            cand_v = torch.cat([mem_v, vc], dim=1)
            cand_s = torch.cat([mem_s, sc], dim=1)
            causal = torch.cat(
                [
                    torch.ones(c_len, k_slots, device=q.device, dtype=torch.bool),
                    torch.tril(torch.ones(c_len, c_len, device=q.device, dtype=torch.bool)),
                ],
                dim=1,
            )

        masked_s = cand_s.unsqueeze(1).expand(bsz, c_len, n_cand).masked_fill(~causal.unsqueeze(0), neg_inf)
        top_s, top_idx = torch.topk(masked_s, k=k_eff, dim=-1)

        b_off = (torch.arange(bsz, device=q.device) * n_cand).view(bsz, 1, 1)
        flat_idx = (b_off + top_idx).reshape(-1)
        sel_k = cand_k.reshape(bsz * n_cand, d_scam)[flat_idx].view(bsz, c_len, k_eff, d_scam)
        sel_v = cand_v.reshape(bsz * n_cand, d_scam)[flat_idx].view(bsz, c_len, k_eff, d_scam)

        logits = (
            torch.matmul(qc.unsqueeze(-2), sel_k.transpose(-1, -2)).squeeze(-2).float() * scale
            + 0.1 * top_s.clamp(min=-20.0, max=20.0)
        )
        logits = logits.masked_fill(top_s <= -1e8, -1e9)
        attn = torch.softmax(logits, dim=-1).to(qc.dtype)
        outs.append(torch.matmul(attn.unsqueeze(-2), sel_v).squeeze(-2))

        # Update persistent cache at end of chunk
        if k_eff < k_slots:
            mem_k[:, :k_eff] = sel_k[:, -1]
            mem_v[:, :k_eff] = sel_v[:, -1]
            mem_s[:, :k_eff] = top_s[:, -1]
        else:
            mem_k = sel_k[:, -1]
            mem_v = sel_v[:, -1]
            mem_s = top_s[:, -1]

    return torch.cat(outs, dim=1), {"k": mem_k, "v": mem_v, "s": mem_s, "step": step_offset + t_len}


def scam_step_attention(
    q_t: Tensor,
    k_t: Tensor,
    v_t: Tensor,
    s_t: Tensor,
    mem_state: dict[str, Any] | None,
    k_slots: int,
) -> tuple[Tensor, dict[str, Any]]:
    """Exact single-token recurrent dual to `scam_causal_topk_attention`."""
    bsz, d_scam = q_t.shape
    neg_inf = -1e9
    if mem_state is None:
        mem_k = torch.zeros(bsz, k_slots, d_scam, device=q_t.device, dtype=q_t.dtype)
        mem_v = torch.zeros(bsz, k_slots, d_scam, device=q_t.device, dtype=q_t.dtype)
        mem_s = torch.full((bsz, k_slots), neg_inf, device=q_t.device, dtype=torch.float32)
        step = 0
    else:
        mem_k = mem_state["k"].clone()
        mem_v = mem_state["v"].clone()
        mem_s = mem_state["s"].float().clone()
        step = int(mem_state["step"])

    if k_slots == 0:
        return torch.zeros_like(q_t), {"k": mem_k, "v": mem_v, "s": mem_s, "step": step + 1}

    score_adj = s_t.float() + step * 1e-6
    min_s, min_idx = torch.min(mem_s, dim=1)
    replace_mask = score_adj > min_s
    for b_idx in range(bsz):
        if bool(replace_mask[b_idx]):
            slot = int(min_idx[b_idx].item())
            mem_s[b_idx, slot] = score_adj[b_idx]
            mem_k[b_idx, slot] = k_t[b_idx]
            mem_v[b_idx, slot] = v_t[b_idx]

    logits = torch.einsum("bd,bkd->bk", q_t, mem_k).float() * (d_scam ** -0.5) + 0.1 * mem_s.clamp(
        min=-20.0, max=20.0
    )
    logits = logits.masked_fill(mem_s <= -1e8, -1e9)
    attn = torch.softmax(logits, dim=-1).to(q_t.dtype)
    out = torch.einsum("bk,bkd->bd", attn, mem_v)
    return out, {"k": mem_k, "v": mem_v, "s": mem_s, "step": step + 1}


# ---------------------------------------------------------------------------
# Sequence Operators
# ---------------------------------------------------------------------------


class GQAAttentionOperator(nn.Module):
    """Standard Grouped-Query Attention with causal RoPE (Baseline B0 / B1 global)."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int, theta: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.theta = theta
        kv_dim = n_kv_heads * head_dim
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, state: dict[str, Tensor] | None = None) -> tuple[Tensor, Tensor | None, dict[str, Tensor]]:
        bsz, t_len, d_model = x.shape
        q = self.q_proj(x).view(bsz, t_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, t_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, t_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        offset = int(state["k"].shape[2]) if state is not None and "k" in state else 0
        q = apply_rope(q, seq_offset=offset, theta=self.theta)
        k = apply_rope(k, seq_offset=offset, theta=self.theta)
        if state is not None and "k" in state:
            k_full = torch.cat([state["k"], k], dim=2)
            v_full = torch.cat([state["v"], v], dim=2)
        else:
            k_full, v_full = k, v
        rep = self.n_heads // self.n_kv_heads
        k_exp = k_full.repeat_interleave(rep, dim=1) if rep > 1 else k_full
        v_exp = v_full.repeat_interleave(rep, dim=1) if rep > 1 else v_full
        is_causal = state is None
        attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=is_causal)
        if not is_causal and t_len > 1:
            raise ValueError("Incremental KV cache mode expects single-step queries when t_len > 1 without mask")
        out = self.o_proj(attn_out.transpose(1, 2).contiguous().view(bsz, t_len, d_model))
        return out, None, {"k": k_full, "v": v_full}


class GDN2Operator(nn.Module):
    """Modern Gated DeltaNet-2 baseline operator via exact WY chunk scan + asymmetric key-shift."""

    def __init__(self, d_model: int, n_heads: int, head_dim: int, chunk_size: int = 32) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.key_shift = CausalKeyShift(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, 2 * n_heads, bias=True)

    def forward(self, x: Tensor, state: dict[str, Any] | None = None) -> tuple[Tensor, Tensor | None, dict[str, Any]]:
        bsz, t_len, d_model = x.shape
        x_addr, next_shift = self.key_shift(x, state.get("shift") if state is not None else None)
        q = self.q_proj(x_addr).view(bsz, t_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = (
            F.normalize(self.k_proj(x_addr).view(bsz, t_len, self.n_heads, self.head_dim).float(), dim=-1, eps=1e-6)
            .transpose(1, 2)
            .to(x.dtype)
        )
        v = self.v_proj(x).view(bsz, t_len, self.n_heads, self.head_dim).transpose(1, 2)
        gates = self.gate_proj(x).view(bsz, t_len, self.n_heads, 2).permute(0, 2, 1, 3)
        log_gamma = -F.softplus(gates[..., 0].float()) * 0.25
        beta = torch.sigmoid(gates[..., 1].float()).unsqueeze(-1)
        init_s = state["s"] if state is not None else None
        if t_len == 1 and state is not None:
            with _no_autocast_ctx(x.device):
                s_prev = init_s.float()
                g = torch.exp(log_gamma[:, :, 0]).unsqueeze(-1).unsqueeze(-1)
                wt = k[:, :, 0].float()
                vt = v[:, :, 0].float()
                qt = q[:, :, 0].float()
                bt = beta[:, :, 0].unsqueeze(-1)
                s_g = g * s_prev
                ws = torch.einsum("bhd,bhde->bhe", wt, s_g)
                s_next = s_g + bt * torch.einsum("bhd,bhe->bhde", wt, vt - ws)
                y = torch.einsum("bhd,bhde->bhe", qt, s_next).unsqueeze(2).to(x.dtype)
        else:
            y, s_next = wy_metriplectic_chunk_scan(
                k, v, q, beta.to(x.dtype), log_gamma, init_state=init_s, chunk_size=self.chunk_size
            )
        out = self.o_proj(y.transpose(1, 2).contiguous().view(bsz, t_len, d_model))
        return out, None, {"s": s_next, "shift": next_shift}


class PortHamiltonianOperator(nn.Module):
    """Invention 1 (PHMN): Conservative Householder group transport + dissipative precision port.

    Update: S_t = gamma_t * (I - tau_t k_t k_t^T)(I - beta_t u_t u_t^T) S_{t-1} + tau_t k_t v_t^T
    with ||u_t||=||k_t||=1, beta_t in (0, 2), tau_t in (0, 1), gamma_t in (0, 1].
    Homogeneous transition spectral norm satisfies ||A_t||_2 <= gamma_t <= 1 unconditionally.
    """

    def __init__(self, d_model: int, n_heads: int, head_dim: int, chunk_size: int = 32) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.key_shift = CausalKeyShift(d_model)
        self.u_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, 3 * n_heads, bias=True)

    def _project_terms(
        self, x: Tensor, x_addr: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        bsz, t_len, _ = x.shape
        addr = x if x_addr is None else x_addr
        u = F.normalize(self.u_proj(addr).view(bsz, t_len, self.n_heads, self.head_dim).float(), dim=-1, eps=1e-6).transpose(1, 2)
        k = F.normalize(self.k_proj(addr).view(bsz, t_len, self.n_heads, self.head_dim).float(), dim=-1, eps=1e-6).transpose(1, 2)
        v = self.v_proj(x).view(bsz, t_len, self.n_heads, self.head_dim).transpose(1, 2).float()
        q = self.q_proj(addr).view(bsz, t_len, self.n_heads, self.head_dim).transpose(1, 2).float()
        gates = self.gate_proj(x).view(bsz, t_len, self.n_heads, 3).permute(0, 2, 1, 3).float()
        beta = 1.98 * torch.sigmoid(gates[..., 0])  # spans (0, 1.98), including reflections > 1
        tau = torch.sigmoid(gates[..., 1])          # (0, 1) dissipative learning rate
        log_gamma = -F.softplus(gates[..., 2]) * 0.20
        return u, k, v, q, beta, tau, log_gamma

    def forward(self, x: Tensor, state: dict[str, Any] | None = None) -> tuple[Tensor, Tensor | None, dict[str, Any]]:
        bsz, t_len, d_model = x.shape
        x_addr, next_shift = self.key_shift(x, state.get("shift") if state is not None else None)
        u, k, v, q, beta, tau, log_gamma = self._project_terms(x, x_addr=x_addr)
        init_s = state["s"] if state is not None else None
        y, s_next = _run_symplectic_wy_scan(u, k, v, q, beta, tau, log_gamma, init_s, self.chunk_size)
        out = self.o_proj(y.to(x.dtype).transpose(1, 2).contiguous().view(bsz, t_len, d_model))
        return out, None, {"s": s_next, "shift": next_shift}


class NEBNOperator(nn.Module):
    """Invention 2 (NEBN): Exponential-family natural-parameter filter with epistemic uncertainty."""

    def __init__(self, d_model: int, n_heads: int, head_dim: int, chunk_size: int = 128) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.key_shift = CausalKeyShift(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, 2 * n_heads, bias=True)
        self.prior_log_prec = nn.Parameter(torch.zeros(n_heads))

    def forward(self, x: Tensor, state: dict[str, Any] | None = None) -> tuple[Tensor, Tensor, dict[str, Any]]:
        bsz, t_len, d_model = x.shape
        x_addr, next_shift = self.key_shift(x, state.get("shift") if state is not None else None)
        q = self.q_proj(x_addr).view(bsz, t_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_addr).view(bsz, t_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, t_len, self.n_heads, self.head_dim).transpose(1, 2)
        gates = self.gate_proj(x).view(bsz, t_len, self.n_heads, 2).permute(0, 2, 1, 3).float()
        log_gamma = -F.softplus(gates[..., 0]) * 0.25
        tau = F.softplus(gates[..., 1])
        prior_prec = F.softplus(self.prior_log_prec.float()) + 1e-2

        if t_len == 1:
            with _no_autocast_ctx(x.device):
                phi_q = _positive_feature_map(q)[:, :, 0]
                phi_k = _positive_feature_map(k)[:, :, 0]
                k_tilde = phi_k * tau[:, :, 0].unsqueeze(-1)
                vt = v[:, :, 0].float()
                g = torch.exp(log_gamma[:, :, 0])
                if state is None:
                    h_prev = torch.zeros(bsz, self.n_heads, self.head_dim, self.head_dim, device=x.device, dtype=torch.float32)
                    p_prev = torch.zeros(bsz, self.n_heads, self.head_dim, device=x.device, dtype=torch.float32)
                    ell_prev = torch.zeros(bsz, self.n_heads, device=x.device, dtype=torch.float32)
                else:
                    h_prev = state["h"].float()
                    p_prev = state["p"].float()
                    ell_prev = state["ell"].float()
                h_next = g.unsqueeze(-1).unsqueeze(-1) * h_prev + torch.einsum("bhd,bhe->bhde", k_tilde, vt)
                p_next = g.unsqueeze(-1) * p_prev + k_tilde
                ell_next = ell_prev + log_gamma[:, :, 0]
                prec = torch.einsum("bhd,bhd->bh", phi_q, p_next) + prior_prec.view(1, -1) * torch.exp(ell_next.clamp(max=0.0)) + 1e-4
                mu = (torch.einsum("bhd,bhde->bhe", phi_q, h_next) / prec.unsqueeze(-1)).unsqueeze(2).to(x.dtype)
                unc = (1.0 / prec).unsqueeze(2).to(x.dtype)
                next_state: dict[str, Any] = {"h": h_next, "p": p_next, "ell": ell_next, "shift": next_shift}
        else:
            mu, unc, filter_state = nebn_dual_filter(
                q, k, v, log_gamma, tau, prior_prec, init_state=state, chunk_size=self.chunk_size
            )
            next_state = {**filter_state, "shift": next_shift}
        out = self.o_proj(mu.transpose(1, 2).contiguous().view(bsz, t_len, d_model))
        return out, unc.transpose(1, 2), next_state  # unc: [B, T, H]


def _run_symplectic_wy_scan(
    u: Tensor | None,
    k: Tensor,
    v: Tensor,
    q: Tensor,
    beta: Tensor | None,
    tau: Tensor,
    log_gamma: Tensor,
    init_s: Tensor | None,
    chunk_size: int,
) -> tuple[Tensor, Tensor]:
    """Run exact scalar-decayed rank-1 Householder reflection + dissipative write scan."""
    bsz, n_heads, t_len, head_dim = k.shape
    if t_len == 1:
        with _no_autocast_ctx(k.device):
            s_cur = (
                torch.zeros(bsz, n_heads, head_dim, head_dim, device=k.device, dtype=torch.float32)
                if init_s is None
                else init_s.float()
            )
            if u is not None and beta is not None:
                ut = u[:, :, 0].float()
                bt = beta[:, :, 0].float().unsqueeze(-1).unsqueeze(-1)
                s_rot = s_cur - bt * torch.einsum("bhd,bhe->bhde", ut, torch.einsum("bhd,bhde->bhe", ut, s_cur))
            else:
                s_rot = s_cur
            gt = torch.exp(log_gamma[:, :, 0].float()).unsqueeze(-1).unsqueeze(-1)
            kt = k[:, :, 0].float()
            vt = v[:, :, 0].float()
            qt = q[:, :, 0].float()
            tt = tau[:, :, 0].float().unsqueeze(-1).unsqueeze(-1)
            s_g = gt * s_rot
            s_next = s_g + tt * torch.einsum("bhd,bhe->bhde", kt, vt - torch.einsum("bhd,bhde->bhe", kt, s_g))
            y_met = torch.einsum("bhd,bhde->bhe", qt, s_next).unsqueeze(2)
        return y_met, s_next

    if u is not None and beta is not None:
        w_2 = torch.stack([u, k], dim=3).view(bsz, n_heads, 2 * t_len, head_dim)
        v_2 = torch.stack([torch.zeros_like(v), v], dim=3).view(bsz, n_heads, 2 * t_len, head_dim)
        q_2 = torch.stack([torch.zeros_like(q), q], dim=3).view(bsz, n_heads, 2 * t_len, head_dim)
        b_2 = torch.stack([beta, tau], dim=3).view(bsz, n_heads, 2 * t_len, 1)
        lg_2 = torch.stack([torch.zeros_like(log_gamma), log_gamma], dim=3).view(bsz, n_heads, 2 * t_len)
        y_2, s_next = wy_metriplectic_chunk_scan(w_2, v_2, q_2, b_2, lg_2, init_state=init_s, chunk_size=chunk_size)
        return y_2[:, :, 1::2, :], s_next
    return wy_metriplectic_chunk_scan(k, v, q, tau.unsqueeze(-1), log_gamma, init_state=init_s, chunk_size=chunk_size)


def _apply_dilated_wave_bank(
    sig: Tensor,
    state_ring: Tensor | None,
    dilations: tuple[int, ...],
    bands: nn.ModuleList,
    kernel_size: int,
) -> tuple[Tensor, Tensor]:
    """Apply causal multi-dilation depthwise wave convolutions and maintain fixed O(1) ring buffer."""
    bsz, d_model, t_len = sig.shape
    max_hist = max(d * (kernel_size - 1) for d in dilations)
    sig_full = torch.cat([state_ring.to(sig.dtype), sig], dim=-1) if state_ring is not None else sig
    cur_len = sig_full.shape[-1]
    acc = torch.zeros_like(sig)
    if t_len == 1:
        for dil, conv in zip(dilations, bands):
            w = conv.weight.squeeze(1).to(sig.dtype)
            band = torch.zeros(bsz, d_model, device=sig.device, dtype=sig.dtype)
            for k_idx in range(kernel_size):
                src = cur_len - 1 - (kernel_size - 1 - k_idx) * dil
                if src >= 0:
                    band = band + sig_full[:, :, src] * w[:, k_idx]
            acc[:, :, 0] = acc[:, :, 0] + band
    else:
        for dil, conv in zip(dilations, bands):
            pad_left = dil * (kernel_size - 1)
            need = t_len + pad_left
            seg = sig_full[:, :, -need:] if cur_len >= need else F.pad(sig_full, (need - cur_len, 0))
            acc = acc + conv(seg)
    next_ring = sig_full[:, :, -max_hist:].detach().clone()
    return acc / float(len(dilations)), next_ring


class HRWNOperator(nn.Module):
    """Invention 3 (HRWN): Multi-scale causal dyadic wave field + SCAM episodic registers."""

    DILATIONS = (1, 4, 16, 64)
    KERNEL_SIZE = 5

    def __init__(self, d_model: int, scam_slots: int = 32) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_scam = d_model // 2
        self.scam_slots = scam_slots
        self.wave_in = nn.Linear(d_model, 2 * d_model, bias=False)
        self.wave_phase = nn.Linear(d_model, 2 * d_model, bias=False)
        self.wave_bands = nn.ModuleList(
            nn.Conv1d(d_model, d_model, kernel_size=self.KERNEL_SIZE, groups=d_model, dilation=d, bias=False)
            for d in self.DILATIONS
        )
        self.scam_q = nn.Linear(d_model, self.d_scam, bias=False)
        self.scam_k = nn.Linear(d_model, self.d_scam, bias=False)
        self.scam_v = nn.Linear(d_model, self.d_scam, bias=False)
        self.scam_o = nn.Linear(self.d_scam, d_model, bias=False)
        self.surp_gate = nn.Linear(d_model, 1, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, state: dict[str, Any] | None = None) -> tuple[Tensor, Tensor, dict[str, Any]]:
        bsz, t_len, d_model = x.shape
        re_in, im_in = self.wave_in(x).chunk(2, dim=-1)
        re_ph, im_ph = self.wave_phase(x).chunk(2, dim=-1)
        sig_ring = state.get("wave_ring") if state is not None else None
        wave_re, next_ring = _apply_dilated_wave_bank(
            re_in.transpose(1, 2), sig_ring, self.DILATIONS, self.wave_bands, self.KERNEL_SIZE
        )
        wave_re = wave_re.transpose(1, 2)
        prev_env = state.get("prev_wave") if state is not None else None
        if prev_env is None:
            lagged = F.pad(wave_re[:, :-1, :], (0, 0, 1, 0))
        else:
            lagged = torch.cat([prev_env, wave_re[:, :-1, :]], dim=1) if t_len > 1 else prev_env
        pred_err = (re_in - lagged).float().norm(dim=-1) / re_in.size(-1) ** 0.5
        surp_score = pred_err + F.softplus(self.surp_gate(x).squeeze(-1).float())

        wave_out = wave_re * torch.tanh(re_ph) + im_in * torch.tanh(im_ph)
        q_s = self.scam_q(x)
        k_s = self.scam_k(x)
        v_s = self.scam_v(x)
        scam_state = state.get("scam") if state is not None else None
        if t_len == 1:
            scam_y, next_scam = scam_step_attention(
                q_s[:, 0], k_s[:, 0], v_s[:, 0], surp_score[:, 0], scam_state, self.scam_slots
            )
            scam_y = scam_y.unsqueeze(1)
        else:
            scam_y, next_scam = scam_causal_topk_attention(
                q_s, k_s, v_s, surp_score, k_slots=self.scam_slots, mem_state=scam_state
            )

        out = self.o_proj(wave_out + self.scam_o(scam_y))
        unc_proxy = (1.0 / (1.0 + surp_score)).unsqueeze(-1).to(x.dtype)
        next_state = {
            "wave_ring": next_ring,
            "prev_wave": wave_re[:, -1:, :].detach().clone(),
            "scam": next_scam,
        }
        return out, unc_proxy, next_state


class HMWROperator(nn.Module):
    """Cycle 3 Unified Operator (G3_HMWR): Holographic Metriplectic Wave-Resonance + Shifted SCAM."""

    DILATIONS = (1, 4, 16, 64)
    KERNEL_SIZE = 5

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        chunk_size: int = 32,
        scam_slots: int = 32,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = max(1, n_heads // 2)
        self.head_dim = head_dim
        self.d_met = self.n_heads * head_dim
        self.d_scam = max(2, d_model // 4)
        self.chunk_size = chunk_size
        self.scam_slots = scam_slots

        self.key_shift = CausalKeyShift(d_model)
        self.wave_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.wave_bands = nn.ModuleList(
            nn.Conv1d(d_model, d_model, kernel_size=self.KERNEL_SIZE, groups=d_model, dilation=d, bias=False)
            for d in self.DILATIONS
        )
        self.u_proj = nn.Linear(d_model, self.d_met, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_met, bias=False)
        self.v_proj = nn.Linear(d_model, self.d_met, bias=False)
        self.q_proj = nn.Linear(d_model, self.d_met, bias=False)
        self.met_o = nn.Linear(self.d_met, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, 3 * self.n_heads, bias=True)

        self.scam_q = nn.Linear(d_model, self.d_scam, bias=False)
        self.scam_k = nn.Linear(d_model, self.d_scam, bias=False)
        self.scam_v = nn.Linear(d_model, self.d_scam, bias=False)
        self.scam_o = nn.Linear(self.d_scam, d_model, bias=False)
        self.surp_gate = nn.Linear(d_model, 1, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def _project_terms(
        self, x: Tensor, x_addr: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        bsz, t_len, _ = x.shape
        addr = x if x_addr is None else x_addr
        u = F.normalize(self.u_proj(addr).view(bsz, t_len, self.n_heads, self.head_dim).float(), dim=-1, eps=1e-6).transpose(1, 2)
        k = F.normalize(self.k_proj(addr).view(bsz, t_len, self.n_heads, self.head_dim).float(), dim=-1, eps=1e-6).transpose(1, 2)
        v = self.v_proj(x).view(bsz, t_len, self.n_heads, self.head_dim).transpose(1, 2).float()
        q = self.q_proj(addr).view(bsz, t_len, self.n_heads, self.head_dim).transpose(1, 2).float()
        gates = self.gate_proj(x).view(bsz, t_len, self.n_heads, 3).permute(0, 2, 1, 3).float()
        beta = 1.98 * torch.sigmoid(gates[..., 0])
        tau = torch.sigmoid(gates[..., 1])
        log_gamma = -F.softplus(gates[..., 2]) * 0.20
        return u, k, v, q, beta, tau, log_gamma

    def forward(self, x: Tensor, state: dict[str, Any] | None = None) -> tuple[Tensor, Tensor, dict[str, Any]]:
        bsz, t_len, _ = x.shape
        x_addr, next_shift = self.key_shift(x, state.get("shift") if state is not None else None)

        # 1. Causal Multi-Scale Dyadic Wave Field + predictive surprise
        re_in, re_ph = self.wave_proj(x).chunk(2, dim=-1)
        sig_ring = state.get("wave_ring") if state is not None else None
        wave_re, next_ring = _apply_dilated_wave_bank(
            re_in.transpose(1, 2), sig_ring, self.DILATIONS, self.wave_bands, self.KERNEL_SIZE
        )
        wave_re = wave_re.transpose(1, 2)
        prev_env = state.get("prev_wave") if state is not None else None
        if prev_env is None:
            lagged = F.pad(wave_re[:, :-1, :], (0, 0, 1, 0))
        else:
            lagged = torch.cat([prev_env, wave_re[:, :-1, :]], dim=1) if t_len > 1 else prev_env
        pred_err = (re_in - lagged).float().norm(dim=-1) / re_in.size(-1) ** 0.5
        surp_score = pred_err + F.softplus(self.surp_gate(x).squeeze(-1).float())
        wave_out = wave_re * torch.tanh(re_ph)

        # 2. Compact Half-Subspace Metriplectic WY Group Transport
        u, k, v, q, beta, tau, log_gamma = self._project_terms(x, x_addr=x_addr)
        y_met, s_next = _run_symplectic_wy_scan(
            u, k, v, q, beta, tau, log_gamma, state.get("s") if state is not None else None, self.chunk_size
        )
        met_out = self.met_o(y_met.to(x.dtype).transpose(1, 2).contiguous().view(bsz, t_len, self.d_met))

        # 3. Asymmetric-Shifted SCAM Episodic Register Bank
        q_s = self.scam_q(x_addr)
        k_s = self.scam_k(x_addr)
        v_s = self.scam_v(x)
        scam_state = state.get("scam") if state is not None else None
        if t_len == 1:
            scam_y, next_scam = scam_step_attention(
                q_s[:, 0], k_s[:, 0], v_s[:, 0], surp_score[:, 0], scam_state, self.scam_slots
            )
            scam_y = scam_y.unsqueeze(1)
        else:
            scam_y, next_scam = scam_causal_topk_attention(
                q_s, k_s, v_s, surp_score, k_slots=self.scam_slots, mem_state=scam_state
            )
        scam_out = self.scam_o(scam_y)

        # 4. Non-inverted causal epistemic uncertainty: surprise relative to retrieved memory energy
        retr_rms = (y_met.pow(2).mean(dim=(1, 3)) + scam_y.float().pow(2).mean(dim=-1) + 1e-3).sqrt()
        unc_proxy = (surp_score / retr_rms).unsqueeze(-1).to(x.dtype)

        out = self.o_proj(wave_out + met_out + scam_out)
        next_state = {
            "s": s_next,
            "shift": next_shift,
            "wave_ring": next_ring,
            "prev_wave": wave_re[:, -1:, :].detach().clone(),
            "scam": next_scam,
        }
        return out, unc_proxy, next_state


class MPRNOperator(nn.Module):
    """Cycle 2 Unified Operator: Metriplectic WY Transport + Natural Precision + Epistemic SCAM."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        chunk_size: int = 32,
        scam_slots: int = 32,
        *,
        use_symplectic_reflection: bool = True,
        use_precision_norm: bool = True,
        use_scam_memory: bool = True,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.scam_slots = scam_slots
        self.use_symplectic_reflection = use_symplectic_reflection
        self.use_precision_norm = use_precision_norm
        self.use_scam_memory = use_scam_memory and scam_slots > 0

        self.key_shift = CausalKeyShift(d_model)
        if self.use_symplectic_reflection:
            self.u_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        n_gates = 3 if self.use_symplectic_reflection else 2
        self.gate_proj = nn.Linear(d_model, n_gates * n_heads, bias=True)
        if self.use_precision_norm:
            self.prior_log_prec = nn.Parameter(torch.zeros(n_heads))
        if self.use_scam_memory:
            d_scam = d_model // 2
            self.scam_q = nn.Linear(d_model, d_scam, bias=False)
            self.scam_k = nn.Linear(d_model, d_scam, bias=False)
            self.scam_v = nn.Linear(d_model, d_scam, bias=False)
            self.scam_o = nn.Linear(d_scam, d_model, bias=False)
            self.scam_alpha = nn.Linear(d_model, 1, bias=True)

    def forward(self, x: Tensor, state: dict[str, Any] | None = None) -> tuple[Tensor, Tensor | None, dict[str, Any]]:
        bsz, t_len, d_model = x.shape
        x_addr, next_shift = self.key_shift(x, state.get("shift") if state is not None else None)
        k_raw = self.k_proj(x_addr).view(bsz, t_len, self.n_heads, self.head_dim).transpose(1, 2)
        q_raw = self.q_proj(x_addr).view(bsz, t_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, t_len, self.n_heads, self.head_dim).transpose(1, 2).float()
        k = F.normalize(k_raw.float(), dim=-1, eps=1e-6)
        q = q_raw.float()

        n_gates = 3 if self.use_symplectic_reflection else 2
        gates = self.gate_proj(x).view(bsz, t_len, self.n_heads, n_gates).permute(0, 2, 1, 3).float()
        if self.use_symplectic_reflection:
            u = F.normalize(self.u_proj(x_addr).view(bsz, t_len, self.n_heads, self.head_dim).float(), dim=-1, eps=1e-6).transpose(1, 2)
            beta = 1.98 * torch.sigmoid(gates[..., 0])
            tau = torch.sigmoid(gates[..., 1])
            log_gamma = -F.softplus(gates[..., 2]) * 0.20
        else:
            u = None
            beta = None
            tau = torch.sigmoid(gates[..., 0])
            log_gamma = -F.softplus(gates[..., 1]) * 0.20

        init_s = state["s"] if state is not None else None
        y_met, s_next = _run_symplectic_wy_scan(
            u, k, v, q, beta, tau, log_gamma, init_s, self.chunk_size
        )

        # Precision normalization & epistemic uncertainty u_t
        unc_out: Tensor | None = None
        next_prec: dict[str, Tensor] | None = None
        if self.use_precision_norm:
            prior_prec = F.softplus(self.prior_log_prec.float()) + 1e-2
            prec_state = state.get("prec") if state is not None else None
            if t_len == 1:
                with _no_autocast_ctx(x.device):
                    phi_q = _positive_feature_map(q_raw)[:, :, 0]
                    phi_k = _positive_feature_map(k_raw)[:, :, 0]
                    kt_p = phi_k * tau[:, :, 0].unsqueeze(-1)
                    if prec_state is None:
                        p_prev = torch.zeros(bsz, self.n_heads, self.head_dim, device=x.device, dtype=torch.float32)
                        ell_prev = torch.zeros(bsz, self.n_heads, device=x.device, dtype=torch.float32)
                    else:
                        p_prev = prec_state["p"].float()
                        ell_prev = prec_state["ell"].float()
                    g1 = torch.exp(log_gamma[:, :, 0])
                    p_next = g1.unsqueeze(-1) * p_prev + kt_p
                    ell_next = ell_prev + log_gamma[:, :, 0]
                    prec = torch.einsum("bhd,bhd->bh", phi_q, p_next) + prior_prec.view(1, -1) * torch.exp(
                        ell_next.clamp(max=0.0)
                    ) + 1e-4
                    unc = (1.0 / prec).unsqueeze(2)
                    next_prec = {"p": p_next, "ell": ell_next}
            else:
                unc, next_prec = nebn_precision_trace(
                    q_raw, k_raw, log_gamma, tau, prior_prec, init_state=prec_state
                )
            # Subspace-energy epistemic uncertainty + centered precision confidence scaling
            subspace_energy = y_met.norm(dim=-1) / y_met.size(-1) ** 0.5 + 1e-2
            conf = 2.0 / (1.0 + unc.clamp(max=10.0).unsqueeze(-1))
            y_met = y_met * conf
            unc_out = (unc / subspace_energy).transpose(1, 2).to(x.dtype)  # [B, T, H]

        y_cat = y_met.to(x.dtype).transpose(1, 2).contiguous().view(bsz, t_len, d_model)

        # Fixed-budget epistemic-surprise SCAM register bank
        next_scam: dict[str, Any] | None = None
        if self.use_scam_memory:
            q_s = self.scam_q(x_addr)
            k_s = self.scam_k(x_addr)
            v_s = self.scam_v(x)
            base_unc = unc_out.float().mean(dim=-1) if unc_out is not None else torch.zeros(bsz, t_len, device=x.device)
            scam_score = self.scam_alpha(x).squeeze(-1).float() + base_unc
            scam_state = state.get("scam") if state is not None else None
            if t_len == 1:
                scam_y, next_scam = scam_step_attention(
                    q_s[:, 0], k_s[:, 0], v_s[:, 0], scam_score[:, 0], scam_state, self.scam_slots
                )
                scam_y = scam_y.unsqueeze(1)
            else:
                scam_y, next_scam = scam_causal_topk_attention(
                    q_s, k_s, v_s, scam_score, k_slots=self.scam_slots, mem_state=scam_state
                )
            y_cat = y_cat + self.scam_o(scam_y)

        out = self.o_proj(y_cat)
        next_state: dict[str, Any] = {"s": s_next, "shift": next_shift}
        if next_prec is not None:
            next_state["prec"] = next_prec
        if next_scam is not None:
            next_state["scam"] = next_scam
        return out, unc_out, next_state


# ---------------------------------------------------------------------------
# Synthesis / Channel-Mixing Blocks
# ---------------------------------------------------------------------------


class SwiGLUSynthesis(nn.Module):
    """Conventional SwiGLU MLP channel block."""

    def __init__(self, d_model: int, ffn_hidden: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, ffn_hidden, bias=False)
        self.w_up = nn.Linear(d_model, ffn_hidden, bias=False)
        self.w_down = nn.Linear(ffn_hidden, d_model, bias=False)

    def forward(self, x: Tensor, unc: Tensor | None = None) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class SymplecticHamiltonianSynthesis(nn.Module):
    """Port-Hamiltonian (q, p) phase-space symplectic evolution + Rayleigh damping port."""

    def __init__(self, d_model: int, ffn_hidden: int, n_heads: int = 0) -> None:
        super().__init__()
        half = d_model // 2
        self.w_pot_in = nn.Linear(half, ffn_hidden, bias=False)
        self.w_pot_out = nn.Linear(ffn_hidden, half, bias=False)
        self.w_kin_in = nn.Linear(half, ffn_hidden, bias=False)
        self.w_kin_out = nn.Linear(ffn_hidden, half, bias=False)
        self.w_damp = nn.Linear(d_model, half, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)
        self.unc_scale = nn.Linear(n_heads, d_model, bias=True) if n_heads > 0 else None

    def forward(self, x: Tensor, unc: Tensor | None = None) -> Tensor:
        q, p = x.chunk(2, dim=-1)
        # Leapfrog-like phase update: velocity from T(p), force from -V(q) minus damped port
        dq = self.w_kin_out(F.silu(self.w_kin_in(p)))
        q_mid = q + 0.5 * dq
        force = -self.w_pot_out(F.silu(self.w_pot_in(q_mid)))
        damp = torch.sigmoid(self.w_damp(x))
        p_next = (1.0 - 0.5 * damp) * p + force
        q_next = q_mid + 0.5 * dq
        phase = torch.cat([q_next, p_next], dim=-1)
        if self.unc_scale is not None and unc is not None:
            gate = torch.sigmoid(self.unc_scale(unc.to(x.dtype)))
            phase = phase * (0.5 + gate)
        return self.w_out(phase)


class VariationalDendriticSynthesis(nn.Module):
    """Epistemic-uncertainty-gated nonlinear refinement block for NEBN."""

    def __init__(self, d_model: int, n_heads: int, ffn_hidden: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, ffn_hidden, bias=False)
        self.w_up = nn.Linear(d_model, ffn_hidden, bias=False)
        self.w_down = nn.Linear(ffn_hidden, d_model, bias=False)
        self.unc_gate = nn.Linear(n_heads, ffn_hidden, bias=True)

    def forward(self, x: Tensor, unc: Tensor | None = None) -> Tensor:
        val = self.w_up(x)
        nonlin = F.silu(self.w_gate(x)) * val
        if unc is not None:
            g_unc = torch.sigmoid(self.unc_gate(unc.to(x.dtype)))
            mixed = g_unc * nonlin + (1.0 - g_unc) * val
        else:
            mixed = nonlin
        return self.w_down(mixed)
