"""Fused Triton Non-Abelian SU(2) Holonomy Scan with Analytical Reverse-Mode Adjoint.

Key properties:
1. Single-pass HBM -> SRAM associative scan in OpenAI Triton (when CUDA + Triton are
   available) keeping tile-boundary carry states ``Omega_{block_end}`` in FP32 registers.
2. Exact analytical quaternionic reverse-mode adjoint via time-reversed conjugate scan:
   saves ONLY ``(g, init_state, omega_all)`` across forward instead of O(log T) binary
   scan intermediates, cutting backward activation memory dramatically.
3. Seamless CPU/MPS fallback using the exact same analytical adjoint over
   ``parallel_holonomy_scan`` so ``gradcheck`` and local unit tests validate the exact
   gradient equations Bit-for-Bit.
"""

from __future__ import annotations

import torch
from torch import Tensor

from nanolm_holonomy.spinor_algebra import (
    hamilton_product,
    parallel_holonomy_scan,
    spinor_conjugate,
)

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - optional on non-CUDA local hosts
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON = False


if HAS_TRITON:  # pragma: no cover - exercised on RunPod CUDA

    @triton.jit
    def _su2_affine_combine(
        ag0, ag1, ag2, ag3, as0, as1, as2, as3,
        bg0, bg1, bg2, bg3, bs0, bs1, bs2, bs3,
    ):
        """Associative left-acting composition ``b o a = (b_g * a_g, b_g * a_s + b_s)``."""
        ng0 = bg0 * ag0 - bg1 * ag1 - bg2 * ag2 - bg3 * ag3
        ng1 = bg0 * ag1 + bg1 * ag0 + bg2 * ag3 - bg3 * ag2
        ng2 = bg0 * ag2 - bg1 * ag3 + bg2 * ag0 + bg3 * ag1
        ng3 = bg0 * ag3 + bg1 * ag2 - bg2 * ag1 + bg3 * ag0

        ns0 = bg0 * as0 - bg1 * as1 - bg2 * as2 - bg3 * as3 + bs0
        ns1 = bg0 * as1 + bg1 * as0 + bg2 * as3 - bg3 * as2 + bs1
        ns2 = bg0 * as2 - bg1 * as3 + bg2 * as0 + bg3 * as1 + bs2
        ns3 = bg0 * as3 + bg1 * as2 - bg2 * as1 + bg3 * as0 + bs3
        return ng0, ng1, ng2, ng3, ns0, ns1, ns2, ns3

    @triton.jit
    def _triton_holonomy_fwd_kernel(
        G_ptr,
        S_ptr,
        Init_ptr,
        Out_ptr,
        T,
        has_init: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        pid = tl.program_id(0)
        seq_base = pid * T * 4

        if has_init:
            ib = pid * 4
            c0 = tl.load(Init_ptr + ib + 0).to(tl.float32)
            c1 = tl.load(Init_ptr + ib + 1).to(tl.float32)
            c2 = tl.load(Init_ptr + ib + 2).to(tl.float32)
            c3 = tl.load(Init_ptr + ib + 3).to(tl.float32)
        else:
            c0 = 0.0
            c1 = 0.0
            c2 = 0.0
            c3 = 0.0

        local_idx = tl.arange(0, BLOCK_T)
        is_first = local_idx == 0
        is_last = local_idx == (BLOCK_T - 1)

        n_tiles = tl.cdiv(T, BLOCK_T)
        for tile_idx in range(n_tiles):
            t_offs = tile_idx * BLOCK_T + local_idx
            mask = t_offs < T
            ptr_offs = seq_base + t_offs * 4

            g0 = tl.load(G_ptr + ptr_offs + 0, mask=mask, other=1.0).to(tl.float32)
            g1 = tl.load(G_ptr + ptr_offs + 1, mask=mask, other=0.0).to(tl.float32)
            g2 = tl.load(G_ptr + ptr_offs + 2, mask=mask, other=0.0).to(tl.float32)
            g3 = tl.load(G_ptr + ptr_offs + 3, mask=mask, other=0.0).to(tl.float32)

            s0 = tl.load(S_ptr + ptr_offs + 0, mask=mask, other=0.0).to(tl.float32)
            s1 = tl.load(S_ptr + ptr_offs + 1, mask=mask, other=0.0).to(tl.float32)
            s2 = tl.load(S_ptr + ptr_offs + 2, mask=mask, other=0.0).to(tl.float32)
            s3 = tl.load(S_ptr + ptr_offs + 3, mask=mask, other=0.0).to(tl.float32)

            # Fold previous tile carry into t_local == 0: s_0' = g_0 * carry + s_0
            fc0 = g0 * c0 - g1 * c1 - g2 * c2 - g3 * c3
            fc1 = g0 * c1 + g1 * c0 + g2 * c3 - g3 * c2
            fc2 = g0 * c2 - g1 * c3 + g2 * c0 + g3 * c1
            fc3 = g0 * c3 + g1 * c2 - g2 * c1 + g3 * c0

            s0 = tl.where(is_first, s0 + fc0, s0)
            s1 = tl.where(is_first, s1 + fc1, s1)
            s2 = tl.where(is_first, s2 + fc2, s2)
            s3 = tl.where(is_first, s3 + fc3, s3)

            _, _, _, _, o0, o1, o2, o3 = tl.associative_scan(
                (g0, g1, g2, g3, s0, s1, s2, s3),
                0,
                _su2_affine_combine,
            )

            tl.store(Out_ptr + ptr_offs + 0, o0, mask=mask)
            tl.store(Out_ptr + ptr_offs + 1, o1, mask=mask)
            tl.store(Out_ptr + ptr_offs + 2, o2, mask=mask)
            tl.store(Out_ptr + ptr_offs + 3, o3, mask=mask)

            c0 = tl.sum(tl.where(is_last, o0, 0.0), axis=0)
            c1 = tl.sum(tl.where(is_last, o1, 0.0), axis=0)
            c2 = tl.sum(tl.where(is_last, o2, 0.0), axis=0)
            c3 = tl.sum(tl.where(is_last, o3, 0.0), axis=0)


def _triton_scan_impl(g: Tensor, s: Tensor, init_state: Tensor | None) -> Tensor:
    """Execute fused Triton SU(2) scan on CUDA flattened ``[N, T, 4]`` tensors."""
    orig_shape = s.shape
    seq_len = orig_shape[-2]
    g_flat = g.contiguous().view(-1, seq_len, 4)
    s_flat = s.contiguous().view(-1, seq_len, 4)
    n_seq = g_flat.shape[0]
    out_flat = torch.empty_like(s_flat, dtype=torch.float32)

    if init_state is not None:
        init_flat = init_state.contiguous().view(n_seq, 4)
        has_init = True
    else:
        init_flat = s_flat  # unused placeholder pointer
        has_init = False

    block_t = 128 if seq_len <= 128 else 256
    _triton_holonomy_fwd_kernel[(n_seq,)](
        g_flat,
        s_flat,
        init_flat,
        out_flat,
        seq_len,
        has_init=has_init,
        BLOCK_T=block_t,
    )
    return out_flat.view(*orig_shape)


def _raw_scan_dispatch(g: Tensor, s: Tensor, init_state: Tensor | None, chunk_size: int | None) -> Tensor:
    if HAS_TRITON and g.is_cuda and g.dtype in (torch.float32, torch.bfloat16, torch.float16):
        return _triton_scan_impl(g, s, init_state)
    if g.dtype == torch.float64 or s.dtype == torch.float64:
        # Exact double precision sequential scan for float64 gradcheck
        seq_len = g.shape[-2]
        gf, sf = g.double(), s.double()
        state = init_state.double() if init_state is not None else torch.zeros_like(sf[..., 0, :])
        outs = []
        for t in range(seq_len):
            state = hamilton_product(gf[..., t, :], state) + sf[..., t, :]
            outs.append(state)
        return torch.stack(outs, dim=-2)
    out, _ = parallel_holonomy_scan(g, s, init_state=init_state, chunk_size=chunk_size)
    return out.float()


class HolonomyScanFn(torch.autograd.Function):
    """Custom autograd using fused Triton forward and analytical time-reversed conjugate scan."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        g: Tensor,
        s: Tensor,
        init_state: Tensor | None,
        chunk_size: int | None,
    ) -> Tensor:
        out_fp = _raw_scan_dispatch(g, s, init_state, chunk_size)
        ctx.save_for_backward(g, init_state if init_state is not None else torch.empty(0, device=g.device), out_fp)
        ctx.has_init = init_state is not None
        ctx.chunk_size = chunk_size
        ctx.in_dtype = s.dtype
        return out_fp.to(s.dtype)

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # type: ignore[override]
        g, saved_init, omega_all = ctx.saved_tensors
        seq_len = g.shape[-2]
        w_dtype = torch.float64 if ctx.in_dtype == torch.float64 else torch.float32
        gf = g.to(w_dtype)
        g_conj_rev = torch.flip(spinor_conjugate(gf), dims=(-2,))
        zero_pad = torch.zeros_like(gf[..., :1, :])
        g_rev = torch.cat((zero_pad, g_conj_rev[..., :-1, :]), dim=-2) if seq_len > 1 else zero_pad
        delta_rev = torch.flip(grad_out.to(w_dtype), dims=(-2,))

        lam_rev = _raw_scan_dispatch(g_rev, delta_rev, None, ctx.chunk_size).to(w_dtype)
        lam = torch.flip(lam_rev, dims=(-2,))

        # Grad w.r.t. s_t: exact lambda_t
        grad_s = lam.to(ctx.in_dtype)

        # Previous states [Omega_{-1}, Omega_0, ..., Omega_{T-2}]
        init_prev = saved_init.to(w_dtype).unsqueeze(-2) if ctx.has_init else torch.zeros_like(omega_all.to(w_dtype)[..., :1, :])
        if seq_len > 1:
            omega_prev = torch.cat((init_prev, omega_all.to(w_dtype)[..., :-1, :]), dim=-2)
        else:
            omega_prev = init_prev

        # Grad w.r.t. g_t: lambda_t * conj(Omega_{t-1})
        grad_g = hamilton_product(lam, spinor_conjugate(omega_prev)).to(g.dtype)

        if ctx.has_init:
            grad_init = hamilton_product(spinor_conjugate(gf[..., 0, :]), lam[..., 0, :]).to(saved_init.dtype)
        else:
            grad_init = None

        return grad_g, grad_s, grad_init, None


def holonomy_scan_fast(
    g: Tensor,
    s: Tensor,
    *,
    init_state: Tensor | None = None,
    chunk_size: int | None = 512,
) -> tuple[Tensor, Tensor]:
    """Differentiable fast SU(2) holonomy scan returning ``(all_states, final_state)``."""
    all_states = HolonomyScanFn.apply(g, s, init_state, chunk_size)
    return all_states, all_states[..., -1, :]
