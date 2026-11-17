"""Non-Abelian SU(2) Spinor Gauge Transport & Matched U(1) Abelian Control Primitives.

Implements:
1. Exact vectorized quaternionic Hamilton product ``hamilton_product(p, q)``,
   conjugate unbinding ``spinor_conjugate(q)``, and Cayley Lie-algebraic rotor
   ``cayley_rotor(omega)`` in SU(2).
2. Parallel Hillis-Steele O(log T) associative scan ``parallel_holonomy_scan(g, s)``
   for non-commutative affine gauge recurrence Omega_t = g_t * Omega_{t-1} + s_t,
   plus O(1)-state single-step inference ``step_holonomy`` and chunked execution.
3. Exact matched Abelian U(1) complex algebra (``complex_product``, ``complex_cayley_rotor``,
   ``abelian_complex_scan``) enabling pure falsifiable commutativity vs non-commutativity
   ablations at identical state width and scan depth.
"""

from __future__ import annotations

import torch
from torch import Tensor


def hamilton_product(p: Tensor, q: Tensor) -> Tensor:
    """Exact non-commutative, associative Hamilton product on trailing dim 4 ``(w, x, y, z)``."""
    p0, p1, p2, p3 = p.unbind(dim=-1)
    q0, q1, q2, q3 = q.unbind(dim=-1)
    r0 = p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3
    r1 = p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2
    r2 = p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1
    r3 = p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0
    return torch.stack((r0, r1, r2, r3), dim=-1)


def spinor_conjugate(q: Tensor) -> Tensor:
    """Quaternionic conjugate ``q_bar = (w, -x, -y, -z)`` for inverse unbinding."""
    q0, q1, q2, q3 = q.unbind(dim=-1)
    return torch.stack((q0, -q1, -q2, -q3), dim=-1)


def cayley_rotor(omega: Tensor, eps: float = 1e-8) -> Tensor:
    """Map su(2) vector ``omega[..., 3]`` via algebraic Cayley map into exact unit quaternion in SU(2)."""
    ones = torch.ones_like(omega[..., :1])
    raw = torch.cat((ones, omega), dim=-1).float()
    norm = torch.sqrt(torch.sum(raw * raw, dim=-1, keepdim=True).clamp_min(eps))
    return (raw / norm).to(omega.dtype)


def step_holonomy(g_t: Tensor, s_t: Tensor, state: Tensor | None) -> Tensor:
    """Exact O(1)-memory single-step update ``Omega_t = g_t * Omega_{t-1} + s_t``."""
    if state is None:
        return s_t.float().to(s_t.dtype)
    out = hamilton_product(g_t.float(), state.float()) + s_t.float()
    return out.to(s_t.dtype)


def sequential_holonomy_reference(
    g: Tensor,
    s: Tensor,
    *,
    init_state: Tensor | None = None,
) -> Tensor:
    """Token-by-token reference recurrence for mathematical verification (O(T) Python loop)."""
    seq_len = g.shape[-2]
    state = init_state.float() if init_state is not None else torch.zeros_like(s[..., 0, :], dtype=torch.float32)
    gf = g.float()
    sf = s.float()
    outputs: list[Tensor] = []
    for t in range(seq_len):
        state = hamilton_product(gf[..., t, :], state) + sf[..., t, :]
        outputs.append(state)
    return torch.stack(outputs, dim=-2).to(s.dtype)


def parallel_holonomy_scan(
    g: Tensor,
    s: Tensor,
    *,
    init_state: Tensor | None = None,
    chunk_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """O(log T) Hillis-Steele associative scan for non-abelian recurrence ``Omega_t = g_t * Omega_{t-1} + s_t``.

    Notice composition preserves exact non-commutative order:
        ``(g2, s2) o (g1, s1) = (g2 * g1, g2 * s1 + s2)``.

    Args:
        g: ``[..., T, 4]`` transition rotors.
        s: ``[..., T, 4]`` injected bound spinors.
        init_state: Optional ``[..., 4]`` carry state from prior chunk.
        chunk_size: If provided, runs exact chunk-carried scans to cap peak activation memory.
    Returns:
        ``(all_states [..., T, 4], final_state [..., 4])`` in input dtype.
    """
    orig_dtype = s.dtype
    seq_len = g.shape[-2]

    if chunk_size is not None and seq_len > chunk_size:
        chunks_out: list[Tensor] = []
        carry = init_state
        for start in range(0, seq_len, chunk_size):
            end = min(seq_len, start + chunk_size)
            c_out, carry = parallel_holonomy_scan(
                g[..., start:end, :],
                s[..., start:end, :],
                init_state=carry,
                chunk_size=None,
            )
            chunks_out.append(c_out)
        assert carry is not None
        return torch.cat(chunks_out, dim=-2), carry

    cur_g = g.float()
    cur_s = s.float()
    if init_state is not None:
        # Fold init_state cleanly into t=0 via s_0' = g_0 * init_state + s_0
        first_s = hamilton_product(cur_g[..., :1, :], init_state.float().unsqueeze(-2)) + cur_s[..., :1, :]
        cur_s = torch.cat((first_s, cur_s[..., 1:, :]), dim=-2)

    offset = 1
    while offset < seq_len:
        g_hi = cur_g[..., offset:, :]
        g_lo = cur_g[..., :-offset, :]
        s_hi = cur_s[..., offset:, :]
        s_lo = cur_s[..., :-offset, :]
        g_comb = hamilton_product(g_hi, g_lo)
        s_comb = hamilton_product(g_hi, s_lo) + s_hi
        cur_g = torch.cat((cur_g[..., :offset, :], g_comb), dim=-2)
        cur_s = torch.cat((cur_s[..., :offset, :], s_comb), dim=-2)
        offset <<= 1

    final_state = cur_s[..., -1, :].to(orig_dtype)
    return cur_s.to(orig_dtype), final_state


# ---------------------------------------------------------------------------
# Abelian U(1) Commutative Control Algebra (Exact Iso-Dim / Iso-Scan Depth)
# ---------------------------------------------------------------------------

def complex_product(p: Tensor, q: Tensor) -> Tensor:
    """Commutative complex multiplication on trailing dim 2 ``(re, im)``."""
    p0, p1 = p.unbind(dim=-1)
    q0, q1 = q.unbind(dim=-1)
    return torch.stack((p0 * q0 - p1 * q1, p0 * q1 + p1 * q0), dim=-1)


def complex_conjugate(q: Tensor) -> Tensor:
    """Complex conjugate ``(re, -im)``."""
    q0, q1 = q.unbind(dim=-1)
    return torch.stack((q0, -q1), dim=-1)


def complex_cayley_rotor(theta: Tensor, eps: float = 1e-8) -> Tensor:
    """Algebraic U(1) unit phase rotor from scalar angle parameter ``theta[..., 1]``."""
    ones = torch.ones_like(theta)
    raw = torch.cat((ones, theta), dim=-1).float()
    norm = torch.sqrt(torch.sum(raw * raw, dim=-1, keepdim=True).clamp_min(eps))
    return (raw / norm).to(theta.dtype)


def abelian_complex_scan(
    g: Tensor,
    s: Tensor,
    *,
    init_state: Tensor | None = None,
    chunk_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """O(log T) Hillis-Steele associative scan under commutative U(1) complex multiplication."""
    orig_dtype = s.dtype
    seq_len = g.shape[-2]

    if chunk_size is not None and seq_len > chunk_size:
        chunks_out: list[Tensor] = []
        carry = init_state
        for start in range(0, seq_len, chunk_size):
            end = min(seq_len, start + chunk_size)
            c_out, carry = abelian_complex_scan(
                g[..., start:end, :],
                s[..., start:end, :],
                init_state=carry,
                chunk_size=None,
            )
            chunks_out.append(c_out)
        assert carry is not None
        return torch.cat(chunks_out, dim=-2), carry

    cur_g = g.float()
    cur_s = s.float()
    if init_state is not None:
        first_s = complex_product(cur_g[..., :1, :], init_state.float().unsqueeze(-2)) + cur_s[..., :1, :]
        cur_s = torch.cat((first_s, cur_s[..., 1:, :]), dim=-2)

    offset = 1
    while offset < seq_len:
        g_hi = cur_g[..., offset:, :]
        g_lo = cur_g[..., :-offset, :]
        s_hi = cur_s[..., offset:, :]
        s_lo = cur_s[..., :-offset, :]
        cur_g = torch.cat((cur_g[..., :offset, :], complex_product(g_hi, g_lo)), dim=-2)
        cur_s = torch.cat((cur_s[..., :offset, :], complex_product(g_hi, s_lo) + s_hi), dim=-2)
        offset <<= 1

    return cur_s.to(orig_dtype), cur_s[..., -1, :].to(orig_dtype)
