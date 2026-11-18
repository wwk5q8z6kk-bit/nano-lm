"""NANOLM Omega Architectural Innovations: Lie-Exp SU(2)/SO(3) Adjoint & Spectral Nullspace LoRA.

1. ``su2_lie_exponential`` + ``so3_projective_adjoint``:
   Overcomes Cayley-map pole compression at 180-degree involutions (w = 0) and
   sign ambiguity (+/- q) via an analytical Lie-group exponential coupled with
   the faithful Z2-invariant SO(3) adjoint representation R(q) in R^{3x3}.
2. ``SpectralNullspaceLoRALinear`` + ``apply_nullspace_lora``:
   Prevents catastrophic forgetting of pretrained coding & reasoning circuits by
   projecting incoming LoRA features onto the orthogonal complement of the
   top-k right singular subspace V_k of the frozen weight W_0.
3. ``HSPAOmegaSidecar`` + ``apply_omega_graft``:
   Unifies Spectral Nullspace LoRA with a zero-initialized SU(2)->SO(3)
   holonomic register sidecar on upper decoder layers.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nanolm_frontier.surgery_and_rlvr import _bypass_adapters
from nanolm_holonomy.spinor_algebra import hamilton_product, spinor_conjugate, step_holonomy
from nanolm_holonomy.triton_holonomy import holonomy_scan_fast
from nanolm_posttrain.architecture import ABSOLUTE_MAX_PARAMETERS, ParameterBudgetExceededError
from nanolm_posttrain.lora import LoRAConfig


def su2_lie_exponential(omega: Tensor) -> Tensor:
    """Exact SU(2) exponential map ``exp(sum w_i e_i)`` producing unit quaternions.

    Smooth everywhere including ``||omega|| = pi/2`` (pure involutions ``w = 0``),
    where the Cayley rational map has a coordinate pole at infinity.
    """
    wf = omega.float()
    theta = torch.linalg.vector_norm(wf, dim=-1, keepdim=True)
    cos_t = torch.cos(theta)
    # 2nd-order Taylor fallback around 0 keeps autograd smooth and exact at origin
    sinc_t = torch.where(
        theta > 1e-4,
        torch.sin(theta) / theta.clamp_min(1e-4),
        1.0 - (theta * theta) / 6.0,
    )
    q = torch.cat((cos_t, sinc_t * wf), dim=-1)
    return q.to(omega.dtype)


def so3_projective_adjoint(q: Tensor) -> Tensor:
    """Exact 3x3 SO(3) adjoint rotation representation ``R(q)`` flattened to ``[..., 9]``.

    Satisfies:
    - Exact ``Z_2`` double-cover invariance: ``R(-q) == R(q)``
    - Homomorphism on non-abelian group multiplication: ``R(q2 * q1) == R(q2) @ R(q1)``
    """
    qf = q.float()
    norm = torch.linalg.vector_norm(qf, dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = (qf / norm).unbind(dim=-1)
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - w * z)
    r02 = 2.0 * (x * z + w * y)
    r10 = 2.0 * (x * y + w * z)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - w * x)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return torch.stack((r00, r01, r02, r10, r11, r12, r20, r21, r22), dim=-1).to(q.dtype)


class SpectralNullspaceLoRALinear(nn.Module):
    """Frozen ``nn.Linear`` wrapped with top-k singular-subspace-orthogonal LoRA."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        nullspace_k: int = 48,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        in_features = base.in_features
        out_features = base.out_features
        self.rank = rank
        self.scaling = alpha / float(rank)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        device = base.weight.device
        buf_dtype = base.weight.dtype if base.weight.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
        k_eff = max(0, min(nullspace_k, min(out_features, in_features) // 4))
        self.nullspace_k = k_eff

        if k_eff > 0:
            w_f = base.weight.detach().float()
            try:
                q_over = min(k_eff + 8, min(w_f.shape))
                _, _, v_mat = torch.svd_lowrank(w_f, q=q_over, niter=2)
                v_k = v_mat[:, :k_eff].contiguous()
            except RuntimeError:
                _, _, vh = torch.linalg.svd(w_f, full_matrices=False)
                v_k = vh[:k_eff, :].T.contiguous()
            # Re-orthonormalize strictly for exact P_perp idempotency
            v_k, _ = torch.linalg.qr(v_k)
            self.register_buffer("v_proj_subspace", v_k.to(device=device, dtype=buf_dtype))
        else:
            self.register_buffer(
                "v_proj_subspace",
                torch.empty((in_features, 0), device=device, dtype=buf_dtype),
            )

        self.lora_a = nn.Parameter(torch.empty((rank, in_features), device=device, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros((out_features, rank), device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def effective_delta_weight(self) -> Tensor:
        """Exact effective weight delta ``scaling * (B @ A) @ (I - V_k V_k^T)``."""
        raw_d = self.scaling * (self.lora_b @ self.lora_a)
        if self.nullspace_k <= 0:
            return raw_d
        vk = self.v_proj_subspace.float()
        return raw_d - (raw_d @ vk) @ vk.T

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        if self.scaling == 0.0:
            return base_out
        xd = self.drop(x)
        if self.nullspace_k > 0:
            vk = self.v_proj_subspace
            x_v = xd.to(vk.dtype)
            coeff = x_v @ vk
            x_perp = x_v.float() - (coeff @ vk.T).float()
        else:
            x_perp = xd.float()
        delta = F.linear(F.linear(x_perp, self.lora_a), self.lora_b) * self.scaling
        return base_out + delta.to(base_out.dtype)


def apply_nullspace_lora(
    model: nn.Module,
    config: LoRAConfig = LoRAConfig(),
    *,
    nullspace_k: int = 48,
) -> int:
    """Freeze base model (<= 4B cap) and attach SpectralNullspaceLoRALinear adapters."""
    total_base = sum(p.numel() for p in model.parameters())
    if total_base > ABSOLUTE_MAX_PARAMETERS:
        raise ParameterBudgetExceededError(
            f"base model has {total_base:,} params > 4B limit"
        )
    for p in model.parameters():
        p.requires_grad_(False)

    replaced = 0
    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(full_name == s or full_name.endswith("." + s) for s in config.target_suffixes):
            continue
        parts = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(
            parent,
            parts[-1],
            SpectralNullspaceLoRALinear(
                module,
                rank=config.rank,
                alpha=config.alpha,
                nullspace_k=nullspace_k,
                dropout=config.dropout,
            ),
        )
        replaced += 1
    if replaced == 0:
        raise ValueError(f"no linear layers matched target_suffixes={config.target_suffixes}")
    return replaced


class HSPAOmegaSidecar(nn.Module):
    """Upper-layer Non-Abelian Lie-Exp SU(2) -> SO(3) Projective Adjoint Sidecar."""

    def __init__(self, d_model: int, *, n_registers: int = 4, d_fiber: int = 128) -> None:
        super().__init__()
        if d_fiber % 4 != 0:
            raise ValueError("d_fiber must be divisible by 4")
        self.R = n_registers
        self.G = d_fiber // 4
        self.d_fiber = d_fiber
        self.bypass = False
        self._cached_scan_state: Tensor | None = None

        self.in_norm = nn.RMSNorm(d_model)
        self.phi_proj = nn.Linear(d_model, d_fiber, bias=False)
        self.gate_proj = nn.Linear(d_model, 3 * self.R, bias=False)
        self.spinor_proj = nn.Linear(d_model, 9 * self.G, bias=False)
        # Bifurcate registers: half near-unitary holonomy tracking, half episodic decay
        beta_init = torch.tensor([-3.5 if r < self.R // 2 else -0.5 for r in range(self.R)], dtype=torch.float32)
        self.beta_bias = nn.Parameter(beta_init)
        # Project concatenated 4D quaternion state + 9D SO(3) adjoint readout
        self.so3_proj = nn.Linear(self.G * 9, d_fiber, bias=False)
        self.o_proj = nn.Linear(d_fiber, d_model, bias=False)
        self.sidecar_gate = nn.Parameter(torch.zeros(d_model, dtype=torch.float32))

    def reset_cache(self) -> None:
        self._cached_scan_state = None

    def forward(self, h: Tensor) -> Tensor:
        if self.bypass:
            return torch.zeros_like(h)

        bsz, seq_len, _ = h.shape
        x = self.in_norm(h.float())
        phi = self.phi_proj(x).view(bsz, seq_len, self.G, 4).permute(0, 2, 1, 3).unsqueeze(1)
        w_l, b_l, r_l = self.gate_proj(x).chunk(3, dim=-1)
        b_l = b_l + self.beta_bias.view(1, 1, self.R)
        w = F.softmax(w_l, dim=-1).transpose(1, 2).unsqueeze(2).unsqueeze(-1)
        beta = torch.sigmoid(b_l).transpose(1, 2).unsqueeze(2).unsqueeze(-1)
        read_pi = F.softmax(r_l, dim=-1).transpose(1, 2).unsqueeze(-1)

        o_gen, rho_gen, q_gen = self.spinor_proj(x).chunk(3, dim=-1)
        o_gen = o_gen.view(bsz, seq_len, self.G, 3).permute(0, 2, 1, 3).unsqueeze(1)
        rho_gen = rho_gen.view(bsz, seq_len, self.G, 3).permute(0, 2, 1, 3).unsqueeze(1)
        q_gen = q_gen.view(bsz, seq_len, self.G, 3).permute(0, 2, 1, 3).unsqueeze(1)

        g = (1.0 - w * beta).clamp_min(1e-4) * su2_lie_exponential(o_gen)
        s = w * hamilton_product(su2_lie_exponential(rho_gen), phi)

        if (
            not self.training
            and seq_len == 1
            and self._cached_scan_state is not None
            and self._cached_scan_state.shape[0] == bsz
        ):
            nxt = step_holonomy(g.squeeze(-2), s.squeeze(-2), self._cached_scan_state)
            self._cached_scan_state = nxt.detach()
            all_omega = nxt.unsqueeze(-2)
        else:
            all_omega, nxt = holonomy_scan_fast(g, s, init_state=None, chunk_size=256)
            if not self.training:
                self._cached_scan_state = nxt.detach()

        unbound = hamilton_product(spinor_conjugate(su2_lie_exponential(q_gen)), all_omega)
        adj = so3_projective_adjoint(unbound).permute(0, 1, 3, 2, 4).reshape(bsz, self.R, seq_len, self.G * 9)
        flat = unbound.permute(0, 1, 3, 2, 4).reshape(bsz, self.R, seq_len, self.d_fiber)
        combined = flat + self.so3_proj(adj)
        out = self.o_proj((read_pi * combined).sum(dim=1))
        return (torch.tanh(self.sidecar_gate) * out).to(h.dtype)


class _OmegaWrappedLayer(nn.Module):
    def __init__(self, base_layer: nn.Module, sidecar: HSPAOmegaSidecar) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.sidecar = sidecar

    def forward(self, hidden_states: Tensor, *args: Any, **kwargs: Any) -> Any:
        out = self.base_layer(hidden_states, *args, **kwargs)
        if isinstance(out, tuple):
            h_out = out[0]
            return (h_out + self.sidecar(h_out), *out[1:])
        return out + self.sidecar(out)


def apply_omega_graft(
    model: nn.Module,
    *,
    lora_config: LoRAConfig = LoRAConfig(rank=16, alpha=32.0),
    nullspace_k: int = 48,
    n_sidecar_layers: int = 8,
    n_registers: int = 4,
    d_fiber: int = 128,
) -> dict[str, Any]:
    """Attach SpectralNullspaceLoRA + zero-init Lie-Exp SU(2)->SO(3) Sidecars."""
    total_before = sum(p.numel() for p in model.parameters())
    if total_before > ABSOLUTE_MAX_PARAMETERS:
        raise ParameterBudgetExceededError(f"base model {total_before:,} > 4B limit")

    lora_n = apply_nullspace_lora(model, lora_config, nullspace_k=nullspace_k)
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else getattr(model, "layers", None)
    d_model = int(getattr(model.config, "hidden_size", 0))
    ref_param = next(model.parameters())

    sidecars = 0
    if layers is not None and d_model > 0:
        n_layers = len(layers)
        start_idx = max(0, n_layers - n_sidecar_layers)
        for idx in range(start_idx, n_layers):
            sc = HSPAOmegaSidecar(d_model, n_registers=n_registers, d_fiber=d_fiber).to(ref_param.device)
            layers[idx] = _OmegaWrappedLayer(layers[idx], sc)
            sidecars += 1

    total_after = sum(p.numel() for p in model.parameters())
    if total_after > ABSOLUTE_MAX_PARAMETERS:
        raise ParameterBudgetExceededError(f"grafted model {total_after:,} > 4B limit")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "nullspace_lora_layers": lora_n,
        "nullspace_k": nullspace_k,
        "omega_sidecar_layers": sidecars,
        "total_parameters": total_after,
        "trainable_parameters": trainable,
        "trainable_pct": round(100.0 * trainable / max(total_after, 1), 4),
    }


@contextmanager
def _bypass_omega_adapters(model: nn.Module) -> Iterator[None]:
    saved_ns: list[tuple[SpectralNullspaceLoRALinear, float]] = []
    saved_sc: list[tuple[HSPAOmegaSidecar, bool]] = []
    for mod in model.modules():
        if isinstance(mod, SpectralNullspaceLoRALinear):
            saved_ns.append((mod, mod.scaling))
            mod.scaling = 0.0
        elif isinstance(mod, HSPAOmegaSidecar):
            saved_sc.append((mod, mod.bypass))
            mod.bypass = True
    with _bypass_adapters(model):
        try:
            yield
        finally:
            for mod, sc in saved_ns:
                mod.scaling = sc
            for mod, flg in saved_sc:
                mod.bypass = flg
