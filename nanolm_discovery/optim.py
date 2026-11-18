"""Muon + AdamW decoupled optimizer for isolating architecture vs optimizer effects."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer


def zeropower_via_newtonschulz5(grad: Tensor, steps: int = 5) -> Tensor:
    """Orthogonalize a 2D gradient/update matrix using 5 Newton-Schulz iterations in float32."""
    if grad.ndim != 2:
        raise ValueError(f"Newton-Schulz expects 2D tensor, got {grad.ndim}D")
    a, b, c = 3.4445, -4.7750, 2.0315
    x = grad.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.T
        gram2 = gram @ gram
        x = a * x + (b * gram + c * gram2) @ x
    return x.T if transposed else x


class MuonAdamW(Optimizer):
    """Applies Newton-Schulz orthogonalized Muon updates to internal 2D linear weights and AdamW elsewhere."""

    def __init__(
        self,
        muon_params: Iterable[nn.Parameter],
        adamw_params: Iterable[nn.Parameter],
        *,
        lr_muon: float = 1.5e-3,
        lr_adamw: float = 6e-4,
        momentum: float = 0.92,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.05,
        ns_steps: int = 5,
    ) -> None:
        muon_list = [p for p in muon_params if p.requires_grad]
        adamw_list = [p for p in adamw_params if p.requires_grad]
        param_groups: list[dict[str, Any]] = []
        if muon_list:
            param_groups.append(
                {
                    "params": muon_list,
                    "kind": "muon",
                    "lr": lr_muon,
                    "momentum": momentum,
                    "weight_decay": weight_decay,
                    "ns_steps": ns_steps,
                }
            )
        if adamw_list:
            param_groups.append(
                {
                    "params": adamw_list,
                    "kind": "adamw",
                    "lr": lr_adamw,
                    "betas": betas,
                    "weight_decay": weight_decay,
                    "eps": 1e-8,
                }
            )
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure: Any = None) -> Tensor | None:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            kind = group["kind"]
            if kind == "muon":
                lr = float(group["lr"])
                mom = float(group["momentum"])
                wd = float(group["weight_decay"])
                ns_steps = int(group["ns_steps"])
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad.float()
                    state = self.state[p]
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.zeros_like(g)
                        state["momentum_buffer"] = buf
                    buf.mul_(mom).add_(g)
                    update_dir = zeropower_via_newtonschulz5(buf, steps=ns_steps)
                    scale = max(1.0, float(p.shape[0]) / float(p.shape[1])) ** 0.5
                    if wd > 0.0:
                        p.mul_(1.0 - lr * wd)
                    p.add_(update_dir.to(p.dtype), alpha=-lr * scale)
            else:
                lr = float(group["lr"])
                b1, b2 = group["betas"]
                wd = float(group["weight_decay"])
                eps = float(group["eps"])
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad.float()
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(g)
                        state["exp_avg_sq"] = torch.zeros_like(g)
                    state["step"] += 1
                    t = state["step"]
                    m, v = state["exp_avg"], state["exp_avg_sq"]
                    m.mul_(b1).add_(g, alpha=1.0 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                    m_hat = m / (1.0 - b1**t)
                    v_hat = v / (1.0 - b2**t)
                    if wd > 0.0 and p.ndim >= 2:
                        p.mul_(1.0 - lr * wd)
                    p.add_((-lr * (m_hat / (v_hat.sqrt() + eps))).to(p.dtype))
        return loss


def build_optimizer(
    model: Any,
    *,
    kind: str = "adamw",
    lr: float = 8e-4,
    lr_muon: float = 1.5e-3,
    weight_decay: float = 0.05,
) -> Optimizer:
    """Create either decoupled `adamw` or `muon` (2D internal matrices + AdamW embeddings/1D)."""
    if kind == "adamw":
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            (decay if p.ndim >= 2 else no_decay).append(p)
        return AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, 0.95),
        )
    if kind == "muon":
        emb_ids = model.embedding_param_ids() if hasattr(model, "embedding_param_ids") else set()
        muon_p: list[nn.Parameter] = []
        adamw_p: list[nn.Parameter] = []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 2 and id(p) not in emb_ids and min(p.shape) >= 8:
                muon_p.append(p)
            else:
                adamw_p.append(p)
        return MuonAdamW(
            muon_p,
            adamw_p,
            lr_muon=lr_muon,
            lr_adamw=lr,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unknown optimizer kind {kind!r}")
