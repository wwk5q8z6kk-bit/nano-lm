"""Compact Transformer arms (Dense, ValueRes, BlockAttnRes, CodaRefine, Dense10) + Optax-Parity Muon.

Designed for controlled, matched-data and matched-wall-clock evaluation on Corpus v1 (32k BPE):
  - `dense`:         8-layer PreNorm GQA Transformer (14,292,224 params, 8 layer passes)
  - `value_res`:     8-layer PreNorm GQA + learned Layer-0 attention value skip (14,292,231 params, 8 passes)
  - `block_attnres`: 8-layer GQA + Block Attention Residuals (H-13, B=4 blocks of 2 layers, 14,300,416 params, 8 passes)
  - `coda_refine`:   8-layer GQA + zero-init 1-step shared latent refinement of top 2 layers (H-10, 14,292,480 params, 10 passes)
  - `dense10`:       10-layer untied PreNorm GQA Transformer (16,523,008 params, 10 passes; equal-FLOP control for coda_refine)

Optimizer (`OptaxParityMuon`):
  Exact PyTorch match to `optax.contrib.muon` as configured in `train_jax.py`:
  - Global gradient L2 norm clipping at `1.0` prior to optimizer transformation.
  - 2D block matrices: Nesterov momentum (`beta=0.95`) with optax bias correction, Frobenius-normalized
    5-step Newton-Schulz orthogonalization with coefficients `(3.4445, -4.7750, 2.0315)`, shape scaling
    `consistent_rms * sqrt(max(fan_in, fan_out))` (`consistent_rms = 0.2`), and decoupled weight decay `wd`.
  - Embeddings, 1D norms/gates/scalars, and 1D Block-AttnRes depth queries: Nesterov AdamW
    (`b1=0.9, b2=0.95, eps=1e-8`, decoupled `wd`).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


ARM_SPECS = (
    "dense",
    "value_res",
    "block_attnres",
    "coda_refine",
    "dense10",
)


@dataclass(frozen=True)
class ModelCfg:
    arm: str = "dense"
    vocab: int = 32768
    d: int = 256
    layers: int = 8
    n_head: int = 4
    n_kv: int = 2
    ffn: int = 704
    rope_theta: float = 10000.0
    attnres_block_layers: int = 2  # B = layers // 2 = 4 blocks for 8-layer model
    coda_layers: int = 2           # top-2 layers refined once in coda_refine


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.w.float()).to(x.dtype)


def rope_cache(T: int, hd: int, theta: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd))
    freqs = torch.outer(torch.arange(T, device=device, dtype=torch.float32), inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, hd]
    x1, x2 = x[..., ::2], x[..., 1::2]
    c = cos[None, None, : x.shape[2]].to(x.dtype)
    s = sin[None, None, : x.shape[2]].to(x.dtype)
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)


class GQAAttention(nn.Module):
    def __init__(self, cfg: ModelCfg, use_value_res: bool):
        super().__init__()
        assert cfg.d % cfg.n_head == 0 and cfg.n_head % cfg.n_kv == 0
        self.h = cfg.n_head
        self.kv = cfg.n_kv
        self.hd = cfg.d // cfg.n_head
        self.rep = self.h // self.kv
        self.q = nn.Linear(cfg.d, self.h * self.hd, bias=False)
        self.k = nn.Linear(cfg.d, self.kv * self.hd, bias=False)
        self.v = nn.Linear(cfg.d, self.kv * self.hd, bias=False)
        self.o = nn.Linear(self.h * self.hd, cfg.d, bias=False)
        self.qn = RMSNorm(self.hd)
        self.kn = RMSNorm(self.hd)
        self.use_value_res = use_value_res
        if use_value_res:
            # Initialized at 0.5 in logit/linear space -> 0.5 mix, matching dense.py value_residual
            self.lam = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        v0: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, _ = x.shape
        q = self.qn(self.q(x).view(B, T, self.h, self.hd)).transpose(1, 2)
        k = self.kn(self.k(x).view(B, T, self.kv, self.hd)).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        if v0 is None:
            v0_out = v
        else:
            v0_out = v0
            if self.use_value_res:
                lam = self.lam.to(v.dtype)
                v = lam * v + (1.0 - lam) * v0
        q = apply_rope(q, *rope)
        k = apply_rope(k, *rope)
        k_exp = k.repeat_interleave(self.rep, dim=1)
        v_exp = v.repeat_interleave(self.rep, dim=1)
        y = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
        return self.o(y.transpose(1, 2).contiguous().view(B, T, -1)), v0_out


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.gu = nn.Linear(cfg.d, 2 * cfg.ffn, bias=False)
        self.dn = nn.Linear(cfg.ffn, cfg.d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, u = self.gu(x).chunk(2, dim=-1)
        return self.dn(F.silu(g) * u)


class DepthRouter(nn.Module):
    """Block Attention Residuals router (Kimi Team, arXiv:2603.15031).

    Given completed block representations [v_0, ..., v_{k-1}] and current partial block state s_k
    (or [v_0, ..., v_{k-1}] at a block start), normalizes keys per-slot with RMSNorm and computes
    position-local depth softmax weights from a zero-initialized query vector `q_depth` in R^d.
    Strictly causal across sequence position T because each position t only mixes its own depth stack.
    """

    def __init__(self, d: int):
        super().__init__()
        self.key_norm = RMSNorm(d)
        self.q_depth = nn.Parameter(torch.zeros(d, dtype=torch.float32))
        self.scale = d ** -0.5

    def forward(self, states: list[torch.Tensor]) -> torch.Tensor:
        if len(states) == 1:
            return states[0]
        # stacked: [B, T, K, d]
        stacked = torch.stack(states, dim=2)
        keys = self.key_norm(stacked).float()
        scores = torch.einsum("btkd,d->btk", keys, self.q_depth.float()) * self.scale
        weights = F.softmax(scores, dim=-1).to(stacked.dtype)
        return torch.einsum("btk,btkd->btd", weights, stacked)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelCfg, layer_idx: int):
        super().__init__()
        self.n1 = RMSNorm(cfg.d)
        self.n2 = RMSNorm(cfg.d)
        use_vr = (cfg.arm == "value_res") and (layer_idx > 0)
        self.attn = GQAAttention(cfg, use_value_res=use_vr)
        self.mlp = SwiGLU(cfg)
        self.use_attnres = (cfg.arm == "block_attnres")
        if self.use_attnres:
            self.route_attn = DepthRouter(cfg.d)
            self.route_mlp = DepthRouter(cfg.d)

    def forward_standard(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        v0: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        h_attn, v0 = self.attn(self.n1(x), rope, v0)
        x = x + h_attn
        x = x + self.mlp(self.n2(x))
        return x, v0

    def forward_attnres(
        self,
        completed_blocks: list[torch.Tensor],
        partial_block: torch.Tensor | None,
        rope: tuple[torch.Tensor, torch.Tensor],
        v0: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        states_1 = completed_blocks if partial_block is None else [*completed_blocks, partial_block]
        x_in1 = self.route_attn(states_1)
        u_attn, v0 = self.attn(self.n1(x_in1), rope, v0)
        partial_block = u_attn if partial_block is None else (partial_block + u_attn)

        states_2 = [*completed_blocks, partial_block]
        x_in2 = self.route_mlp(states_2)
        u_mlp = self.mlp(self.n2(x_in2))
        partial_block = partial_block + u_mlp
        return partial_block, v0


class NanoMechanismLM(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        if cfg.arm not in ARM_SPECS:
            raise ValueError(f"Unknown arm: {cfg.arm}")
        if cfg.arm == "dense10":
            cfg = ModelCfg(
                arm="dense10",
                vocab=cfg.vocab,
                d=cfg.d,
                layers=cfg.layers + 2,
                n_head=cfg.n_head,
                n_kv=cfg.n_kv,
                ffn=cfg.ffn,
                rope_theta=cfg.rope_theta,
            )
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.layers)])
        self.nf = RMSNorm(cfg.d)
        if cfg.arm == "block_attnres":
            self.route_final = DepthRouter(cfg.d)
        if cfg.arm == "coda_refine":
            # Zero-initialized per-channel residual gate so step-0 forward pass is bit-identical to `dense`.
            self.refine_gate = nn.Parameter(torch.zeros(cfg.d, dtype=torch.float32))
        self._init_weights()

    def effective_layer_passes(self) -> int:
        if self.cfg.arm == "coda_refine":
            return self.cfg.layers + self.cfg.coda_layers
        return self.cfg.layers

    def _init_weights(self) -> None:
        res_scale = 1.0 / math.sqrt(2 * self.cfg.layers)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        for name, p in self.named_parameters():
            if p.ndim == 2 and name != "emb.weight":
                nn.init.normal_(p, mean=0.0, std=0.02)
                if name.endswith((".o.weight", ".dn.weight")):
                    with torch.no_grad():
                        p.mul_(res_scale)

    def forward_logits(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        rope = rope_cache(T, self.cfg.d // self.cfg.n_head, self.cfg.rope_theta, idx.device)
        x0 = self.emb(idx)
        v0: torch.Tensor | None = None

        if self.cfg.arm == "block_attnres":
            completed: list[torch.Tensor] = [x0]
            partial: torch.Tensor | None = None
            blk_sz = self.cfg.attnres_block_layers
            for i, blk in enumerate(self.blocks):
                partial, v0 = blk.forward_attnres(completed, partial, rope, v0)
                if (i + 1) % blk_sz == 0 and (i + 1) < self.cfg.layers:
                    assert partial is not None
                    completed.append(partial)
                    partial = None
            assert partial is not None
            completed.append(partial)
            x = self.route_final(completed)
        else:
            x = x0
            for blk in self.blocks:
                x, v0 = blk.forward_standard(x, rope, v0)
            if self.cfg.arm == "coda_refine":
                x_ref = x
                for blk in self.blocks[-self.cfg.coda_layers :]:
                    x_ref, _ = blk.forward_standard(x_ref, rope, v0)
                gate = self.refine_gate.to(x.dtype)
                x = x + gate * (x_ref - x)

        h = self.nf(x)
        return F.linear(h, self.emb.weight)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.forward_logits(idx)
        if targets is None:
            return logits
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))


def newton_schulz_5_batched(
    g_batch: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-8,
    coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
) -> torch.Tensor:
    """Exact FP32 Frobenius-preconditioned 5-step Newton-Schulz over [K, M, N] batch matching optax.contrib.muon."""
    assert g_batch.ndim == 3
    a, b, c = coeffs
    x = g_batch.float()
    transposed = x.shape[1] > x.shape[2]
    if transposed:
        x = x.transpose(1, 2)
    norms = x.flatten(1).norm(dim=1).view(-1, 1, 1)
    x = x / (norms + eps)
    for _ in range(steps):
        gram = torch.bmm(x, x.transpose(1, 2))
        x = a * x + torch.bmm(b * gram + c * torch.bmm(gram, gram), x)
    if transposed:
        x = x.transpose(1, 2)
    return x


class OptaxParityMuon(torch.optim.Optimizer):
    """Exact PyTorch match to `optax.contrib.muon` (`consistent_rms=0.2`, `nesterov=True`)."""

    def __init__(
        self,
        named_params: Iterable[tuple[str, nn.Parameter]],
        lr: float = 4e-3,
        wd: float = 0.1,
        muon_beta: float = 0.95,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        consistent_rms: float = 0.2,
        ns_steps: int = 5,
    ):
        params_list = []
        self.param_is_muon: dict[int, bool] = {}
        for name, p in named_params:
            if not p.requires_grad:
                continue
            params_list.append(p)
            # In train_jax.py: 2D matrices inside `blocks` use Muon; `emb` and all 1D tensors use AdamW.
            is_muon = (p.ndim == 2) and ("blocks." in name)
            self.param_is_muon[id(p)] = is_muon
        defaults = dict(
            lr=lr,
            wd=wd,
            muon_beta=muon_beta,
            adam_betas=adam_betas,
            eps=eps,
            consistent_rms=consistent_rms,
            ns_steps=ns_steps,
        )
        super().__init__(params_list, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            wd: float = group["wd"]
            beta: float = group["muon_beta"]
            b1, b2 = group["adam_betas"]
            eps: float = group["eps"]
            c_rms: float = group["consistent_rms"]
            ns_steps: int = group["ns_steps"]

            muon_buckets: dict[tuple[int, int], list[tuple[nn.Parameter, torch.Tensor]]] = {}
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    if self.param_is_muon[id(p)]:
                        st["mu"] = torch.zeros_like(p, dtype=torch.float32)
                    else:
                        st["m"] = torch.zeros_like(p, dtype=torch.float32)
                        st["v"] = torch.zeros_like(p, dtype=torch.float32)

                st["step"] += 1
                t = st["step"]

                if self.param_is_muon[id(p)]:
                    mu = st["mu"]
                    mu.mul_(beta).add_(g, alpha=1.0 - beta)
                    mu_corr = mu / (1.0 - (beta ** (t + 1)))
                    g_corr = g / (1.0 - (beta ** t))
                    mu_hat = beta * mu_corr + (1.0 - beta) * g_corr
                    muon_buckets.setdefault((p.shape[0], p.shape[1]), []).append((p, mu_hat))
                else:
                    m = st["m"]
                    v = st["v"]
                    m.mul_(b1).add_(g, alpha=1.0 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                    m_hat = b1 * (m / (1.0 - (b1 ** (t + 1)))) + (1.0 - b1) * (g / (1.0 - (b1 ** t)))
                    v_hat = v / (1.0 - (b2 ** t))
                    upd = m_hat / (v_hat.sqrt() + eps) + wd * p.float()
                    p.add_(upd.to(p.dtype), alpha=-lr)

            for (r_dim, c_dim), items in muon_buckets.items():
                stacked = torch.stack([mh for _, mh in items], dim=0)
                orthos = newton_schulz_5_batched(stacked, steps=ns_steps, eps=eps)
                shape_scale = c_rms * math.sqrt(max(r_dim, c_dim))
                for idx_p, (p, _) in enumerate(items):
                    upd = orthos[idx_p] * shape_scale + wd * p.float()
                    p.add_(upd.to(p.dtype), alpha=-lr)

        return loss
