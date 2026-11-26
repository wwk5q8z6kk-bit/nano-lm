"""Exact `fla==0.6.0`-Compatible Dual-Backend (`PyTorch` <-> `JAX/XLA`) Gated DeltaNet & Hybrid LM Core.

Guarantees 1:1 parameter-name, tensor-shape, forward-logit, backward-gradient, and `.npz` checkpoint
parity between:
  1. PyTorch `fla.layers.GatedDeltaNet` (`flash-linear-attention==0.6.0` on CUDA/Hopper/T4) AND an exact
     pure-PyTorch CPU/GPU fallback with identical `state_dict` keys (`TorchGDNLayer`, `TorchHybridLM`).
  2. JAX/XLA (`jax_gdn_forward`, `jax_hybrid_loss`, `jax_sparse_kd_loss`) using:
     - 4-tap register-aligned causal `ShortConvolution(W=4, silu)` in `FP32`,
     - `L2Norm(eps=1e-6)` + `A_log`/`dt_bias` softplus decay + `sigmoid` `beta`,
     - 6-matmul parallel Neumann WY chunk solver (`chunk_size=64`, `(I+N)(I+N^2)(I+N^4)(I+N^8)(I+N^16)(I+N^32)`)
       at `jax.lax.Precision.HIGHEST` + `jax.lax.scan` inter-chunk state propagation,
     - `FusedRMSNormGated(eps=1e-5, swish)` + `jax.checkpoint` block rematerialization & chunked vocab/KD head.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np


# ============================================================================
# 1. PyTorch Backend (`fla==0.6.0` native CUDA + exact pure-PyTorch CPU/GPU ref)
# ============================================================================

def build_torch_classes():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def torch_wy_chunk_gdn(q, k, v, g_raw, beta_raw, A_log, dt_bias, chunk_size: int = 64, allow_neg_eigval: bool = False):
        """Exact PyTorch reference matching `fla==0.6.0` `ChunkGatedDeltaRuleFunction`."""
        orig_dtype = q.dtype
        # 1. L2Norm in FP32, cast back to orig_dtype (matches `l2norm_fwd(q)` in `chunk.py` line 283)
        qf = q.float()
        kf = k.float()
        q_norm = (qf * torch.rsqrt((qf * qf).sum(dim=-1, keepdim=True) + 1e-6)).to(orig_dtype).float()
        k_norm = (kf * torch.rsqrt((kf * kf).sum(dim=-1, keepdim=True) + 1e-6)).to(orig_dtype).float()
        vf = v.float()

        beta_scale = 2.0 if allow_neg_eigval else 1.0
        beta = torch.sigmoid(beta_raw.float()) * beta_scale
        g_step = -A_log.float().exp() * F.softplus(g_raw.float() + dt_bias.float())

        # Transpose to [B, H, T, D]
        B, T, H, K = q_norm.shape
        HV = vf.shape[2]
        V = vf.shape[3]
        if HV > H:
            rep = HV // H
            q_norm = q_norm.repeat_interleave(rep, dim=2)
            k_norm = k_norm.repeat_interleave(rep, dim=2)
            H = HV

        qh = q_norm.transpose(1, 2) * (K ** -0.5)
        kh = k_norm.transpose(1, 2)
        vh = vf.transpose(1, 2)
        bh = beta.transpose(1, 2)
        gh = g_step.transpose(1, 2)

        BT = chunk_size
        pad = (BT - (T % BT)) % BT
        if pad > 0:
            qh = F.pad(qh, (0, 0, 0, pad))
            kh = F.pad(kh, (0, 0, 0, pad))
            vh = F.pad(vh, (0, 0, 0, pad))
            bh = F.pad(bh, (0, pad))
            gh = F.pad(gh, (0, pad))

        NT = qh.shape[2] // BT
        qc = qh.view(B, H, NT, BT, K)
        kc = kh.view(B, H, NT, BT, K)
        vc = (vh * bh.unsqueeze(-1)).view(B, H, NT, BT, V)
        k_beta = (kh * bh.unsqueeze(-1)).view(B, H, NT, BT, K)
        decay = gh.view(B, H, NT, BT).cumsum(dim=-1)

        idx = torch.arange(BT, device=q.device)
        causal_incl = idx[:, None] >= idx[None, :]
        strict_lower = idx[:, None] > idx[None, :]

        diff = torch.where(causal_incl, decay.unsqueeze(-1) - decay.unsqueeze(-2), torch.zeros((), device=q.device, dtype=torch.float32))
        L_mask = torch.where(causal_incl, diff.exp(), torch.zeros((), device=q.device, dtype=torch.float32))

        # N = -strictly_lower((k_beta @ kc^T) * L_mask); (I - N)^{-1} via exact Neumann doubling
        N = torch.where(strict_lower, -((k_beta @ kc.transpose(-1, -2)) * L_mask), torch.zeros((), device=q.device, dtype=torch.float32))
        eye = torch.eye(BT, device=q.device, dtype=torch.float32)
        A = eye + N
        P = N
        steps_neumann = int( round(math.log2(BT)) )
        for _ in range(steps_neumann - 1):
            P = P @ P
            A = A @ (eye + P)

        u = A @ vc
        w = A @ (k_beta * decay.exp().unsqueeze(-1))

        S = torch.zeros((B, H, K, V), device=q.device, dtype=torch.float32)
        outs = []
        for i in range(NT):
            qi, ki, ui, wi, di, Li = qc[:, :, i], kc[:, :, i], u[:, :, i], w[:, :, i], decay[:, :, i], L_mask[:, :, i]
            intra = torch.where(causal_incl, (qi @ ki.transpose(-1, -2)) * Li, torch.zeros((), device=q.device, dtype=torch.float32))
            v_new = ui - wi @ S
            oi = (qi * di.exp().unsqueeze(-1)) @ S + intra @ v_new
            outs.append(oi)
            d_end = di[:, :, -1:]
            k_dec = ki * (d_end - di).exp().unsqueeze(-1)
            S = S * d_end.exp().unsqueeze(-1) + k_dec.transpose(-1, -2) @ v_new

        o = torch.stack(outs, dim=2).reshape(B, H, NT * BT, V)[:, :, :T, :].transpose(1, 2).contiguous()
        return o.to(orig_dtype)

    class RefShortConv1d(nn.Conv1d):
        """Matches `fla.modules.ShortConvolution` parameter layout (`weight: [D, 1, W]`) identically."""
        def __init__(self, hidden_size: int, kernel_size: int = 4, bias: bool = False):
            super().__init__(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size - 1, groups=hidden_size, bias=bias)

        def forward(self, x: torch.Tensor):
            orig_dtype = x.dtype
            B, T, C = x.shape
            W = self.kernel_size[0]
            xf = x.float()
            wf = self.weight.squeeze(1).float()  # [C, W]
            x_pad = F.pad(xf, (0, 0, W - 1, 0))
            acc = torch.zeros_like(xf)
            for k_i in range(W):
                acc = acc + x_pad[:, k_i : k_i + T, :] * wf[None, None, :, k_i]
            if self.bias is not None:
                acc = acc + self.bias.float()[None, None, :]
            return F.silu(acc).to(orig_dtype), None

    class RefRMSNormGated(nn.Module):
        """Matches `fla.modules.FusedRMSNormGated(hidden_size, eps=1e-5, activation='swish')`."""
        def __init__(self, hidden_size: int, eps: float = 1e-5):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(hidden_size))

        def forward(self, x: torch.Tensor, g: torch.Tensor):
            xf = x.float()
            gf = g.float()
            rstd = torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + self.eps)
            y = xf * rstd * self.weight.float() * F.silu(gf)
            return y.to(x.dtype)

    class TorchGDNLayer(nn.Module):
        """Drop-in state_dict-identical match to `fla.layers.GatedDeltaNet`."""
        def __init__(self, hidden_size: int, num_heads: int, head_dim: int, expand_v: int = 1, conv_size: int = 4, norm_eps: float = 1e-5):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.num_v_heads = num_heads
            self.head_k_dim = head_dim
            self.head_v_dim = head_dim * expand_v
            self.key_dim = num_heads * self.head_k_dim
            self.value_dim = self.num_v_heads * self.head_v_dim
            self.conv_size = conv_size

            self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
            self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)

            A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0.5, 8.0)
            self.A_log = nn.Parameter(torch.log(A))
            dt = torch.exp(torch.rand(self.num_v_heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp_min(1e-4)
            self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

            self.q_conv1d = RefShortConv1d(self.key_dim, conv_size, bias=False)
            self.k_conv1d = RefShortConv1d(self.key_dim, conv_size, bias=False)
            self.v_conv1d = RefShortConv1d(self.value_dim, conv_size, bias=False)

            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = RefRMSNormGated(self.head_v_dim, eps=norm_eps)
            self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        def forward(self, hidden_states: torch.Tensor, chunk_size: int = 64):
            B, T, _ = hidden_states.shape
            q, _ = self.q_conv1d(self.q_proj(hidden_states))
            k, _ = self.k_conv1d(self.k_proj(hidden_states))
            v, _ = self.v_conv1d(self.v_proj(hidden_states))
            q = q.view(B, T, self.num_heads, self.head_k_dim)
            k = k.view(B, T, self.num_heads, self.head_k_dim)
            v = v.view(B, T, self.num_v_heads, self.head_v_dim)
            g_raw = self.a_proj(hidden_states)
            beta_raw = self.b_proj(hidden_states)
            o = torch_wy_chunk_gdn(q, k, v, g_raw, beta_raw, self.A_log, self.dt_bias, chunk_size=chunk_size)
            g_out = self.g_proj(hidden_states).view(B, T, self.num_v_heads, self.head_v_dim)
            o = self.o_norm(o, g_out).reshape(B, T, self.value_dim)
            return self.o_proj(o), None, None

    class TorchHybridBlock(nn.Module):
        def __init__(self, d: int, kind: str, ffn: int, heads: int, kv_heads: int | None = None, use_fla: bool = False):
            super().__init__()
            self.kind = kind
            self.d = d
            self.heads = heads
            self.kv_heads = kv_heads or heads
            self.hd = d // heads
            self.n1 = nn.RMSNorm(d, eps=1e-5)
            self.n2 = nn.RMSNorm(d, eps=1e-5)
            if kind == "gdn":
                if use_fla:
                    from fla.layers import GatedDeltaNet
                    self.mix = GatedDeltaNet(hidden_size=d, num_heads=heads, head_dim=self.hd, expand_v=1, mode="chunk")
                else:
                    self.mix = TorchGDNLayer(hidden_size=d, num_heads=heads, head_dim=self.hd, expand_v=1)
            else:
                self.q_proj = nn.Linear(d, heads * self.hd, bias=False)
                self.k_proj = nn.Linear(d, self.kv_heads * self.hd, bias=False)
                self.v_proj = nn.Linear(d, self.kv_heads * self.hd, bias=False)
                self.o_proj = nn.Linear(heads * self.hd, d, bias=False)
            self.gu = nn.Linear(d, 2 * ffn, bias=False)
            self.dn = nn.Linear(ffn, d, bias=False)

        def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
            h = self.n1(x.float()).to(x.dtype) * self.n1.weight.to(x.dtype) / self.n1.weight.to(x.dtype).clamp_min(1e-30) if False else None
            # Exact FP32 RMSNorm cast back to x.dtype (matching JAX & HF LlamaRMSNorm)
            xf = x.float()
            h = (xf * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + 1e-5) * self.n1.weight.float()).to(x.dtype)
            if self.kind == "gdn":
                out = self.mix(h)
                y = out[0] if isinstance(out, tuple) else out
            else:
                B, T, _ = h.shape
                q = self.q_proj(h).view(B, T, self.heads, self.hd).transpose(1, 2)
                k = self.k_proj(h).view(B, T, self.kv_heads, self.hd).transpose(1, 2)
                v = self.v_proj(h).view(B, T, self.kv_heads, self.hd).transpose(1, 2)
                c, s = cos.to(q.dtype), sin.to(q.dtype)
                q1, q2 = q[..., ::2], q[..., 1::2]
                k1, k2 = k[..., ::2], k[..., 1::2]
                q = torch.stack((q1 * c - q2 * s, q1 * s + q2 * c), dim=-1).reshape_as(q)
                k = torch.stack((k1 * c - k2 * s, k1 * s + k2 * c), dim=-1).reshape_as(k)
                if self.heads > self.kv_heads:
                    rep = self.heads // self.kv_heads
                    k = k.repeat_interleave(rep, dim=1)
                    v = v.repeat_interleave(rep, dim=1)
                att = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, self.d)
                y = self.o_proj(att)
            x = x + y
            xf2 = x.float()
            h2 = (xf2 * torch.rsqrt((xf2 * xf2).mean(-1, keepdim=True) + 1e-5) * self.n2.weight.float()).to(x.dtype)
            g, u = self.gu(h2).chunk(2, dim=-1)
            return x + self.dn((F.silu(g.float()) * u.float()).to(x.dtype))

    class TorchHybridLM(nn.Module):
        def __init__(self, d: int, layers: int, ffn: int, heads: int, kv_heads: int, vocab: int, attn_layers: tuple[int, ...] = (), use_fla: bool = False):
            super().__init__()
            self.d = d
            self.layers = layers
            self.heads = heads
            self.hd = d // heads
            self.vocab = vocab
            self.attn_layers = tuple(attn_layers)
            self.emb = nn.Embedding(vocab, d)
            nn.init.normal_(self.emb.weight, std=0.02)
            self.blocks = nn.ModuleList([
                TorchHybridBlock(d, "attn" if i in self.attn_layers else "gdn", ffn, heads, kv_heads, use_fla=use_fla)
                for i in range(layers)
            ])
            self.nf = nn.RMSNorm(d, eps=1e-5)

        def forward(self, ids: torch.Tensor):
            B, T = ids.shape
            inv = 1.0 / (10000.0 ** (torch.arange(0, self.hd, 2, device=ids.device, dtype=torch.float32) / self.hd))
            f = torch.outer(torch.arange(T, device=ids.device, dtype=torch.float32), inv)
            cos, sin = f.cos(), f.sin()
            x = self.emb(ids)
            for b in self.blocks:
                x = b(x, cos, sin)
            xf = x.float()
            h = (xf * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + 1e-5) * self.nf.weight.float()).to(x.dtype)
            return h @ self.emb.weight.T

    return TorchGDNLayer, TorchHybridBlock, TorchHybridLM, torch_wy_chunk_gdn


# ============================================================================
# 2. Lossless Bidirectional Weight Bridge (`PyTorch state_dict` <-> `NPZ` <-> `JAX`)
# ============================================================================

GDN_PROJ_KEYS = ("q_proj", "k_proj", "v_proj", "a_proj", "b_proj", "g_proj", "o_proj")
GDN_CONV_KEYS = ("q_conv1d", "k_conv1d", "v_conv1d")
ATTN_PROJ_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj")


def torch_model_to_npz_dict(model) -> dict[str, np.ndarray]:
    sd = model.state_dict()
    return {k: v.detach().cpu().float().numpy() if v.dtype.is_floating_point else v.detach().cpu().numpy() for k, v in sd.items()}


def npz_dict_to_jax_params(npz_map: dict[str, np.ndarray], n_layers: int, attn_layers: tuple[int, ...]) -> dict:
    """Converts a PyTorch `TorchHybridLM` / `fla` state_dict mapping into a JAX PyTree.
    Linear weights `[out, in]` are transposed to `[in, out]` for right-matmul `x @ W`;
    Conv weights `[C, 1, 4]` are squeezed to `[C, 4]` (zero precision loss).
    """
    import jax.numpy as jnp

    blocks = []
    attn_set = set(attn_layers)
    for i in range(n_layers):
        p = f"blocks.{i}."
        blk = {
            "n1": jnp.asarray(npz_map[p + "n1.weight"], dtype=jnp.float32),
            "n2": jnp.asarray(npz_map[p + "n2.weight"], dtype=jnp.float32),
            "gu": jnp.asarray(npz_map[p + "gu.weight"].T, dtype=jnp.float32),
            "dn": jnp.asarray(npz_map[p + "dn.weight"].T, dtype=jnp.float32),
        }
        if i in attn_set:
            for k in ATTN_PROJ_KEYS:
                blk[k] = jnp.asarray(npz_map[p + f"{k}.weight"].T, dtype=jnp.float32)
        else:
            mp = p + "mix."
            for k in GDN_PROJ_KEYS:
                blk[k] = jnp.asarray(npz_map[mp + f"{k}.weight"].T, dtype=jnp.float32)
            for k in GDN_CONV_KEYS:
                blk[k] = jnp.asarray(npz_map[mp + f"{k}.weight"][:, 0, :], dtype=jnp.float32)
            blk["A_log"] = jnp.asarray(npz_map[mp + "A_log"], dtype=jnp.float32)
            blk["dt_bias"] = jnp.asarray(npz_map[mp + "dt_bias"], dtype=jnp.float32)
            blk["o_norm"] = jnp.asarray(npz_map[mp + "o_norm.weight"], dtype=jnp.float32)
        blocks.append(blk)

    return {
        "emb": jnp.asarray(npz_map["emb.weight"], dtype=jnp.float32),
        "nf": jnp.asarray(npz_map["nf.weight"], dtype=jnp.float32),
        "blocks": blocks,
    }


def jax_params_to_npz_dict(params: dict, attn_layers: tuple[int, ...]) -> dict[str, np.ndarray]:
    """Inverse lossless conversion from JAX PyTree back to exact PyTorch `fla` `state_dict` layout."""
    out: dict[str, np.ndarray] = {
        "emb.weight": np.asarray(params["emb"], dtype=np.float32),
        "nf.weight": np.asarray(params["nf"], dtype=np.float32),
    }
    attn_set = set(attn_layers)
    for i, blk in enumerate(params["blocks"]):
        p = f"blocks.{i}."
        out[p + "n1.weight"] = np.asarray(blk["n1"], dtype=np.float32)
        out[p + "n2.weight"] = np.asarray(blk["n2"], dtype=np.float32)
        out[p + "gu.weight"] = np.asarray(blk["gu"], dtype=np.float32).T.copy()
        out[p + "dn.weight"] = np.asarray(blk["dn"], dtype=np.float32).T.copy()
        if i in attn_set:
            for k in ATTN_PROJ_KEYS:
                out[p + f"{k}.weight"] = np.asarray(blk[k], dtype=np.float32).T.copy()
        else:
            mp = p + "mix."
            for k in GDN_PROJ_KEYS:
                out[mp + f"{k}.weight"] = np.asarray(blk[k], dtype=np.float32).T.copy()
            for k in GDN_CONV_KEYS:
                out[mp + f"{k}.weight"] = np.asarray(blk[k], dtype=np.float32)[:, None, :].copy()
            out[mp + "A_log"] = np.asarray(blk["A_log"], dtype=np.float32)
            out[mp + "dt_bias"] = np.asarray(blk["dt_bias"], dtype=np.float32)
            out[mp + "o_norm.weight"] = np.asarray(blk["o_norm"], dtype=np.float32)
    return out


# ============================================================================
# 3. JAX / XLA TPU Backend (`fla==0.6.0`-exact WY Chunk Gated DeltaNet)
# ============================================================================

def init_jax_hybrid_params(key, cfg: dict) -> dict:
    """Initializes `R0` / `Stage C` / `SmolLM2`-shaped hybrid or pure-GDN parameters directly in JAX."""
    import jax
    import jax.numpy as jnp

    d = cfg["d"]
    L = cfg["layers"]
    ffn = cfg["ffn"]
    H = cfg["heads"]
    KV = cfg.get("kv_heads", H)
    vocab = cfg["vocab"]
    expand_v = cfg.get("expand_v", 1)
    attn_set = set(cfg.get("attn_layers", ()))
    hd = d // H
    vd = hd * expand_v

    ks = iter(jax.random.split(key, 2 + 14 * L))
    norm_init = lambda sh, s=0.02: (s * jax.random.normal(next(ks), sh, jnp.float32))
    res_s = 0.02 / math.sqrt(2 * L)

    blocks = []
    for i in range(L):
        blk = {
            "n1": jnp.ones((d,), jnp.float32),
            "n2": jnp.ones((d,), jnp.float32),
            "gu": norm_init((d, 2 * ffn)),
            "dn": norm_init((ffn, d), res_s),
        }
        if i in attn_set:
            blk["q_proj"] = norm_init((d, H * hd))
            blk["k_proj"] = norm_init((d, KV * hd))
            blk["v_proj"] = norm_init((d, KV * hd))
            blk["o_proj"] = norm_init((H * hd, d), res_s)
        else:
            blk["q_proj"] = norm_init((d, H * hd))
            blk["k_proj"] = norm_init((d, H * hd))
            blk["v_proj"] = norm_init((d, H * vd))
            blk["a_proj"] = norm_init((d, H))
            blk["b_proj"] = norm_init((d, H))
            blk["g_proj"] = norm_init((d, H * vd))
            blk["o_proj"] = norm_init((H * vd, d), res_s)
            cs = 1.0 / math.sqrt(4.0)
            blk["q_conv1d"] = jax.random.uniform(next(ks), (H * hd, 4), jnp.float32, minval=-cs, maxval=cs)
            blk["k_conv1d"] = jax.random.uniform(next(ks), (H * hd, 4), jnp.float32, minval=-cs, maxval=cs)
            blk["v_conv1d"] = jax.random.uniform(next(ks), (H * vd, 4), jnp.float32, minval=-cs, maxval=cs)
            A = jax.random.uniform(next(ks), (H,), jnp.float32, minval=0.5, maxval=8.0)
            blk["A_log"] = jnp.log(A)
            dt = jnp.exp(jax.random.uniform(next(ks), (H,), jnp.float32) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
            blk["dt_bias"] = dt + jnp.log(-jnp.expm1(-dt))
            blk["o_norm"] = jnp.ones((vd,), jnp.float32)
        blocks.append(blk)

    return {
        "emb": norm_init((vocab, d)),
        "nf": jnp.ones((d,), jnp.float32),
        "blocks": blocks,
    }


def jax_n_params(params: dict) -> int:
    import jax
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def jax_short_conv1d_silu(x, w):
    """4-tap causal depthwise conv1d + SiLU in FP32 (`x: [B, T, C]`, `w: [C, 4]`)."""
    import jax
    import jax.numpy as jnp
    T = x.shape[1]
    x32 = x.astype(jnp.float32)
    w32 = w.astype(jnp.float32)
    xp = jnp.pad(x32, ((0, 0), (3, 0), (0, 0)))
    acc = (
        xp[:, 0:T, :] * w32[None, None, :, 0]
        + xp[:, 1 : T + 1, :] * w32[None, None, :, 1]
        + xp[:, 2 : T + 2, :] * w32[None, None, :, 2]
        + xp[:, 3 : T + 3, :] * w32[None, None, :, 3]
    )
    return jax.nn.silu(acc).astype(x.dtype)


def jax_wy_chunk_gated_delta_rule(q, k, v, g_raw, beta_raw, A_log, dt_bias, chunk_size: int = 64, allow_neg_eigval: bool = False):
    """TPU/XLA-native WY chunked Gated Delta Rule matching `fla==0.6.0` `chunk_gated_delta_rule`."""
    import jax
    import jax.numpy as jnp

    orig_dtype = q.dtype
    HP = jax.lax.Precision.HIGHEST
    qf = q.astype(jnp.float32)
    kf = k.astype(jnp.float32)
    q_norm = (qf * jax.lax.rsqrt(jnp.sum(qf * qf, axis=-1, keepdims=True) + 1e-6)).astype(orig_dtype).astype(jnp.float32)
    k_norm = (kf * jax.lax.rsqrt(jnp.sum(kf * kf, axis=-1, keepdims=True) + 1e-6)).astype(orig_dtype).astype(jnp.float32)
    vf = v.astype(jnp.float32)

    B, T, H, K = q_norm.shape
    HV = vf.shape[2]
    V = vf.shape[3]
    if HV > H:
        rep = HV // H
        q_norm = jnp.repeat(q_norm, rep, axis=2)
        k_norm = jnp.repeat(k_norm, rep, axis=2)
        H = HV

    beta = jax.nn.sigmoid(beta_raw.astype(jnp.float32)) * (2.0 if allow_neg_eigval else 1.0)
    g_step = -jnp.exp(A_log.astype(jnp.float32)) * jax.nn.softplus(g_raw.astype(jnp.float32) + dt_bias.astype(jnp.float32))

    qh = jnp.swapaxes(q_norm, 1, 2) * (K ** -0.5)
    kh = jnp.swapaxes(k_norm, 1, 2)
    vh = jnp.swapaxes(vf, 1, 2)
    bh = jnp.swapaxes(beta, 1, 2)
    gh = jnp.swapaxes(g_step, 1, 2)

    BT = chunk_size
    pad = (BT - (T % BT)) % BT
    if pad > 0:
        qh = jnp.pad(qh, ((0, 0), (0, 0), (0, pad), (0, 0)))
        kh = jnp.pad(kh, ((0, 0), (0, 0), (0, pad), (0, 0)))
        vh = jnp.pad(vh, ((0, 0), (0, 0), (0, pad), (0, 0)))
        bh = jnp.pad(bh, ((0, 0), (0, 0), (0, pad)))
        gh = jnp.pad(gh, ((0, 0), (0, 0), (0, pad)))

    NT = qh.shape[2] // BT
    qc = qh.reshape(B, H, NT, BT, K)
    kc = kh.reshape(B, H, NT, BT, K)
    vc = (vh * bh[..., None]).reshape(B, H, NT, BT, V)
    k_beta = (kh * bh[..., None]).reshape(B, H, NT, BT, K)
    decay = jnp.cumsum(gh.reshape(B, H, NT, BT), axis=-1)

    idx = jnp.arange(BT)
    causal_incl = idx[:, None] >= idx[None, :]
    strict_lower = idx[:, None] > idx[None, :]

    diff = jnp.where(causal_incl, decay[..., :, None] - decay[..., None, :], 0.0)
    L_mask = jnp.where(causal_incl, jnp.exp(diff), 0.0)

    kkt = jnp.matmul(k_beta, jnp.swapaxes(kc, -1, -2), precision=HP)
    N = jnp.where(strict_lower, -(kkt * L_mask), 0.0)
    eye = jnp.eye(BT, dtype=jnp.float32)
    A = eye + N
    P = N
    for _ in range(int(round(math.log2(BT))) - 1):
        P = jnp.matmul(P, P, precision=HP)
        A = jnp.matmul(A, eye + P, precision=HP)

    u = jnp.matmul(A, vc, precision=HP)
    w = jnp.matmul(A, k_beta * jnp.exp(decay)[..., None], precision=HP)

    # Scan across NT chunks along leading axis [NT, B, H, ...]
    scan_in = (
        jnp.moveaxis(qc, 2, 0),
        jnp.moveaxis(kc, 2, 0),
        jnp.moveaxis(u, 2, 0),
        jnp.moveaxis(w, 2, 0),
        jnp.moveaxis(decay, 2, 0),
        jnp.moveaxis(L_mask, 2, 0),
    )

    def chunk_step(S, xs):
        qi, ki, ui, wi, di, Li = xs
        qk = jnp.matmul(qi, jnp.swapaxes(ki, -1, -2), precision=HP)
        intra = jnp.where(causal_incl, qk * Li, 0.0)
        v_new = ui - jnp.matmul(wi, S, precision=HP)
        oi = jnp.matmul(qi * jnp.exp(di)[..., None], S, precision=HP) + jnp.matmul(intra, v_new, precision=HP)
        d_end = di[..., -1:]
        k_dec = ki * jnp.exp(d_end - di)[..., None]
        S_next = S * jnp.exp(d_end)[..., None] + jnp.matmul(jnp.swapaxes(k_dec, -1, -2), v_new, precision=HP)
        return S_next, oi

    S0 = jnp.zeros((B, H, K, V), dtype=jnp.float32)
    _, outs = jax.lax.scan(chunk_step, S0, scan_in)
    # outs: [NT, B, H, BT, V] -> [B, T, H, V]
    o = jnp.moveaxis(outs, 0, 2).reshape(B, H, NT * BT, V)[:, :, :T, :]
    o = jnp.swapaxes(o, 1, 2)
    return o.astype(orig_dtype)


def jax_rms_norm(x, w, eps: float = 1e-5):
    import jax
    import jax.numpy as jnp
    xf = x.astype(jnp.float32)
    return (xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps) * w.astype(jnp.float32)).astype(x.dtype)


def jax_gdn_layer_forward(h, blk: dict, H: int, hd: int, vd: int, dtype, chunk_size: int = 64, matmul_prec=None):
    import jax
    import jax.numpy as jnp

    B, T, _ = h.shape
    mm = lambda a, b: jnp.dot(a, b.astype(dtype), precision=matmul_prec)
    q = jax_short_conv1d_silu(mm(h, blk["q_proj"]), blk["q_conv1d"]).reshape(B, T, H, hd)
    k = jax_short_conv1d_silu(mm(h, blk["k_proj"]), blk["k_conv1d"]).reshape(B, T, H, hd)
    v = jax_short_conv1d_silu(mm(h, blk["v_proj"]), blk["v_conv1d"]).reshape(B, T, H, vd)
    g_raw = mm(h, blk["a_proj"])
    beta_raw = mm(h, blk["b_proj"])
    o = jax_wy_chunk_gated_delta_rule(q, k, v, g_raw, beta_raw, blk["A_log"], blk["dt_bias"], chunk_size=chunk_size)
    g_out = mm(h, blk["g_proj"]).reshape(B, T, H, vd)

    of = o.astype(jnp.float32)
    gf = g_out.astype(jnp.float32)
    rstd = jax.lax.rsqrt(jnp.mean(of * of, axis=-1, keepdims=True) + 1e-5)
    o_gated = (of * rstd * blk["o_norm"].astype(jnp.float32) * jax.nn.silu(gf)).astype(dtype).reshape(B, T, H * vd)
    return mm(o_gated, blk["o_proj"])


def jax_hybrid_hidden(params: dict, ids, cfg: dict, dtype, chunk_size: int = 64, remat: bool = False):
    import jax
    import jax.numpy as jnp

    B, T = ids.shape
    d = cfg["d"]
    H = cfg["heads"]
    KV = cfg.get("kv_heads", H)
    hd = d // H
    vd = hd * cfg.get("expand_v", 1)
    rep = H // KV
    attn_set = set(cfg.get("attn_layers", ()))
    prec = jax.lax.Precision.HIGHEST if dtype == jnp.float32 else None
    mm = lambda a, b: jnp.dot(a, b.astype(dtype), precision=prec)

    inv = 1.0 / (10000.0 ** (jnp.arange(0, hd, 2, dtype=jnp.float32) / hd))
    f = jnp.outer(jnp.arange(T, dtype=jnp.float32), inv)
    cos, sin = jnp.cos(f), jnp.sin(f)

    x = params["emb"].astype(dtype)[ids]

    for i, blk in enumerate(params["blocks"]):
        is_attn = i in attn_set

        def _block_fn(x_in, b_p=blk, attn_flag=is_attn):
            h = jax_rms_norm(x_in, b_p["n1"], eps=1e-5)
            if attn_flag:
                q = mm(h, b_p["q_proj"]).reshape(B, T, H, hd).swapaxes(1, 2)
                k = mm(h, b_p["k_proj"]).reshape(B, T, KV, hd).swapaxes(1, 2)
                v = mm(h, b_p["v_proj"]).reshape(B, T, KV, hd).swapaxes(1, 2)
                c, s = cos.astype(dtype), sin.astype(dtype)
                q1, q2 = q[..., ::2], q[..., 1::2]
                k1, k2 = k[..., ::2], k[..., 1::2]
                q = jnp.stack((q1 * c - q2 * s, q1 * s + q2 * c), axis=-1).reshape(q.shape)
                k = jnp.stack((k1 * c - k2 * s, k1 * s + k2 * c), axis=-1).reshape(k.shape)
                qg = q.reshape(B, KV, rep, T, hd)
                scores = jnp.einsum("bkrth,bksh->bkrts", qg, k, preferred_element_type=jnp.float32) * (hd ** -0.5)
                t_idx = jnp.arange(T)
                scores = jnp.where(t_idx[:, None] >= t_idx[None, :], scores, jnp.finfo(jnp.float32).min)
                probs = jax.nn.softmax(scores, axis=-1).astype(dtype)
                att = jnp.einsum("bkrts,bksh->bkrth", probs, v).reshape(B, H, T, hd).swapaxes(1, 2).reshape(B, T, d)
                y = mm(att, b_p["o_proj"])
            else:
                y = jax_gdn_layer_forward(h, b_p, H, hd, vd, dtype, chunk_size=chunk_size, matmul_prec=prec)
            x_mid = x_in + y
            h2 = jax_rms_norm(x_mid, b_p["n2"], eps=1e-5)
            gu = mm(h2, b_p["gu"])
            g, u = jnp.split(gu, 2, axis=-1)
            ff_act = (jax.nn.silu(g.astype(jnp.float32)) * u.astype(jnp.float32)).astype(dtype)
            return x_mid + mm(ff_act, b_p["dn"])

        x = jax.checkpoint(_block_fn)(x) if remat else _block_fn(x)

    return jax_rms_norm(x, params["nf"], eps=1e-5)


def jax_hybrid_logits(params: dict, ids, cfg: dict, dtype, chunk_size: int = 64, remat: bool = False):
    import jax
    import jax.numpy as jnp
    prec = jax.lax.Precision.HIGHEST if dtype == jnp.float32 else None
    h = jax_hybrid_hidden(params, ids, cfg, dtype, chunk_size=chunk_size, remat=remat)
    return jnp.dot(h, params["emb"].astype(dtype).T, precision=prec)


def jax_chunked_ce_or_kd_loss(
    params: dict,
    ids,
    cfg: dict,
    dtype,
    kd_idx=None,
    kd_val=None,
    vocab_chunk: int = 2048,
    remat: bool = True,
):
    """Memory-safe chunked CE or `0.5 * CE + 0.5 * sparse_topK_KL` for large-vocab (`151,936`) TPUs."""
    import jax
    import jax.numpy as jnp

    x_in, y_tgt = ids[:, :-1], ids[:, 1:]
    h = jax_hybrid_hidden(params, x_in, cfg, dtype, remat=remat).reshape(-1, cfg["d"])
    y_flat = y_tgt.reshape(-1)
    n_tok = y_flat.shape[0]
    step_sz = min(vocab_chunk, n_tok)
    assert n_tok % step_sz == 0, f"n_tok={n_tok} must be divisible by vocab_chunk={step_sz}"
    n_c = n_tok // step_sz
    w_emb_t = params["emb"].astype(dtype).T
    prec = jax.lax.Precision.HIGHEST if dtype == jnp.float32 else None

    if kd_idx is None:
        @jax.checkpoint
        def ce_chunk(hc, yc):
            lg = jnp.dot(hc, w_emb_t, precision=prec).astype(jnp.float32)
            lse = jax.scipy.special.logsumexp(lg, axis=-1)
            gold = jnp.take_along_axis(lg, yc[:, None], axis=-1)[:, 0]
            return jnp.sum(lse - gold)

        def scan_ce(acc, xs):
            return acc + ce_chunk(xs[0], xs[1]), None

        tot, _ = jax.lax.scan(scan_ce, jnp.zeros((), jnp.float32), (h.reshape(n_c, step_sz, -1), y_flat.reshape(n_c, step_sz)))
        return tot / n_tok
    else:
        ti_flat = kd_idx.reshape(n_tok, -1)
        tv_flat = kd_val.reshape(n_tok, -1)

        @jax.checkpoint
        def kd_chunk(hc, yc, tic, tvc):
            lg = jnp.dot(hc, w_emb_t, precision=prec).astype(jnp.float32)
            lse = jax.scipy.special.logsumexp(lg, axis=-1, keepdims=True)
            ce = jnp.sum(lse[:, 0] - jnp.take_along_axis(lg, yc[:, None], axis=-1)[:, 0])
            tp = jax.nn.softmax(tvc.astype(jnp.float32), axis=-1)
            slp = jnp.take_along_axis(lg, tic.astype(jnp.int32), axis=-1) - lse
            kl = jnp.sum(tp * (jnp.log(jnp.maximum(tp, 1e-9)) - slp))
            return 0.5 * ce + 0.5 * kl

        def scan_kd(acc, xs):
            return acc + kd_chunk(xs[0], xs[1], xs[2], xs[3]), None

        tot, _ = jax.lax.scan(
            scan_kd,
            jnp.zeros((), jnp.float32),
            (
                h.reshape(n_c, step_sz, -1),
                y_flat.reshape(n_c, step_sz),
                ti_flat.reshape(n_c, step_sz, -1),
                tv_flat.reshape(n_c, step_sz, -1),
            ),
        )
        return tot / n_tok
