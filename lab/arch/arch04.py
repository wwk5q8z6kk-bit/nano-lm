"""Track A Phase-2 Breakthrough Architectures (`LAB-ARCH-04`: `X5_atrd_muon` & `X6_pkm_muon`).

Implements two original, strictly causal mechanisms derived from Phase-1 falsifications:

  1. `CausalEntropyRefiner` (`X5_atrd_muon` — Adaptive Token-Routed Residual Depth):
     - Pass 1 runs `NanoLM-v2` (`DDDADDDADA`: Sharp-Delta + Gated-VRes + QK-Canon), caching the
       final anchor attention layer's causal `(K^(1), V^(1))` working memory and Pass-1 hidden `h^(1)`.
     - Computes strictly causal token predictive Shannon entropy `u_t = H(p_1(. | x_{<=t}))`.
     - Compares `u_t` against an EMA running quantile `tau` (updated during training, frozen buffer
       during eval -> zero future-token leakage!) so ONLY hard/uncertain tokens (`~25%`) execute the
       step-conditioned Cross-Memory Refinement block (`Q_hard` attending causally to full `(K^(1), V^(1))`
       plus a SwiGLU refinement FFN), saving 75% of Pass-2 compute!
     - Includes an Anti-Circularity Cosine Penalty preventing Pass-2 updates from collinearly
       repeating Pass-1 updates.

  2. `SparsePersistentMemory` (`X6_pkm_muon` — Confidence-Gated Persistent Knowledge Slots):
     - Stores `M = 2,048` persistent key-value memory slots per bank while activating ONLY `top_s = 8`
       (`0.39%`) value slots per token, gated by retrieval confidence `sigmoid(max_sim - tau_conf)` so
       low-match tokens are never polluted by irrelevant memory slots.
     - Reports `active_non_embedding` parameters and `persistent_memory` parameters strictly separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from lab.arch.model import RMSNorm, apply_rope, rope_cache
from lab.arch.nextgen import NextGenBlock, NextGenCfg


@dataclass
class Arch04Cfg:
    vocab: int = 32768
    d: int = 512
    n_head: int = 8
    n_kv: int = 4
    ffn: int = 1152
    layout: str = "DDDADDDADA"
    seq: int = 1024
    rope_theta: float = 10000.0
    conv: int = 4
    chunk: int = 64
    tie: bool = True
    # Mechanism toggles:
    use_atrd: bool = False            # X5/X8: Causal Adaptive Token-Routed Residual Depth
    use_margin_gate: bool = False     # X8: Vocabulary-invariant Top-2 margin u_t = 1 - (p_(1) - p_(2)) + copy shield
    atrd_frac: float = 0.25           # Target fraction of hard tokens routed to Pass 2
    anticirc_weight: float = 0.02     # Anti-circularity orthogonality penalty weight
    use_pkm: bool = False             # X6: Confidence-Gated Sparse Persistent Memory
    pkm_layers: tuple[int, ...] = (4, 8)
    pkm_slots: int = 2048
    pkm_topk: int = 8
    pkm_key_dim: int = 128


class SparsePersistentMemory(nn.Module):
    """Confidence-gated top-s associative memory bank decoupling slot capacity from active FLOPs."""

    def __init__(self, d: int, n_slots: int = 2048, topk: int = 8, key_dim: int = 128):
        super().__init__()
        self.d = d
        self.n_slots = n_slots
        self.topk = topk
        self.key_dim = key_dim
        self.norm = RMSNorm(d)
        self.q_proj = nn.Linear(d, key_dim, bias=False)
        self.mem_keys = nn.Parameter(torch.randn(n_slots, key_dim) * (key_dim ** -0.5))
        self.mem_vals = nn.Parameter(torch.randn(n_slots, d) * 0.02)
        self.conf_tau = nn.Parameter(torch.tensor(0.25))
        self.conf_temp = nn.Parameter(torch.tensor(4.0))
        # Zero-init output scale so memory starts exact-identity and grows where useful
        self.out_scale = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        B, T, D = x.shape
        q = F.normalize(self.q_proj(self.norm(x)).float(), dim=-1)          # B, T, kd
        k = F.normalize(self.mem_keys.float(), dim=-1)                      # M, kd
        sims = q @ k.T                                                      # B, T, M
        top_sim, top_idx = torch.topk(sims, k=self.topk, dim=-1)            # B, T, s
        weights = F.softmax(top_sim * 10.0, dim=-1)                         # B, T, s
        # Confidence gate: suppress retrieval if even best slot similarity is weak
        conf_gate = torch.sigmoid(self.conf_temp.abs() * (top_sim[..., :1] - self.conf_tau))  # B, T, 1
        # Gather ONLY top-s value vectors -> O(s * D) active value ops per token
        gathered_v = self.mem_vals[top_idx.reshape(-1)].view(B, T, self.topk, D).float()
        mem_out = (weights.unsqueeze(-1) * gathered_v).sum(dim=-2)          # B, T, D
        gated = (conf_gate * mem_out * self.out_scale.float()).to(x.dtype)
        return x + gated, float(conf_gate.mean().detach())

    def persistent_slot_params(self) -> int:
        return self.mem_keys.numel() + self.mem_vals.numel()

    def active_params_per_token(self) -> int:
        # Query projection + norm + keys scan + ONLY topk active value rows!
        return self.q_proj.weight.numel() + self.mem_keys.numel() + self.topk * self.d + self.d * 2


class CausalEntropyRefiner(nn.Module):
    """Pass-2 sparse token refiner reading Pass-1 causal working memory (K^(1), V^(1)) with zero future leak."""

    def __init__(self, c: Arch04Cfg):
        super().__init__()
        self.h = c.n_head
        self.kv = c.n_kv
        self.hd = c.d // c.n_head
        self.d = c.d
        self.target_frac = c.atrd_frac
        self.use_margin_gate = c.use_margin_gate
        self.step_emb = nn.Parameter(torch.randn(1, 1, c.d) * 0.02)
        self.n1 = RMSNorm(c.d)
        self.q = nn.Linear(c.d, c.d, bias=False)
        self.qn = RMSNorm(self.hd)
        self.o = nn.Linear(c.d, c.d, bias=False)
        self.n2 = RMSNorm(c.d)
        ref_ffn = max(256, (c.ffn // 2 // 32) * 32)
        self.gu = nn.Linear(c.d, 2 * ref_ffn, bias=False)
        self.dn = nn.Linear(ref_ffn, c.d, bias=False)
        # Zero-init output gates so Pass-2 refinement begins as exact identity (no Pass-1 disruption!)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.dn.weight)
        self.register_buffer("ema_entropy_tau", torch.tensor(3.5))
        self.register_buffer("ema_inited", torch.tensor(False))

    def forward(
        self,
        x1: torch.Tensor,
        delta_h1: torch.Tensor,
        k1_full: torch.Tensor,
        v1_full: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        entropies: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Refine ONLY causal high-entropy positions `entropies > ema_entropy_tau`.

        Shapes:
          x1, delta_h1: (B, T, D)
          k1_full, v1_full: (B, H, T, hd) from Pass-1's final Attention anchor (already RoPE'd / expanded)
          entropies: (B, T) strictly causal Pass-1 Shannon entropies
        """
        B, T, D = x1.shape
        if self.training:
            with torch.no_grad():
                q_val = torch.quantile(entropies.detach().float(), 1.0 - self.target_frac)
                if not bool(self.ema_inited.item()):
                    self.ema_entropy_tau.copy_(q_val)
                    self.ema_inited.fill_(True)
                else:
                    self.ema_entropy_tau.lerp_(q_val, 0.05)

        tau = self.ema_entropy_tau
        # Strictly causal per-token routing mask: depends ONLY on u_t (causal) and buffer `tau`!
        hard_mask = entropies > tau                                         # B, T (bool)
        temp_scale = 8.0 if self.use_margin_gate else 2.0
        soft_gate = torch.sigmoid((entropies - tau) * temp_scale)
        if self.use_margin_gate:
            # Copy-protection damping: scales residual update by margin ambiguity u_t in [0, 1]
            soft_gate = soft_gate * entropies.clamp(0.0, 1.0)
        # Straight-through gate: exact 0 on easy tokens, differentiable wrt hidden
        if self.use_margin_gate:
            gate = torch.where(hard_mask, entropies.clamp(0.0, 1.0), torch.zeros_like(soft_gate))
        else:
            gate = torch.where(hard_mask, soft_gate / soft_gate.detach().clamp_min(1e-4), torch.zeros_like(soft_gate))

        routed_frac = float(hard_mask.float().mean().detach())
        if not bool(hard_mask.any()):
            zero = x1.sum() * 0.0
            return x1, {"routed_frac": 0.0, "anticirc_loss": zero, "entropy_tau": float(tau.detach())}

        # 32-bucket aligned sparse gather -> eliminates dynamic-shape CUDA allocator fragmentation!
        counts = hard_mask.sum(dim=-1)                                      # (B,)
        raw_max_k = int(counts.max().item())
        max_k = min(T, max(1, ((raw_max_k + 31) // 32) * 32)) if T >= 32 else raw_max_k
        # Pad batch rows with fewer than max_k hard tokens using index 0 (masked out before scatter!)
        topk_idx = torch.topk(hard_mask.int() * (T + 1) + torch.arange(T, device=x1.device), k=max_k, dim=-1).indices
        topk_idx, _ = torch.sort(topk_idx, dim=-1)                          # B, K_max in causal order
        valid_slot = hard_mask.gather(1, topk_idx)                          # B, K_max (bool)

        idx_d = topk_idx.unsqueeze(-1).expand(B, max_k, D)
        x_sel = x1.gather(1, idx_d) + self.step_emb.to(x1.dtype)            # B, K_max, D

        # Apply RoPE at the exact original token positions `topk_idx`!
        cos_all, sin_all = rope                                             # (T, hd//2)
        c_sel = cos_all[topk_idx].unsqueeze(1)                              # (B, 1, max_k, hd//2)
        s_sel = sin_all[topk_idx].unsqueeze(1)
        q_raw = self.qn(self.q(self.n1(x_sel)).view(B, max_k, self.h, self.hd)).transpose(1, 2)
        q1, q2 = q_raw[..., ::2], q_raw[..., 1::2]
        q_sel = torch.stack((q1 * c_sel - q2 * s_sel, q1 * s_sel + q2 * c_sel), -1).flatten(-2).to(k1_full.dtype)

        # Strictly causal cross-attention mask: query at pos `topk_idx[b, q]` can ONLY read keys `<= topk_idx[b, q]`
        all_pos = torch.arange(T, device=x1.device).view(1, 1, 1, T)
        q_pos = topk_idx.view(B, 1, max_k, 1)
        attn_mask = q_pos >= all_pos                                        # B, 1, K_max, T (bool)

        y_attn = F.scaled_dot_product_attention(q_sel, k1_full, v1_full, attn_mask=attn_mask)
        dx_attn = self.o(y_attn.transpose(1, 2).reshape(B, max_k, D))
        x_mid = x_sel + dx_attn
        g_ffn, u_ffn = self.gu(self.n2(x_mid)).chunk(2, dim=-1)
        dx_total = dx_attn + self.dn(F.silu(g_ffn) * u_ffn)                 # B, K_max, D

        # Anti-circularity cosine penalty against Pass-1 update direction `delta_h1` at selected positions
        dh1_sel = delta_h1.gather(1, idx_d).float()
        cos_sim = F.cosine_similarity(dx_total.float(), dh1_sel, dim=-1, eps=1e-6)  # B, K_max
        pen_tok = F.relu(cos_sim - 0.25).pow(2) * valid_slot.float()
        anticirc = pen_tok.sum() / valid_slot.float().sum().clamp_min(1.0)

        # Scatter refined deltas back ONLY onto valid hard positions
        gate_sel = gate.gather(1, topk_idx) * valid_slot.to(gate.dtype)
        scaled_dx = (dx_total.float() * gate_sel.unsqueeze(-1)).to(x1.dtype)
        x2 = x1.clone()
        x2.scatter_add_(1, idx_d, scaled_dx)

        return x2, {
            "routed_frac": round(routed_frac, 4),
            "anticirc_loss": anticirc,
            "entropy_tau": round(float(tau.detach()), 4),
        }


class Arch04LM(nn.Module):
    """NanoLM-v2 extended with strictly causal Adaptive Token-Routed Depth (X5) & Sparse PKM (X6)."""

    def __init__(self, c: Arch04Cfg):
        super().__init__()
        self.c = c
        ng_cfg = NextGenCfg(
            vocab=c.vocab, d=c.d, n_head=c.n_head, n_kv=c.n_kv, ffn=c.ffn,
            layout=c.layout, seq=c.seq, rope_theta=c.rope_theta, conv=c.conv,
            chunk=c.chunk, tie=c.tie, gated_vres=True, sharp_v=True, qk_canon_attn=True,
        )
        self.ng_cfg = ng_cfg
        self.emb = nn.Embedding(c.vocab, c.d)
        self.blocks = nn.ModuleList(NextGenBlock(ng_cfg, k) for k in c.layout)
        self.pkm_banks = nn.ModuleDict()
        if c.use_pkm:
            for li in c.pkm_layers:
                if 0 <= li < len(c.layout):
                    self.pkm_banks[str(li)] = SparsePersistentMemory(
                        c.d, n_slots=c.pkm_slots, topk=c.pkm_topk, key_dim=c.pkm_key_dim
                    )
        self.nf = RMSNorm(c.d)
        self.head = None if c.tie else nn.Linear(c.d, c.vocab, bias=False)
        self.refiner = CausalEntropyRefiner(c) if c.use_atrd else None
        self._last_attn_idx = max(i for i, k in enumerate(c.layout) if k == "A")
        self._init_weights()

    def _init_weights(self) -> None:
        n_eff = len(self.c.layout)
        for name, p in self.named_parameters():
            if any(tag in name for tag in ("emb", "v_gate", "refiner", "pkm_banks")):
                continue
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02)
                if name.endswith(("o.weight", "dn.weight")):
                    p.data.mul_(1.0 / math.sqrt(2 * n_eff))
        nn.init.normal_(self.emb.weight, std=0.02)

    def _run_anchor_attn_with_kv(self, blk: NextGenBlock, x: torch.Tensor, rope, v0):
        """Execute final Attention block while returning expanded (k_exp, v_exp) for Pass-2 cross-memory."""
        attn = blk.mix
        nx = blk.n1(x)
        B, T, _ = nx.shape
        q_raw, k_raw = attn.q(nx), attn.k(nx)
        if attn.qk_conv is not None:
            qk = attn.qk_conv(torch.cat([q_raw, k_raw], dim=-1))
            q_raw, k_raw = qk.split([attn.h * attn.hd, attn.kv * attn.hd], dim=-1)
        q = attn.qn(q_raw.view(B, T, attn.h, attn.hd)).transpose(1, 2)
        k = attn.kn(k_raw.view(B, T, attn.kv, attn.hd)).transpose(1, 2)
        v = attn.v(nx).view(B, T, attn.kv, attn.hd).transpose(1, 2)
        q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        rep = attn.h // attn.kv
        k_exp = k.repeat_interleave(rep, 1)
        v_exp = v.repeat_interleave(rep, 1)
        if attn.gated_vres:
            if v0 is None:
                v0 = v_exp
            else:
                alpha = torch.sigmoid(attn.v_gate(nx)).transpose(1, 2).unsqueeze(-1)
                v_exp = alpha * v_exp + (1.0 - alpha) * v0.to(v_exp.dtype)
        y = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
        h = attn.o(y.transpose(1, 2).reshape(B, T, -1))
        x = x + h
        g, u = blk.gu(blk.n2(x)).chunk(2, dim=-1)
        return x + blk.dn(F.silu(g) * u), v0, k_exp, v_exp

    def forward(self, idx: torch.Tensor, tgt: torch.Tensor | None = None, return_aux: bool = False):
        B, T = idx.shape
        x0 = self.emb(idx)
        x = x0
        hd = self.c.d // self.c.n_head
        rope = rope_cache(T, hd, self.c.rope_theta, idx.device)
        v0 = None
        k1_anchor = v1_anchor = None
        pkm_gates: list[float] = []

        for i, b in enumerate(self.blocks):
            if self.refiner is not None and i == self._last_attn_idx:
                x, v0, k1_anchor, v1_anchor = self._run_anchor_attn_with_kv(b, x, rope, v0)
            else:
                x, v0, _ = b(x, rope, v0)
            if str(i) in self.pkm_banks:
                x, cg = self.pkm_banks[str(i)](x)
                pkm_gates.append(cg)

        aux: dict[str, torch.Tensor | float] = {}
        if pkm_gates:
            aux["pkm_conf_gate"] = round(sum(pkm_gates) / len(pkm_gates), 4)

        w_out = self.emb.weight if self.head is None else self.head.weight
        if self.refiner is not None and k1_anchor is not None and v1_anchor is not None:
            # Compute strictly causal Pass-1 uncertainty u_t without tracking gradients into Pass-1 router
            with torch.no_grad():
                h1_norm = self.nf(x)
                lg1 = (h1_norm @ w_out.T).float()
                if self.c.use_margin_gate:
                    # Vocabulary-invariant Top-2 hypothesis margin ambiguity: u_t = 1 - (p_(1) - p_(2)) in [0, 1]
                    top2_lg = torch.topk(lg1, k=2, dim=-1).values
                    lse = torch.logsumexp(lg1, dim=-1, keepdim=True)
                    top2_p = (top2_lg - lse).exp()
                    entropies = 1.0 - (top2_p[..., 0] - top2_p[..., 1])
                else:
                    lp1 = F.log_softmax(lg1, dim=-1)
                    entropies = -(lp1.exp() * lp1).sum(dim=-1)              # B, T
            delta_h1 = (x - x0).detach()
            x, ref_aux = self.refiner(x, delta_h1, k1_anchor, v1_anchor, rope, entropies)
            aux.update(ref_aux)

        lg = (self.nf(x) @ w_out.T).float()
        if tgt is None:
            return (lg, aux) if return_aux else lg
        ce = F.cross_entropy(lg.view(-1, lg.size(-1)), tgt.reshape(-1), reduction="mean")
        loss = ce + (self.c.anticirc_weight * aux["anticirc_loss"] if "anticirc_loss" in aux else 0.0)
        return (loss, aux) if return_aux else loss

    def param_report(self) -> dict[str, int]:
        tot = sum(p.numel() for p in self.parameters())
        emb = self.emb.weight.numel() + (self.head.weight.numel() if self.head is not None else 0)
        pkm_slot_p = sum(m.persistent_slot_params() for m in self.pkm_banks.values())
        pkm_active_slot_p = sum(m.topk * m.d + m.mem_keys.numel() for m in self.pkm_banks.values())
        ref_p = sum(p.numel() for p in self.refiner.parameters()) if self.refiner is not None else 0
        # Active non-embedding parameters per average token:
        #   backbone + active PKM lookups + (atrd_frac * refiner_params)
        base_non_emb = tot - emb - pkm_slot_p - ref_p
        active_non_emb = int(base_non_emb + pkm_active_slot_p + (self.c.atrd_frac * ref_p if self.refiner else 0))
        return {
            "total": tot,
            "embedding": emb,
            "non_embedding": tot - emb,
            "active_non_embedding": active_non_emb,
            "persistent_memory_params": pkm_slot_p,
            "adaptive_refiner_params": ref_p,
            "effective_depth": len(self.c.layout) + (1 if self.refiner else 0),
        }
