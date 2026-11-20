"""Two native copy-head baselines and training generator recovered from EXACT-RECALL-03."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from backbone import Cfg, LM, RMSNorm

PAD, SEP, BOS, _UNUSED = 0, 1, 2, 3
N_KEYS, PRIMARY_KEYS, SEEN_KEYS = 512, 224, 448   # 0..223 primary, 224..447 semantic alias (k+224), 448..511 unseen
ALIAS_OFFSET = 224
N_VALS, SEEN_VALS = 256, 224                      # 0..223 seen, 224..255 unseen
K0 = 4
V0 = K0 + N_KEYS                                  # 516
VOCAB = V0 + N_VALS                               # 772
Q = 16
NEURAL_MODES = {"DA_gen": "gen", "DA_idcopy_v2": "idcopy_v2", "DA_dual_idcopy": "dual_idcopy"}


def make_batch(
    rng: np.random.Generator,
    B: int,
    N: int,
    key_lo: int,
    key_hi: int,
    val_lo: int,
    val_hi: int,
    overwrite_prob: float = 0.25,
    query_overwritten_only: bool = False,
    va_pool: int | None = None,
    alias_prob: float = 0.0,
):
    """Build (idx, tgt) supporting overwrites, colliding v_a, and semantic alias queries (k + 224)."""
    T = 1 + 3 * N + 1 + 3 * Q
    idx = np.full((B, T), PAD, dtype=np.int64)
    tgt = np.full((B, T), -100, dtype=np.int64)
    va_hi = val_lo + va_pool if va_pool is not None else val_hi

    for b in range(B):
        do_over = query_overwritten_only or (rng.random() < overwrite_prob)
        n_over = max(1, N // 4) if (do_over and N >= 4) else 0
        u = N - n_over
        uniq_keys = rng.choice(np.arange(key_lo, key_hi), size=u, replace=False) + K0
        vas = rng.integers(val_lo, va_hi, size=u) + V0
        vbs = rng.integers(val_lo, val_hi, size=u) + V0

        order = rng.permutation(u)
        pairs: list[tuple[int, int, int]] = [
            (int(uniq_keys[i]), int(vas[i]), int(vbs[i])) for i in order
        ]
        latest: dict[int, tuple[int, int]] = {
            int(uniq_keys[i]): (int(vas[i]), int(vbs[i])) for i in range(u)
        }

        overwritten_keys: list[int] = []
        if n_over > 0:
            over_idx = rng.choice(u, size=n_over, replace=False)
            for oi in over_idx:
                k = int(uniq_keys[oi])
                old_va, old_vb = latest[k]
                while True:
                    va2 = int(rng.integers(val_lo, va_hi)) + V0
                    vb2 = int(rng.integers(val_lo, val_hi)) + V0
                    if (va2, vb2) != (old_va, old_vb):
                        break
                pairs.append((k, va2, vb2))
                latest[k] = (va2, vb2)
                overwritten_keys.append(k)

        seq = [BOS]
        for k, va, vb in pairs:
            seq += [k, va, vb]
        seq.append(SEP)

        all_keys = [int(k) for k in uniq_keys]
        if query_overwritten_only:
            q_keys = rng.choice(overwritten_keys, size=Q, replace=True)
        elif overwritten_keys:
            half = Q // 2
            q_keys = np.concatenate([
                rng.choice(overwritten_keys, size=half, replace=True),
                rng.choice(all_keys, size=Q - half, replace=True),
            ])
            rng.shuffle(q_keys)
        else:
            q_keys = rng.choice(all_keys, size=Q, replace=True)

        base = len(seq)
        for j, k in enumerate(q_keys):
            k_int = int(k)
            va, vb = latest[k_int]
            use_alias = (k_int < K0 + PRIMARY_KEYS) and (rng.random() < alias_prob)
            q_tok = k_int + ALIAS_OFFSET if use_alias else k_int
            seq += [q_tok, va, vb]
            p = base + 3 * j
            tgt[b, p] = va
            tgt[b, p + 1] = vb
        idx[b] = seq

    return torch.from_numpy(idx), torch.from_numpy(tgt)

class IdCopy2GramOnlyModel(nn.Module):
    """Track III 3-param non-neural baseline: 1-gram + 2-gram offset identity + recency."""

    def __init__(self):
        super().__init__()
        self.gamma1 = nn.Parameter(torch.tensor(6.0))
        self.gamma2 = nn.Parameter(torch.tensor(4.0))
        self.beta_rec = nn.Parameter(torch.tensor(1.0))

    def param_report(self) -> dict[str, int]:
        return {"total": 3, "embedding": 0, "non_embedding": 3, "pointer_gate_params": 3, "backbone_non_embedding": 0}

    def forward_probs(self, idx: torch.Tensor, force_copy: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = idx.shape
        dev = idx.device
        x_prev1 = idx[:, 1:-1]   # (B, T-2): x_{j-1} for j = 2..T-1
        x_prev2 = idx[:, :-2]    # (B, T-2): x_{j-2}
        x_copy = idx[:, 2:]      # (B, T-2): x_j
        x_t_lag = torch.cat([torch.full_like(idx[:, :1], PAD), idx[:, :-1]], dim=1)  # x_{t-1}

        eq1 = (idx.unsqueeze(2) == x_prev1.unsqueeze(1)).float()
        eq2 = (x_t_lag.unsqueeze(2) == x_prev2.unsqueeze(1)).float()
        rec = (torch.arange(2, T, device=dev, dtype=torch.float32) / float(T)).view(1, 1, T - 2)
        scores = self.gamma1 * eq1 + self.gamma2 * eq2 + self.beta_rec * rec

        mask = torch.tril(torch.ones(T, T - 2, dtype=torch.bool, device=dev), diagonal=-3)
        alpha = torch.softmax(scores.masked_fill(~mask, -1e9), dim=-1) * mask.float()

        p_copy = scores.new_zeros((B, T, VOCAB))
        p_copy.scatter_add_(2, x_copy.unsqueeze(1).expand(B, T, T - 2), alpha)

        has_ctx = torch.zeros((B, T, 1), dtype=torch.bool, device=dev)
        has_ctx[:, 3:, :] = True
        p_mix = torch.where(has_ctx, p_copy, torch.full_like(p_copy, 1.0 / VOCAB))
        return p_mix, alpha, has_ctx.float()

    def loss(self, idx: torch.Tensor, tgt: torch.Tensor, force_copy: bool = False) -> torch.Tensor:
        p_mix, _, _ = self.forward_probs(idx, force_copy=force_copy)
        mask = tgt != -100
        picked = p_mix.view(-1, VOCAB)[mask.view(-1)].gather(1, tgt.view(-1)[mask.view(-1)].unsqueeze(1)).squeeze(1)
        return -torch.log(picked + 1e-12).mean()

class DualCopyLM(nn.Module):
    """Wraps 2-layer v3 DA backbone with gen, idcopy_v2, or Dual-Offset Bounded-Cosine Lagged + 2-Gram IdCopy."""

    def __init__(self, ffn: int, mode: str, dc: int = 24):
        super().__init__()
        assert mode in ("gen", "idcopy_v2", "dual_idcopy")
        self.mode = mode
        self.dc = dc
        d = 64
        self.backbone = LM(
            Cfg(vocab=VOCAB, d=d, n_head=2, n_kv=2, ffn=ffn, layout="DA", seq=1024, chunk=32, tie=True)
        )
        if mode == "idcopy_v2":
            self.w_qc = nn.Linear(d, dc, bias=False)
            self.w_kc = nn.Linear(d, dc, bias=False)
            self.q_norm = RMSNorm(dc)
            self.k_norm = RMSNorm(dc)
            self.gate = nn.Linear(d, 1, bias=True)
            self.gamma = nn.Parameter(torch.tensor(3.0))
        elif mode == "dual_idcopy":
            self.w_qc0 = nn.Linear(d, dc, bias=False)
            self.w_kc0 = nn.Linear(d, dc, bias=False)
            self.w_qc1 = nn.Linear(d, dc, bias=False)
            self.w_kc1 = nn.Linear(d, dc, bias=False)
            self.gate = nn.Linear(d, 1, bias=True)
            # softplus(3.98) ~= 4.0
            self.s_raw = nn.Parameter(torch.tensor([3.98, 3.98]))
            self.gamma1 = nn.Parameter(torch.tensor(6.0))
            self.gamma2 = nn.Parameter(torch.tensor(4.0))
            self.beta_rec = nn.Parameter(torch.tensor(1.0))

    def param_report(self) -> dict[str, int]:
        base = self.backbone.param_report()
        tot = sum(p.numel() for p in self.parameters())
        emb = self.backbone.emb.weight.numel()
        return {
            "total": tot,
            "embedding": emb,
            "non_embedding": tot - emb,
            "pointer_gate_params": tot - base["total"],
            "backbone_non_embedding": base["non_embedding"],
        }

    def forward_probs(self, idx: torch.Tensor, force_copy: bool = False) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        h = self.backbone.hidden(idx)
        logits = self.backbone.logits(h).float()
        p_gen = torch.softmax(logits, dim=-1)
        B, T, _ = h.shape

        if self.mode == "gen":
            return p_gen, None, h.new_zeros((B, T, 1), dtype=torch.float32)

        h_f = h.float()
        g = torch.sigmoid(self.gate(h_f))

        if self.mode == "idcopy_v2":
            qc = self.q_norm(self.w_qc(h_f))
            kc = self.k_norm(self.w_kc(h_f))
            scores = (qc @ kc[:, :-1].transpose(-1, -2)) / math.sqrt(self.dc)
            eq = (idx.unsqueeze(2) == idx[:, :-1].unsqueeze(1)).float()
            scores = scores + self.gamma * eq
            mask = torch.tril(torch.ones(T, T - 1, dtype=torch.bool, device=idx.device), diagonal=-2)
            alpha = torch.softmax(scores.masked_fill(~mask, -1e9), dim=-1) * mask.float()
            p_copy = h_f.new_zeros((B, T, VOCAB))
            p_copy.scatter_add_(2, idx[:, 1:].unsqueeze(1).expand(B, T, T - 1), alpha)
            t_mask = torch.ones_like(g)
            t_mask[:, :2, :] = 0.0
            g_eff = torch.ones_like(g) * t_mask if force_copy else g * t_mask
            return (1.0 - g_eff) * p_gen + g_eff * p_copy, alpha, g_eff

        # dual_idcopy: causal 2 <= j < t (j_idx = 0..T-3 corresponds to j = j_idx + 2)
        h_lag = torch.cat([torch.zeros_like(h_f[:, :1]), h_f[:, :-1]], dim=1)
        x_lag = torch.cat([torch.full_like(idx[:, :1], PAD), idx[:, :-1]], dim=1)

        q0 = F.normalize(self.w_qc0(h_f), dim=-1)
        k0 = F.normalize(self.w_kc0(h_f[:, 1:-1]), dim=-1)
        q1 = F.normalize(self.w_qc1(h_lag), dim=-1)
        k1 = F.normalize(self.w_kc1(h_f[:, :-2]), dim=-1)

        scales = F.softplus(self.s_raw)
        cos0 = q0 @ k0.transpose(-1, -2)
        cos1 = q1 @ k1.transpose(-1, -2)
        eq1 = (idx.unsqueeze(2) == idx[:, 1:-1].unsqueeze(1)).float()
        eq2 = (x_lag.unsqueeze(2) == idx[:, :-2].unsqueeze(1)).float()
        rec = (torch.arange(2, T, device=idx.device, dtype=torch.float32) / float(T)).view(1, 1, T - 2)

        scores = scales[0] * cos0 + scales[1] * cos1 + self.gamma1 * eq1 + self.gamma2 * eq2 + self.beta_rec * rec
        mask = torch.tril(torch.ones(T, T - 2, dtype=torch.bool, device=idx.device), diagonal=-3)
        alpha = torch.softmax(scores.masked_fill(~mask, -1e9), dim=-1) * mask.float()

        p_copy = h_f.new_zeros((B, T, VOCAB))
        p_copy.scatter_add_(2, idx[:, 2:].unsqueeze(1).expand(B, T, T - 2), alpha)

        t_mask = torch.ones_like(g)
        t_mask[:, :3, :] = 0.0  # t = 0,1,2 have no valid 2 <= j < t
        g_eff = torch.ones_like(g) * t_mask if force_copy else g * t_mask
        return (1.0 - g_eff) * p_gen + g_eff * p_copy, alpha, g_eff

    def loss(self, idx: torch.Tensor, tgt: torch.Tensor, force_copy: bool = False) -> torch.Tensor:
        p_mix, _, _ = self.forward_probs(idx, force_copy=force_copy)
        mask = tgt != -100
        picked = p_mix.view(-1, VOCAB)[mask.view(-1)].gather(1, tgt.view(-1)[mask.view(-1)].unsqueeze(1)).squeeze(1)
        return -torch.log(picked + 1e-12).mean()

def build_matched_arm(arm: str, ref_ffn: int = 192) -> nn.Module:
    if arm == "IdCopy2Gram_Only":
        return IdCopy2GramOnlyModel()
    target = DualCopyLM(ffn=ref_ffn, mode="dual_idcopy").param_report()["non_embedding"]
    mode = NEURAL_MODES[arm]
    best = min(range(64, 384, 2), key=lambda f: abs(DualCopyLM(f, mode).param_report()["non_embedding"] - target))
    return DualCopyLM(best, mode)

def build_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "emb" in name or "norm" in name or "n1" in name or "n2" in name or "nf" in name or "on." in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.98),
    )
