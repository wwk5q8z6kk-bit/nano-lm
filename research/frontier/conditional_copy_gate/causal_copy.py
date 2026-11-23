"""Experimental prefix-causal pointer mixture for Nano LM.

This is an optional head on a causal language model, NOT a new backbone.
At input position t the model predicts token x[t+1]. Its pointer may copy
x[j] only for 1 <= j <= t, comparing the current hidden state h[t] to
h[j-1] (the context preceding the candidate token).

Only the target probability is materialized during training; no B*T*V
pointer tensor is allocated. The optional full distribution is for
inference/debugging and does require B*T*V memory.

Experiment, not an established architecture improvement. No training data,
weights, metrics or runtime are changed by importing this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class CopyStats:
    """Lightweight diagnostics, detached from the autograd graph."""

    eligible_fraction: float
    mean_copy_gate: float
    mean_max_pointer_weight: float


class CausalCopyMixture(nn.Module):
    """Gated mixture of base next-token LM probabilities and causal copying.

    External hidden MUST be generated with causal attention/state updates.
    token_ids[b,t] is an observed input token, not the next-token target.
    targets[b,t] is the token to be predicted at that position.
    """

    def __init__(self, hidden_dim: int, window: int = 64, pointer_dim: int | None = None,
                 initial_gate_logit: float = -6.0):
        super().__init__()
        if hidden_dim < 1 or window < 1:
            raise ValueError("hidden_dim and window must be positive")
        self.hidden_dim = int(hidden_dim)
        self.window = int(window)
        self.pointer_dim = int(pointer_dim or hidden_dim)
        if self.pointer_dim < 1:
            raise ValueError("pointer_dim must be positive")
        self.query = nn.Linear(hidden_dim, self.pointer_dim, bias=False)
        self.key = nn.Linear(hidden_dim, self.pointer_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, initial_gate_logit)

    def _candidates(self, token_ids: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if token_ids.ndim != 2 or hidden.ndim != 3:
            raise ValueError("expected token_ids [B,T], hidden [B,T,D]")
        b, t = token_ids.shape
        if hidden.shape != (b, t, self.hidden_dim):
            raise ValueError("hidden and token_ids shapes disagree")
        if t == 0:
            raise ValueError("sequence length must be nonzero")
        if token_ids.dtype not in (torch.int32, torch.int64, torch.long):
            raise TypeError("token_ids must be integer token indices")

        w = min(t, self.window)
        now = torch.arange(t, device=token_ids.device).unsqueeze(1)
        offsets = torch.arange(w, device=token_ids.device).unsqueeze(0)
        candidate_positions = now - offsets  # x[j], j <= current position t
        valid = candidate_positions >= 1  # need h[j-1] as predecessor key
        candidate_idx = candidate_positions.clamp(min=0)
        predecessor_idx = (candidate_positions - 1).clamp(min=0)

        queries = self.query(hidden).unsqueeze(-2)
        previous_keys = self.key(hidden)[:, predecessor_idx, :]
        scores = (queries.float() * previous_keys.float()).sum(-1) / math.sqrt(self.pointer_dim)
        scores = scores.masked_fill(~valid.unsqueeze(0), -1e9)
        weights = torch.softmax(scores, dim=-1) * valid.unsqueeze(0)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-30)
        copies = token_ids[:, candidate_idx]
        has_candidates = valid.any(-1).unsqueeze(0).expand(b, -1)
        gate_logit = self.gate(hidden).squeeze(-1).float()
        return copies, weights, gate_logit, has_candidates

    def target_nll(self, base_logits: Tensor, token_ids: Tensor,
                   hidden: Tensor, targets: Tensor) -> Tensor:
        """Return [B,T] token NLL with sparse, stable target-only mixture.

        The target is never supplied to or used by the pointer selection or
        gate; it is used only to score the resulting distribution.
        """
        if base_logits.ndim != 3 or base_logits.shape[:2] != token_ids.shape:
            raise ValueError("base_logits must be [B,T,V]")
        if targets.shape != token_ids.shape:
            raise ValueError("targets must be [B,T]")
        if targets.dtype not in (torch.int32, torch.int64, torch.long):
            raise TypeError("targets must be integer token indices")
        copies, weights, gate_logits, eligible = self._candidates(token_ids, hidden)
        log_base = F.log_softmax(base_logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        copy_target_mass = ((copies == targets.unsqueeze(-1)) * weights).sum(-1)
        log_copy = torch.where(copy_target_mass > 0,
                               copy_target_mass.clamp_min(1e-30).log(),
                               torch.full_like(copy_target_mass, -torch.inf))
        log_gate = torch.where(eligible, F.logsigmoid(gate_logits), torch.full_like(gate_logits, -torch.inf))
        log_keep = torch.where(eligible, F.logsigmoid(-gate_logits), torch.zeros_like(gate_logits))
        return -torch.logaddexp(log_keep + log_base, log_gate + log_copy)

    def full_log_probs(self, base_logits: Tensor, token_ids: Tensor, hidden: Tensor) -> Tensor:
        """Full [B,T,V] distribution for inference; incurs vocabulary-sized storage."""
        if base_logits.ndim != 3 or base_logits.shape[:2] != token_ids.shape:
            raise ValueError("base_logits must be [B,T,V]")
        copies, weights, gate_logits, eligible = self._candidates(token_ids, hidden)
        vocab = base_logits.shape[-1]
        if torch.any((copies < 0) | (copies >= vocab)):
            raise ValueError("candidate ids fall outside vocabulary")
        pointer_prob = torch.zeros_like(base_logits, dtype=torch.float32)
        pointer_prob.scatter_add_(-1, copies.long(), weights)
        log_pointer = torch.where(pointer_prob > 0, pointer_prob.clamp_min(1e-30).log(),
                                  torch.full_like(pointer_prob, -torch.inf))
        log_gate = torch.where(eligible, F.logsigmoid(gate_logits),
                               torch.full_like(gate_logits, -torch.inf))
        log_keep = torch.where(eligible, F.logsigmoid(-gate_logits),
                               torch.zeros_like(gate_logits))
        log_base = F.log_softmax(base_logits.float(), dim=-1)
        return torch.logaddexp(log_keep.unsqueeze(-1) + log_base,
                               log_gate.unsqueeze(-1) + log_pointer)

    @torch.no_grad()
    def stats(self, token_ids: Tensor, hidden: Tensor) -> CopyStats:
        _, weights, logit, eligible = self._candidates(token_ids, hidden)
        gate = torch.sigmoid(logit) * eligible
        return CopyStats(float(eligible.float().mean()), float(gate.mean()),
                         float(weights.amax(-1).mean()))
