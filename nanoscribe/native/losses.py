"""Native Nano training losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nanoscribe.native.config import LossWeights, NativeTrainConfig


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    total: float
    lm: float
    span_port: float
    evidence_align: float
    assertion_state: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "lm": self.lm,
            "span_port": self.span_port,
            "evidence_align": self.evidence_align,
            "assertion_state": self.assertion_state,
        }


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("torch required for native losses") from exc
    return torch, F


def _hash_tokens(text: str, vocab_size: int) -> list[int]:
    tokens = []
    for ch in text[:64]:
        tokens.append((ord(ch) % (vocab_size - 1)) + 1)
    if not tokens:
        tokens = [1]
    return tokens


def compute_batch_loss(
    model: Any,
    batch_prompts: list[str],
    batch_targets: list[str],
    cfg: NativeTrainConfig,
) -> LossBreakdown:
    torch, F = _require_torch()
    device = next(model.parameters()).device
    vocab = cfg.vocab_size
    weights = cfg.loss_weights

    input_ids = []
    labels = []
    for prompt, target in zip(batch_prompts, batch_targets, strict=True):
        prompt_ids = _hash_tokens(prompt, vocab)
        target_ids = _hash_tokens(target, vocab)
        seq = (prompt_ids + target_ids)[: cfg.max_seq]
        if len(seq) < 2:
            seq = seq + [1]
        input_ids.append(seq[:-1])
        labels.append(seq[1:])

    max_len = max(len(row) for row in input_ids)
    pad_id = 0
    x = torch.full((len(input_ids), max_len), pad_id, dtype=torch.long, device=device)
    y = torch.full((len(labels), max_len), -100, dtype=torch.long, device=device)
    for i, (inp, lab) in enumerate(zip(input_ids, labels, strict=True)):
        x[i, : len(inp)] = torch.tensor(inp, device=device)
        y[i, : len(lab)] = torch.tensor(lab, device=device)

    logits = model(x)
    lm = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1), ignore_index=-100)
    span_port = lm * 0.5
    evidence_align = lm * 0.25 if cfg.evidence_aware else torch.tensor(0.0, device=device)
    assertion_state = lm * 0.1 if cfg.evidence_aware else torch.tensor(0.0, device=device)

    total = (
        weights.lm * lm
        + weights.span_port * span_port
        + weights.evidence_align * evidence_align
        + weights.assertion_state * assertion_state
    )
    return LossBreakdown(
        total=total,
        lm=lm,
        span_port=span_port,
        evidence_align=evidence_align,
        assertion_state=assertion_state,
    )
