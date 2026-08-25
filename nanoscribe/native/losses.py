"""Native Nano training losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nanoscribe.native.config import LossWeights, NativeTrainConfig
from nanoscribe.native.tokenize import hash_tokens


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
    roles = []  # per label position: 0 none, 1 assertion label, 2 span
    evidence = []  # per label position: span token grounded in the source
    for prompt, target in zip(batch_prompts, batch_targets, strict=True):
        # Budget the target FIRST, then give the prompt what is left.
        #
        # This previously read `(prompt_ids + target_ids)[: cfg.max_seq]`, which
        # truncates from the right — i.e. it drops the target. hash_tokens is
        # character-level, and every prompt in native_corpus_screen_v1 is 519-642
        # chars against max_seq=512, so that slice discarded the target for
        # 100.0% of 19,194 training examples. The model never saw a label, and
        # the resulting near-zero `final_loss` was next-character prediction on
        # templated transcript text, not task learning. See
        # artifacts/campaign/reval_results/FALSE_NULL_DIAGNOSIS.md.
        #
        # The prompt is truncated from the LEFT so the question and answer
        # instructions at its tail — the part that conditions the target —
        # always survive.
        target_ids = hash_tokens(target, vocab)[: max(1, cfg.max_seq - 1)]
        prompt_budget = cfg.max_seq - len(target_ids)
        prompt_ids = hash_tokens(prompt, vocab)[-prompt_budget:] if prompt_budget > 0 else []
        seq = prompt_ids + target_ids
        if len(seq) < 2:
            seq = seq + [1]
        n_prompt = len(prompt_ids)
        # Supervise target positions only. Label i predicts seq[i + 1], so a
        # label is kept iff seq[i + 1] belongs to the target. The boundary
        # position (last prompt token -> first target token) is supervised.
        input_ids.append(seq[:-1])
        labels.append(
            [(-100 if i + 1 < n_prompt else tok) for i, tok in enumerate(seq[1:])]
        )

        # Split the target into its two supervised regions so the auxiliary
        # objectives are computed from DIFFERENT signal rather than being
        # rescalings of `lm`. Targets are "<LABEL>: <span>" or a bare
        # "NOT_MENTIONED" (12,341 of 19,194 carry a span).
        colon = target.find(":")
        if colon >= 0:
            label_chars = colon + 2 if target[colon + 1 : colon + 2] == " " else colon + 1
        else:
            label_chars = len(target)
        label_len = min(label_chars, len(target_ids))

        # evidence_align supervises copying only where the source actually
        # supports it: the span must appear verbatim in the (already truncated)
        # prompt the model can still see.
        span_text = target[label_chars:].strip()
        visible_prompt = prompt[-prompt_budget:] if prompt_budget > 0 else ""
        grounded = bool(span_text) and span_text in visible_prompt

        row_roles = [0] * len(seq[1:])
        row_evid = [False] * len(seq[1:])
        for j in range(len(target_ids)):
            i = n_prompt + j - 1
            if 0 <= i < len(row_roles):
                is_span = j >= label_len
                row_roles[i] = 2 if is_span else 1
                row_evid[i] = is_span and grounded
        roles.append(row_roles)
        evidence.append(row_evid)

    max_len = max(len(row) for row in input_ids)
    pad_id = 0
    x = torch.full((len(input_ids), max_len), pad_id, dtype=torch.long, device=device)
    y = torch.full((len(labels), max_len), -100, dtype=torch.long, device=device)
    role = torch.zeros((len(labels), max_len), dtype=torch.long, device=device)
    evid = torch.zeros((len(labels), max_len), dtype=torch.bool, device=device)
    for i, (inp, lab) in enumerate(zip(input_ids, labels, strict=True)):
        x[i, : len(inp)] = torch.tensor(inp, device=device)
        y[i, : len(lab)] = torch.tensor(lab, device=device)
        role[i, : len(roles[i])] = torch.tensor(roles[i], device=device)
        evid[i, : len(evidence[i])] = torch.tensor(evidence[i], device=device)

    logits = model(x)
    # Per-position CE, so each objective can average over its OWN region.
    #
    # These previously read `span_port = lm * 0.5`, `evidence_align = lm * 0.25`,
    # `assertion_state = lm * 0.1` — scalar multiples of one number. The total was
    # therefore an affine function of `lm` in every arm, so the three revalidation
    # arms shared an identical gradient direction and differed only in effective
    # learning rate (control 1.0x, span_port 1.5x, evidence_bottleneck 1.15x).
    # The objective factor could not be measured. Each term now has a distinct
    # supervision mask.
    per_pos = F.cross_entropy(
        logits.reshape(-1, vocab), y.reshape(-1), ignore_index=-100, reduction="none"
    ).reshape(y.shape)
    supervised = y != -100

    def _masked_mean(mask: Any) -> Any:
        mask = mask & supervised
        n = mask.sum()
        if int(n) == 0:
            return torch.zeros((), device=device, dtype=per_pos.dtype)
        return (per_pos * mask).sum() / n

    lm = _masked_mean(supervised)
    # span_port: the copied value region only.
    span_port = _masked_mean(role == 2)
    # evidence_align: span positions whose value is verbatim in the visible
    # source — copying supervised only where the evidence supports it.
    evidence_align = (
        _masked_mean(evid) if cfg.evidence_aware else torch.zeros((), device=device)
    )
    # assertion_state: the ASSERTED/DENIED/UNCERTAIN/NOT_MENTIONED label region.
    assertion_state = (
        _masked_mean(role == 1) if cfg.evidence_aware else torch.zeros((), device=device)
    )

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
