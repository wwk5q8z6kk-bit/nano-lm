"""Native Nano inference — candidate scoring for span-port eval.

Hash tokenization is collision-prone for free-form decode, so inference scores a
finite candidate set (matching training target format) via teacher-forced
log-probability and returns a parseable span-port line with quoted evidence.
"""

from __future__ import annotations

import re
from typing import Any

from nanoscribe.adapt import format_label_answer, parse_label_and_quotes
from nanoscribe.native.config import NativeTrainConfig
from nanoscribe.native.tokenize import detokenize, hash_tokens

_LABEL_PREFIXES = ("NOT_MENTIONED", "ASSERTED", "STATED", "DENIED", "UNCERTAIN")
_TRANSCRIPT_RE = re.compile(r"Transcript:\n(.*?)\n\nQuestion:", re.DOTALL)


def _unique_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def transcript_from_prompt(prompt: str) -> str | None:
    match = _TRANSCRIPT_RE.search(prompt)
    return match.group(1).strip() if match else None


def build_target_candidates(
    *,
    raw_value: str | None = None,
    transcript_text: str | None = None,
) -> list[str]:
    """Build span-port targets aligned with native training labels."""
    candidates = ["NOT_MENTIONED"]
    if raw_value:
        value = raw_value.strip()
        candidates.extend(
            [
                f"DENIED: {value}",
                f"UNCERTAIN: {value}",
                f"ASSERTED: {value}",
            ]
        )
    if transcript_text:
        for turn_line in transcript_text.splitlines():
            text = turn_line.split(": ", 1)[-1].strip() if ": " in turn_line else turn_line.strip()
            if not text:
                continue
            candidates.append(f"ASSERTED: {text}")
            if raw_value:
                needle = raw_value.lower()
                hay = text.lower()
                idx = hay.find(needle)
                if idx >= 0:
                    quote = text[idx : idx + len(raw_value)]
                    candidates.append(f"ASSERTED: {quote}")
    return _unique_preserve(candidates)


def score_target(model: Any, prompt: str, target: str, cfg: NativeTrainConfig) -> float:
    """Mean log-probability of target tokens given prompt (teacher forcing)."""
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    vocab = cfg.vocab_size
    prompt_ids = hash_tokens(prompt, vocab)
    target_ids = hash_tokens(target, vocab)
    seq = (prompt_ids + target_ids)[: cfg.max_seq]
    if len(seq) < 2:
        return float("-inf")

    x = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
    logits = model(x)
    log_probs = F.log_softmax(logits[0], dim=-1)

    prompt_len = min(len(prompt_ids), len(seq) - 1)
    total = 0.0
    count = 0
    for pos in range(prompt_len - 1, len(seq) - 1):
        token = seq[pos + 1]
        total += float(log_probs[pos, token].item())
        count += 1
    return total / max(count, 1)


def format_scored_target(target: str) -> str:
    """Normalize a training-style target into a parser-friendly span-port line."""
    label, quotes = parse_label_and_quotes(target)
    if label is None:
        upper = target.upper()
        for prefix in _LABEL_PREFIXES:
            if upper.startswith(prefix):
                label = prefix
                rest = target[len(prefix) :].lstrip(": ").strip().strip('"')
                if rest:
                    quotes = (rest,)
                break
    if label is None:
        return target.strip()
    if label == "NOT_MENTIONED":
        return "NOT_MENTIONED"
    if not quotes and ":" in target:
        rest = target.split(":", 1)[1].strip().strip('"')
        if rest:
            quotes = (rest,)
    return format_label_answer(label, quotes)


def select_best_target(
    model: Any,
    prompt: str,
    cfg: NativeTrainConfig,
    *,
    raw_value: str | None = None,
    transcript_text: str | None = None,
) -> str:
    candidates = build_target_candidates(raw_value=raw_value, transcript_text=transcript_text)
    best = max(candidates, key=lambda candidate: score_target(model, prompt, candidate, cfg))
    return format_scored_target(best)


def _autoregressive_line(
    model: Any,
    prompt: str,
    cfg: NativeTrainConfig,
    *,
    max_new_tokens: int,
) -> str:
    import torch

    device = next(model.parameters()).device
    vocab = cfg.vocab_size
    prompt_ids = hash_tokens(prompt, vocab)
    seq = list(prompt_ids)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            window = seq[-(cfg.max_seq - 1) :]
            x = torch.tensor([window], dtype=torch.long, device=device)
            logits = model(x)
            next_id = int(logits[0, -1].argmax().item())
            if next_id <= 0:
                break
            seq.append(next_id)
            decoded = detokenize(seq[len(prompt_ids) :], vocab)
            if "\n" in decoded:
                break
    return detokenize(seq[len(prompt_ids) :], vocab).strip()


def generate_target_line(
    model: Any,
    prompt: str,
    cfg: NativeTrainConfig,
    *,
    max_new_tokens: int = 64,
    raw_value: str | None = None,
    source: Any | None = None,
) -> str:
    transcript = transcript_from_prompt(prompt)
    if source is not None:
        from nanoscribe.prompt import _format_transcript

        transcript = _format_transcript(source)

    if raw_value is not None or transcript:
        return select_best_target(
            model,
            prompt,
            cfg,
            raw_value=raw_value,
            transcript_text=transcript,
        )

    raw = _autoregressive_line(model, prompt, cfg, max_new_tokens=max_new_tokens)
    formatted = format_scored_target(raw)
    label, _ = parse_label_and_quotes(formatted)
    return formatted if label is not None else raw
