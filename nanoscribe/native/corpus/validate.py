"""Dedupe, leakage and statistics for a corpus build.

A corpus is not trustworthy because it is large. It is trustworthy when the
duplicate rate, the partition separation and the contamination check against the
frozen evaluation suite have all been measured and recorded.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from typing import Any

from nanoscribe.native.corpus import vocab
from nanoscribe.native.corpus.schema import CorpusExample, Partition

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Case/whitespace/punctuation-insensitive form used for near-duplicate detection."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^\w\s']", " ", text)
    return _WS.sub(" ", text).strip()


def content_hash(example: CorpusExample) -> str:
    return hashlib.sha256(f"{example.prompt}\x00{example.target}".encode()).hexdigest()


def near_hash(example: CorpusExample) -> str:
    return hashlib.sha256(
        f"{normalize(example.prompt)}\x00{normalize(example.target)}".encode()
    ).hexdigest()


def dedupe(examples: list[CorpusExample]) -> tuple[list[CorpusExample], dict[str, Any]]:
    """Drop exact and near duplicates, reporting what was removed."""
    seen_exact: set[str] = set()
    seen_near: set[str] = set()
    kept: list[CorpusExample] = []
    n_exact = n_near = 0
    for ex in examples:
        h, nh = content_hash(ex), near_hash(ex)
        if h in seen_exact:
            n_exact += 1
            continue
        if nh in seen_near:
            n_near += 1
            continue
        seen_exact.add(h)
        seen_near.add(nh)
        kept.append(ex)
    return kept, {
        "input": len(examples),
        "kept": len(kept),
        "removed_exact_duplicates": n_exact,
        "removed_near_duplicates": n_near,
        "duplicate_rate": round(1 - len(kept) / len(examples), 6) if examples else 0.0,
    }


def check_leakage(examples: list[CorpusExample]) -> dict[str, Any]:
    """Contamination check against the frozen screening suite.

    Three independent surfaces, because any one of them can leak alone:
      1. reserved VALUES appearing in a generated example,
      2. normalized SOURCE text overlapping an eval transcript,
      3. train/dev VALUE overlap (a held-out value that is also trained on).
    """
    reserved = vocab.forbidden_values()
    value_hits = sorted({ex.raw_value for ex in examples if ex.raw_value.lower() in reserved})

    eval_norm: set[str] = set()
    eval_error: str | None = None
    try:
        from nanoscribe.campaign_datasets import campaign_cases

        for case in campaign_cases("p1_screening_eval_v1"):
            text = getattr(case.model_input.source, "text", "") or ""
            if text:
                eval_norm.add(normalize(text))
    except Exception as exc:  # pragma: no cover
        eval_error = f"{type(exc).__name__}: {exc}"

    source_hits = 0
    if eval_norm:
        for ex in examples:
            body = ex.prompt.split("\n\nQuestion:")[0].replace("Transcript:\n", "")
            if normalize(body) in eval_norm:
                source_hits += 1

    by_part: dict[Partition, set[str]] = defaultdict(set)
    for ex in examples:
        by_part[ex.partition].add(ex.raw_value.lower())
    train_v = by_part.get(Partition.TRAIN, set())
    dev_v = by_part.get(Partition.DEV, set())
    test_v = by_part.get(Partition.INTERNAL_TEST, set())

    return {
        "reserved_value_hits": value_hits,
        "n_reserved_value_hits": len(value_hits),
        "eval_source_overlap_count": source_hits,
        "eval_sources_indexed": len(eval_norm),
        "eval_index_error": eval_error,
        "train_dev_value_overlap": sorted(train_v & dev_v),
        "train_test_value_overlap": sorted(train_v & test_v),
        "pass": (
            not value_hits
            and source_hits == 0
            and not (train_v & dev_v)
            and not (train_v & test_v)
        ),
    }


def statistics(examples: list[CorpusExample]) -> dict[str, Any]:
    """Coverage and volume, per the corpus-manifest requirements."""
    axis_hist: Counter[str] = Counter()
    for ex in examples:
        for axis in ex.axes:
            axis_hist[axis.value] += 1

    prompt_chars = sum(len(ex.prompt) for ex in examples)
    target_chars = sum(len(ex.target) for ex in examples)

    return {
        "n_examples": len(examples),
        "partition_sizes": dict(Counter(ex.partition.value for ex in examples)),
        "layer_sizes": dict(Counter(ex.layer.value for ex in examples)),
        "axis_histogram": dict(axis_hist.most_common()),
        "template_histogram": dict(Counter(ex.template_id for ex in examples).most_common()),
        "unique_values": len({ex.raw_value for ex in examples}),
        "value_histogram_top20": dict(Counter(ex.raw_value for ex in examples).most_common(20)),
        "target_label_histogram": dict(
            Counter(ex.target.split(":")[0] for ex in examples).most_common()
        ),
        "chars": {
            "prompt_total": prompt_chars,
            "target_total": target_chars,
            "total": prompt_chars + target_chars,
            "mean_prompt": round(prompt_chars / len(examples), 1) if examples else 0,
        },
        # The native tokenizer is character-level, so character count IS the
        # training-token count. Recorded explicitly to keep budget claims honest.
        "training_tokens_char_level": prompt_chars + target_chars,
    }


def axis_coverage_floor(
    examples: list[CorpusExample], minimum: int
) -> dict[str, Any]:
    """Which axes fall below a required per-axis example count."""
    hist: Counter[str] = Counter()
    for ex in examples:
        for axis in ex.axes:
            hist[axis.value] += 1
    below = {a: n for a, n in hist.items() if n < minimum}
    return {"minimum_required": minimum, "below_floor": below, "pass": not below}


def sequence_budget(
    examples: list[CorpusExample], max_seq: int = 512
) -> dict[str, Any]:
    """Can each row actually be learned within the model's context window?

    The 2026-08-24 native30 wave trained nine models on a corpus whose every
    prompt was 519-642 characters against max_seq=512. hash_tokens is
    character-level and the trainer sliced `(prompt + target)[:max_seq]`, so the
    target was discarded for 100% of rows and no gate noticed. The trainer now
    budgets the target first and truncates the prompt from the LEFT, so an
    over-length prompt degrades instead of corrupting -- but degradation has its
    own failure mode: if the gold span scrolls out of the visible window, exact
    emission is impossible by construction (the same defect tokenize.py's
    docstring records for the historical text[:64] truncation).

    Two gated invariants:
      * target_fits      -- the target alone must fit, else it is truncated.
      * spans_visible    -- every raw_value must survive prompt left-truncation.

    `prompt_truncated` is reported but NOT gated: losing the head of a long
    transcript is acceptable when the span is still in view.
    """
    over_budget = 0
    truncated = 0
    span_lost: list[str] = []
    worst = 0

    for ex in examples:
        prompt, target = ex.prompt, ex.target
        worst = max(worst, len(prompt) + len(target))
        if len(prompt) + len(target) > max_seq:
            over_budget += 1
        if len(prompt) > max_seq - min(len(target), max_seq - 1):
            truncated += 1
        # Mirror the trainer's budgeting exactly.
        budget = max_seq - min(len(target), max_seq - 1)
        visible = prompt[-budget:] if budget > 0 else ""
        span = (ex.raw_value or "").strip()
        if span and span not in visible:
            span_lost.append(ex.atom_id)

    target_fits = all(len(ex.target) < max_seq for ex in examples)
    return {
        "max_seq": max_seq,
        "worst_prompt_plus_target": worst,
        "n_over_budget": over_budget,
        "n_prompt_truncated": truncated,
        "n_span_lost_to_truncation": len(span_lost),
        "span_lost_examples": span_lost[:20],
        "target_fits": target_fits,
        "spans_visible": not span_lost,
        # Honest scope note. This gate validates the corpus against the CURRENT
        # trainer contract (target budgeted first, prompt truncated left). It
        # cannot detect a trainer that reverts to right-truncation, because the
        # target still "fits" on its own -- that regression is pinned by
        # nanoscribe/test_native_loss_target_budget.py instead. When this flag is
        # true the corpus is only learnable *because* that contract holds.
        "relies_on_left_truncation": over_budget > 0,
        "pass": target_fits and not span_lost,
    }
