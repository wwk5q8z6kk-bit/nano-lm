"""Partition assignment helpers for corpus builds."""

from __future__ import annotations

from collections import Counter
from typing import Any

from nanoscribe.native.corpus.schema import CorpusExample, Partition


def split_counts(examples: list[CorpusExample]) -> dict[str, int]:
    return dict(Counter(ex.partition.value for ex in examples))


def assert_train_only_for_gpu(examples: list[CorpusExample]) -> None:
    """Training loaders must not ingest DEV/INTERNAL_TEST rows."""
    bad = [ex.encounter_id for ex in examples if ex.partition is not Partition.TRAIN]
    if bad:
        raise ValueError(
            f"non-TRAIN partition rows present ({len(bad)}); first={bad[0]!r}"
        )


def partition_report(examples: list[CorpusExample]) -> dict[str, Any]:
    counts = split_counts(examples)
    return {
        "partition_sizes": counts,
        "train_fraction": round(
            counts.get(Partition.TRAIN.value, 0) / len(examples), 4
        )
        if examples
        else 0.0,
    }
