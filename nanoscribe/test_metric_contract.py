"""Regression pins for the explicit rate metric contract."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.eval.metrics import RateMetric, assert_rate_bounds, rate_metric
from nanoscribe.eval.statistics import merge_rate_metrics, summarize_rate


def test_rate_metric_bounds() -> None:
    metric = rate_metric("coverage", 84, 150)
    assert metric.rate == pytest.approx(0.56)
    assert_rate_bounds(metric.rate, name="coverage")


def test_rate_metric_rejects_numerator_gt_denominator() -> None:
    with pytest.raises(ValueError, match="exceeds denominator"):
        rate_metric("assertion_state_correct", 2, 1)


def test_rate_metric_rejects_rate_above_one() -> None:
    with pytest.raises(ValueError, match="exceeds 1.0"):
        RateMetric("broken", 3, 2).validate()


def test_summarize_rate_includes_wilson() -> None:
    summary = summarize_rate("assertion_state_correct", 118, 150)
    assert summary["numerator"] == 118
    assert summary["denominator"] == 150
    assert 0.0 <= summary["rate"] <= 1.0
    assert len(summary["wilson95"]) == 2


def test_merge_rate_metrics_sums_counts() -> None:
    merged = merge_rate_metrics(
        [
            rate_metric("coverage", 40, 75),
            rate_metric("coverage", 44, 75),
        ]
    )
    assert merged.numerator == 84
    assert merged.denominator == 150
    assert merged.rate == pytest.approx(0.56)
