"""Evaluation statistics stubs — Wilson intervals and suite aggregation helpers."""

from __future__ import annotations

from typing import Any

from nanoscribe.eval.metrics import RateMetric, rate_metric
from nanoscribe.eval.power import wilson


def summarize_rate(
    name: str,
    numerator: int,
    denominator: int,
    *,
    eligible_unit: str = "atom",
    aggregation_level: str = "suite",
) -> dict[str, Any]:
    metric = rate_metric(
        name,
        numerator,
        denominator,
        eligible_unit=eligible_unit,
        aggregation_level=aggregation_level,
    )
    lo, hi = wilson(metric.numerator, metric.denominator)
    return {
        **metric.to_dict(),
        "wilson95": [lo, hi],
    }


def merge_rate_metrics(metrics: list[RateMetric]) -> RateMetric:
    if not metrics:
        raise ValueError("no metrics to merge")
    name = metrics[0].name
    num = sum(m.numerator for m in metrics)
    den = sum(m.denominator for m in metrics)
    return rate_metric(
        name,
        num,
        den,
        eligible_unit=metrics[0].eligible_unit,
        aggregation_level=metrics[0].aggregation_level,
    )
