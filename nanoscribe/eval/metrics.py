"""Explicit numerator/denominator/rate metric contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RateMetric:
    name: str
    numerator: int
    denominator: int
    eligible_unit: str = "atom"
    aggregation_level: str = "suite"

    @property
    def rate(self) -> float:
        if self.denominator <= 0:
            return 0.0
        return round(self.numerator / self.denominator, 6)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rate"] = self.rate
        return payload

    def validate(self) -> None:
        if self.numerator < 0 or self.denominator < 0:
            raise ValueError(f"{self.name}: counts must be non-negative")
        if self.numerator > self.denominator:
            raise ValueError(
                f"{self.name}: numerator {self.numerator} exceeds denominator {self.denominator}"
            )
        if self.rate > 1.0:
            raise ValueError(f"{self.name}: rate {self.rate} exceeds 1.0")


def rate_metric(
    name: str,
    numerator: int,
    denominator: int,
    *,
    eligible_unit: str = "atom",
    aggregation_level: str = "suite",
) -> RateMetric:
    metric = RateMetric(
        name=name,
        numerator=numerator,
        denominator=denominator,
        eligible_unit=eligible_unit,
        aggregation_level=aggregation_level,
    )
    metric.validate()
    return metric


def assert_rate_bounds(rate: float, *, name: str = "metric") -> None:
    if rate < 0.0 or rate > 1.0:
        raise ValueError(f"{name}: rate {rate} outside [0, 1]")
