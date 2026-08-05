"""Risk-coverage analysis for evidence-bound (ternary) selective prediction.

Terms are canonical per `papers/SELECTIVE_VOCABULARY.md`; the concepts are
standard selective prediction (El-Yaniv & Wiener, JMLR 2010; Geifman &
El-Yaniv, NeurIPS 2017) and nothing here claims to invent them.

Why this module exists: `evidence_query_inference.apply_global_threshold` and
`_calibration_diagnostics` are both already threshold-parameterized and are
never called in a loop, so the project has only ever seen two points of a curve
it could have plotted all along. `minimal_zero_wrong_presented_inclusive_v1`
picks `max(wrong_confidences)` and reports the resulting risk without ever
naming the coverage it cost.

Two structural differences from textbook selective prediction:

1. **Ternary, not binary.** An evidence-bound scribe may assert a value, assert
   an *absence* (which itself requires positive denial evidence), or abstain.
   Collapsing the last two is the error `¬Found(x) ⇏ ¬x` exists to prevent, so
   every quantity is also reported per epistemic state.
2. **Mixed points and curves.** Subsystems with a continuous score yield curves;
   subsystems whose abstention is categorical (`fabric`) yield single operating
   points. Both are first-class here — demanding a score everywhere would force
   new modeling where only measurement is warranted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


class SelectiveAnalysisError(ValueError):
    """Raised when a risk-coverage computation is not well defined."""


@dataclass(frozen=True)
class OperatingPoint:
    """One (coverage, risk) point, with the abstention ledger that produced it.

    `attempts` is the only denominator the measured system cannot shrink, which
    is why every rate here is denominated in it rather than in `presented`.
    """

    attempts: int
    presented: int
    wrong_presented: int
    threshold: float | None = None
    abstention_benefit: int | None = None
    abstention_cost: int | None = None
    label: str = ""
    per_state: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.attempts <= 0:
            raise SelectiveAnalysisError("attempts must be positive")
        if not 0 <= self.presented <= self.attempts:
            raise SelectiveAnalysisError("presented must lie in [0, attempts]")
        if not 0 <= self.wrong_presented <= self.presented:
            raise SelectiveAnalysisError("wrong_presented must lie in [0, presented]")
        benefit, cost = self.abstention_benefit, self.abstention_cost
        if benefit is not None and cost is not None:
            withheld = self.attempts - self.presented
            if benefit + cost != withheld:
                raise SelectiveAnalysisError(
                    f"abstention ledger does not close: {benefit} + {cost} != {withheld}"
                )

    @property
    def coverage(self) -> float:
        return self.presented / self.attempts

    @property
    def selective_risk(self) -> float:
        """Error conditioned on committing.

        Undefined at zero coverage. It is returned as 0.0 by convention *and*
        `is_degenerate` is True, so callers can refuse it. Never gate on this
        value alone -- that is precisely the fabric/slice.py:248 defect.
        """
        if self.presented == 0:
            return 0.0
        return self.wrong_presented / self.presented

    @property
    def retained_correct(self) -> int:
        return self.presented - self.wrong_presented

    @property
    def is_degenerate(self) -> bool:
        """True when selective risk is vacuous because nothing was presented."""
        return self.presented == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "threshold": self.threshold,
            "attempts": self.attempts,
            "presented": self.presented,
            "coverage": self.coverage,
            "wrong_presented": self.wrong_presented,
            "selective_risk": self.selective_risk,
            "retained_correct": self.retained_correct,
            "abstention_benefit": self.abstention_benefit,
            "abstention_cost": self.abstention_cost,
            "is_degenerate": self.is_degenerate,
            "per_state": dict(self.per_state),
        }


def risk_coverage_curve(points: Sequence[OperatingPoint]) -> list[OperatingPoint]:
    """Order operating points by increasing coverage."""

    if not points:
        raise SelectiveAnalysisError("a curve needs at least one operating point")
    attempts = {p.attempts for p in points}
    if len(attempts) != 1:
        raise SelectiveAnalysisError(
            f"operating points span different attempt counts: {sorted(attempts)}"
        )
    return sorted(points, key=lambda p: (p.coverage, p.selective_risk))


def aurc(points: Sequence[OperatingPoint]) -> float:
    """Area under the risk-coverage curve (trapezoidal), lower is better.

    AURC summarizes how well the score *ranks* correct above incorrect. It is
    only meaningful across a swept threshold, so it requires >= 2 points; a
    categorical abstention policy has no curve and must not be summarized here.
    """

    curve = risk_coverage_curve(points)
    if len(curve) < 2:
        raise SelectiveAnalysisError(
            "AURC needs >= 2 operating points; a categorical policy yields a "
            "point, not a curve"
        )
    area = 0.0
    for left, right in zip(curve, curve[1:]):
        width = right.coverage - left.coverage
        area += width * (left.selective_risk + right.selective_risk) / 2
    span = curve[-1].coverage - curve[0].coverage
    if span <= 0:
        raise SelectiveAnalysisError("operating points do not span any coverage")
    return area / span


def coverage_at_zero_risk(points: Sequence[OperatingPoint]) -> OperatingPoint | None:
    """Highest-coverage point achieving zero wrong-presented, if any.

    Degenerate points (nothing presented) are excluded: a mute system reaches
    zero risk trivially and is not an answer to "how much can we present
    safely?"
    """

    safe = [p for p in points if p.wrong_presented == 0 and not p.is_degenerate]
    return max(safe, key=lambda p: p.coverage) if safe else None


def coverage_cost_of(
    stricter: OperatingPoint, baseline: OperatingPoint
) -> dict[str, Any]:
    """What a stricter operating point paid, in the units that matter.

    This is the number `minimal_zero_wrong_presented_inclusive_v1` never
    reports: it proves risk reached zero and is silent on the coverage spent.
    """

    if stricter.attempts != baseline.attempts:
        raise SelectiveAnalysisError("points must share an attempt count")
    presented_delta = baseline.presented - stricter.presented
    return {
        "coverage_before": baseline.coverage,
        "coverage_after": stricter.coverage,
        "coverage_delta": stricter.coverage - baseline.coverage,
        "presented_given_up": presented_delta,
        "presented_given_up_fraction": (
            presented_delta / baseline.presented if baseline.presented else None
        ),
        "errors_removed": baseline.wrong_presented - stricter.wrong_presented,
        "retained_correct_before": baseline.retained_correct,
        "retained_correct_after": stricter.retained_correct,
        "correct_given_up": baseline.retained_correct - stricter.retained_correct,
    }


def gate_is_degeneracy_safe(
    point: OperatingPoint, coverage_floor: float
) -> tuple[bool, str]:
    """Admission check that a mute system cannot pass.

    The rule (papers/SELECTIVE_VOCABULARY.md): never denominate a gate solely in
    a quantity the gated system controls. `selective_risk` divides by
    `presented`, so it must be paired with a coverage floor denominated in
    `attempts` -- the shape `scribe/gate_grounded.py:129-133` already uses.
    """

    if not 0.0 < coverage_floor <= 1.0:
        raise SelectiveAnalysisError("coverage_floor must lie in (0, 1]")
    if point.is_degenerate:
        return False, "degenerate: nothing presented, selective risk is vacuous"
    if point.coverage < coverage_floor:
        return False, (
            f"coverage {point.coverage:.4f} below floor {coverage_floor:.4f}"
        )
    return True, "coverage floor satisfied"
