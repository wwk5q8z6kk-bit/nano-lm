"""Pin the degeneracy-safety of the H6 quality gates.

Context. `fabric/slice.py` once computed `presented_err / max(1, presented)`,
which a verifier could pass by abstaining on everything -- a conditional-risk
bar denominated in a quantity the system controls. On 2026-08-05 the same shape
was reintroduced in a preregistered criterion (`PREREG_DENIAL_POLARITY.md` C1,
denominated in the recoverable population, which the data drove to ~4).

H6's gates do *not* have that defect, but only because of how they are
composed: `zero_wrong_presented` is a count the model can trivially zero by
presenting nothing, and it is safe solely because it is ANDed with semantic
gates that are absolute numerator floors over fixed gold denominators. Abstain
on everything and those numerators go to zero, so the composite fails.

That property is load-bearing and was never written down. These tests pin it, so
a future refactor that relaxes the semantic floors -- or denominates them in
anything the model controls -- fails here rather than silently restoring the
degeneracy.
"""

from __future__ import annotations

import pytest

from nano_ai.training.evaluate_evidence_query_h6 import (
    _SEMANTIC_GATES,
    _quality_gates,
)


def _metric(numerator: int, denominator: int) -> dict[str, object]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _metrics(*, fraction: float, false_presented: int, failures: int = 0):
    """Build a metrics mapping where every gate is met at `fraction` of gold."""
    out: dict[str, object] = {
        name: _metric(int(denominator * fraction), denominator)
        for name, (_, denominator) in _SEMANTIC_GATES.items()
    }
    out["failures"] = _metric(failures, 1_000)
    # false_presented is denominated in *presented*, which the model controls.
    out["false_presented"] = _metric(false_presented, max(false_presented, 1))
    return out


class TestAbstainEverythingIsRejected:
    def test_presenting_nothing_fails_the_composite_gate(self):
        """The degeneracy: zero errors because zero assertions."""
        metrics = _metrics(fraction=0.0, false_presented=0)
        result = _quality_gates(metrics, require_zero_wrong_presented=True)
        assert result["all_quality_gates_passed"] is False

    def test_the_risk_gate_alone_would_have_been_fooled(self):
        """Documents *why* the composite is safe -- the risk bar is not."""
        metrics = _metrics(fraction=0.0, false_presented=0)
        result = _quality_gates(metrics, require_zero_wrong_presented=True)
        assert result["gate_evidence"]["zero_wrong_presented"]["passed"] is True
        assert result["semantic_and_retention_passed"] is False

    def test_every_semantic_gate_rejects_the_abstainer_individually(self):
        metrics = _metrics(fraction=0.0, false_presented=0)
        result = _quality_gates(metrics, require_zero_wrong_presented=True)
        for name in _SEMANTIC_GATES:
            assert result["gate_evidence"][name]["passed"] is False, name


class TestSemanticFloorsAreGoldDenominated:
    """The safety comes from the denominators. Pin that they are fixed."""

    @pytest.mark.parametrize("name", sorted(_SEMANTIC_GATES))
    def test_denominator_is_a_positive_constant(self, name):
        minimum, denominator = _SEMANTIC_GATES[name]
        assert isinstance(denominator, int) and denominator > 0

    @pytest.mark.parametrize("name", sorted(_SEMANTIC_GATES))
    def test_minimum_is_a_substantial_share_of_gold(self, name):
        """A floor near zero would re-open the degeneracy."""
        minimum, denominator = _SEMANTIC_GATES[name]
        assert minimum / denominator >= 0.5, (
            f"{name} floor is {minimum}/{denominator}; a low floor lets an "
            "abstaining model satisfy the semantic gates"
        )

    def test_a_metric_denominated_in_a_model_controlled_quantity_is_refused(self):
        """`_require_metric` enforces the expected denominator."""
        from nano_ai.training.evaluate_evidence_query_h6 import (
            EvidenceQueryEvaluationError,
        )

        metrics = _metrics(fraction=1.0, false_presented=0)
        # Simulate a refactor that denominated `absence` in what was presented.
        metrics["absence"] = _metric(5, 5)
        with pytest.raises(EvidenceQueryEvaluationError, match="absence"):
            _quality_gates(metrics, require_zero_wrong_presented=True)


class TestGatePassesOnGenuinelyGoodMetrics:
    def test_full_coverage_and_no_errors_passes(self):
        metrics = _metrics(fraction=1.0, false_presented=0)
        result = _quality_gates(metrics, require_zero_wrong_presented=True)
        assert result["all_quality_gates_passed"] is True

    def test_errors_while_presenting_fails_the_risk_gate(self):
        metrics = _metrics(fraction=1.0, false_presented=3)
        result = _quality_gates(metrics, require_zero_wrong_presented=True)
        assert result["gate_evidence"]["zero_wrong_presented"]["passed"] is False
        assert result["all_quality_gates_passed"] is False

    def test_risk_gate_is_skipped_when_not_required(self):
        metrics = _metrics(fraction=1.0, false_presented=3)
        result = _quality_gates(metrics, require_zero_wrong_presented=False)
        assert "zero_wrong_presented" not in result["gate_evidence"]
        assert result["all_quality_gates_passed"] is True

    def test_excess_failures_fail_the_gate(self):
        metrics = _metrics(fraction=1.0, false_presented=0, failures=11)
        result = _quality_gates(metrics, require_zero_wrong_presented=True)
        assert result["gate_evidence"]["failures"]["passed"] is False
        assert result["all_quality_gates_passed"] is False
