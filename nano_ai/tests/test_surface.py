"""Tests for the surface-robustness aggregation and substitution."""

from __future__ import annotations

import pytest

from nano_ai.surface import (
    MIN_SEEDS_FOR_ARM_CLAIM,
    ArmObservation,
    SurfaceArm,
    SurfaceError,
    aggregate,
    report_lines,
    substitute,
)
from nano_ai.surface_arms import (
    ALL_AXES,
    DENIAL_ARMS,
    HEDGE_ARMS,
    TEMPLATE_ARMS,
    VALUE_ARMS,
    VALUE_TEMPLATE_FIELDS,
    _CALIBRATION_ALLERGY_VALUES,
    _CALIBRATION_MEDICATION_VALUES,
    _CALIBRATION_TEMPLATES,
    _DEV_ALLERGY_VALUES,
    _DEV_ALLERGY_TEMPLATE,
    _DEV_MEDICATION_VALUES,
    _DEV_MEDICATION_TEMPLATE,
)


def _arm(mapping=(("old", "new"),)) -> SurfaceArm:
    return SurfaceArm(label="A", axis="denial", mapping=mapping, provenance="test")


def _obs(arm, seed, correct, total=100, *, state="absent", axis="denial", ind=False):
    return ArmObservation(
        arm=arm, axis=axis, seed=seed, state=state,
        correct=correct, total=total, in_distribution=ind,
    )


class TestSubstitution:
    def test_replaces_a_unique_occurrence(self):
        assert substitute("Patient: old", _arm()) == "Patient: new"

    def test_absent_phrase_is_a_no_op(self):
        assert substitute("Patient: hello", _arm()) == "Patient: hello"

    def test_ambiguous_rewrite_raises_rather_than_corrupting_spans(self):
        # Two occurrences means the gold span cannot be re-located unambiguously.
        with pytest.raises(SurfaceError, match="ambiguous"):
            substitute("old and old", _arm())

    def test_uniqueness_can_be_waived_explicitly(self):
        assert substitute("old old", _arm(), require_unique=False) == "new new"

    def test_rewrites_apply_in_order_over_multiple_pairs(self):
        arm = _arm(mapping=(("a", "b"), ("c", "d")))
        assert substitute("a c", arm) == "b d"


class TestAggregate:
    def test_returns_none_when_no_rows_match(self):
        assert aggregate([_obs("A", "s1", 50)], axis="hedge", state="absent") is None

    def test_zero_total_rows_are_excluded(self):
        assert aggregate([_obs("A", "s1", 0, total=0)], axis="denial", state="absent") is None

    def test_robust_is_the_minimum_over_arm_means(self):
        rows = [_obs("A", "s1", 90), _obs("B", "s1", 40), _obs("C", "s1", 70)]
        out = aggregate(rows, axis="denial", state="absent")
        assert out["surface_robust_accuracy"] == pytest.approx(0.40)
        assert out["worst_arm"] == "B"
        assert out["best_arm"] == "A"

    def test_arm_mean_precedes_the_minimum(self):
        # Arm B: seeds 20% and 80% -> mean 50%, which must beat arm A's 40%.
        # Taking min over raw observations would wrongly report 20%.
        rows = [_obs("A", "s1", 40), _obs("A", "s2", 40),
                _obs("B", "s1", 20), _obs("B", "s2", 80)]
        out = aggregate(rows, axis="denial", state="absent")
        assert out["surface_robust_accuracy"] == pytest.approx(0.40)
        assert out["worst_arm"] == "A"

    def test_sensitivity_is_the_span_of_arm_means(self):
        rows = [_obs("A", "s1", 90), _obs("B", "s1", 30)]
        out = aggregate(rows, axis="denial", state="absent")
        assert out["surface_sensitivity"] == pytest.approx(0.60)

    def test_instability_averages_within_arm_seed_spread(self):
        rows = [_obs("A", "s1", 20), _obs("A", "s2", 60),   # spread 0.40
                _obs("B", "s1", 50), _obs("B", "s2", 70)]   # spread 0.20
        out = aggregate(rows, axis="denial", state="absent")
        assert out["seed_instability"] == pytest.approx(0.30)

    def test_instability_is_unmeasured_with_one_seed(self):
        out = aggregate([_obs("A", "s1", 50), _obs("B", "s1", 90)],
                        axis="denial", state="absent")
        assert out["seed_instability"] is None
        assert out["arm_comparison_supported"] is False

    def test_in_distribution_filter_selects_the_reference_arms(self):
        rows = [_obs("T", "s1", 95, ind=True), _obs("X", "s1", 40, ind=False)]
        held = aggregate(rows, axis="denial", state="absent", in_distribution=False)
        ref = aggregate(rows, axis="denial", state="absent", in_distribution=True)
        assert held["arm_means"] == {"X": 0.40}
        assert ref["arm_means"] == {"T": 0.95}
        assert aggregate(rows, axis="denial", state="absent")["arms"] == 2


class TestArmComparisonGuard:
    """The guard exists because 2026-08-05 measured instability > sensitivity."""

    def test_unsupported_when_instability_swamps_sensitivity(self):
        # Arms differ by 5 points; each arm moves 40 points across seeds.
        rows = [_obs("A", "s1", 30), _obs("A", "s2", 70),
                _obs("B", "s1", 35), _obs("B", "s2", 75)]
        out = aggregate(rows, axis="denial", state="absent")
        assert out["surface_sensitivity"] < out["seed_instability"]
        assert out["arm_comparison_supported"] is False
        assert any("NOT supported" in line for line in report_lines(out))

    def test_unsupported_below_the_seed_floor_even_when_stable(self):
        seeds = [f"s{i}" for i in range(MIN_SEEDS_FOR_ARM_CLAIM - 1)]
        rows = [_obs("A", s, 90) for s in seeds] + [_obs("B", s, 30) for s in seeds]
        out = aggregate(rows, axis="denial", state="absent")
        assert out["surface_sensitivity"] == pytest.approx(0.60)
        assert out["seed_instability"] == pytest.approx(0.0)
        assert out["arm_comparison_supported"] is False

    def test_supported_with_enough_seeds_and_a_real_effect(self):
        seeds = [f"s{i}" for i in range(MIN_SEEDS_FOR_ARM_CLAIM)]
        rows = [_obs("A", s, 90) for s in seeds] + [_obs("B", s, 30) for s in seeds]
        out = aggregate(rows, axis="denial", state="absent")
        assert out["arm_comparison_supported"] is True
        assert not any("NOT supported" in line for line in report_lines(out))


class TestArmDefinitions:
    def test_every_axis_has_a_baseline_and_a_reference(self):
        for axis, arms in ALL_AXES.items():
            labels = [a.label for a in arms]
            assert "DEV" in labels, f"{axis} lacks the development baseline"
            assert any(a.in_distribution for a in arms), f"{axis} lacks a reference arm"

    def test_arm_labels_are_unique_within_an_axis(self):
        for axis, arms in ALL_AXES.items():
            labels = [a.label for a in arms]
            assert len(labels) == len(set(labels)), f"{axis} has duplicate arm labels"

    def test_every_arm_declares_provenance(self):
        for arms in ALL_AXES.values():
            for arm in arms:
                assert arm.provenance.strip(), f"{arm.label} has no provenance"

    def test_arms_carry_their_declared_axis(self):
        for axis, arms in ALL_AXES.items():
            assert all(a.axis == axis for a in arms)

    def test_constructed_arms_are_marked_not_independent(self):
        # Author-written wordings must never be read as external evidence.
        for arm in HEDGE_ARMS:
            if arm.label.startswith("CONSTRUCTED"):
                assert "NOT independent" in arm.provenance

    def test_external_denial_arms_cite_a_lexicon(self):
        external = [a for a in DENIAL_ARMS if a.label.startswith("EXTERNAL")]
        assert len(external) >= 8
        for arm in external:
            assert "MIT" in arm.provenance and "lexicon" in arm.provenance

    def test_reference_arms_are_not_the_held_out_baseline(self):
        for arms in ALL_AXES.values():
            baseline = next(a for a in arms if a.label == "DEV")
            assert baseline.in_distribution is False
