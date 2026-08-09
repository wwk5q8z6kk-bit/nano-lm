"""Tests for the surface-robustness aggregation and substitution."""

from __future__ import annotations

import pytest

from nano_ai.adapters.state_span import parse_state_span_summary
from nano_ai.surface import (
    MIN_SEEDS_FOR_ARM_CLAIM,
    ArmObservation,
    SurfaceArm,
    SurfaceError,
    aggregate,
    apply_arm,
    report_lines,
    substitute,
)
from nano_ai.surface_arms import (
    ALL_AXES,
    CONFLICTING_STRUCTURE_ARMS,
    CONFLICTING_VALUE_ARMS,
    DENIAL_ARMS,
    HEDGE_ARMS,
    TEMPLATE_ARMS,
    VALUE_ARMS,
    VALUE_TEMPLATE_FIELDS,
    _CALIBRATION_ALLERGY_VALUES,
    _CALIBRATION_MEDICATION_VALUES,
    _CALIBRATION_TEMPLATES,
    _CONFLICTING_ALLERGY_PAIRS,
    _CONFLICTING_MEDICATION_PAIRS,
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
    # `conflicting_structure` has no natural in-distribution counterpart: the
    # H6 training generator (`state_span_data.py::_variant_lines`, the
    # `conflicting` branch) always appends the repeated question/answer
    # immediately after the base five-turn block, with the alternative value
    # always second -- confirmed 2026-08-06 by reading the generator. Every
    # arm on this axis (ORDER, DISTANCE[n]) is a held-out structural
    # deviation by construction, not a sampling gap the way an unseen wording
    # is for denial/hedge/value/template.
    _AXES_WITHOUT_A_REFERENCE = {"conflicting_structure"}

    def test_every_axis_has_a_baseline_and_a_reference(self):
        for axis, arms in ALL_AXES.items():
            labels = [a.label for a in arms]
            assert "DEV" in labels, f"{axis} lacks the development baseline"
            if axis in self._AXES_WITHOUT_A_REFERENCE:
                continue
            assert any(a.in_distribution for a in arms), f"{axis} lacks a reference arm"

    def test_conflicting_structure_is_documented_as_wholly_held_out(self):
        # Pins the exception above: if this ever stops being true (a future
        # generator variant introduces order/distance variety), the test
        # suite -- not just the comment -- must fail so the exception gets
        # revisited rather than silently going stale.
        assert all(not a.in_distribution for a in CONFLICTING_STRUCTURE_ARMS)

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
        # ORDER/DISTANCE are author-constructed structural probes, not drawn
        # from any lexicon or the training distribution -- same obligation.
        for arm in CONFLICTING_STRUCTURE_ARMS:
            if arm.label != "DEV":
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


class TestValueTemplateArms:
    """`value` and `template` vary the world (which name / which frame) rather
    than a phrase's polarity. Their pools were verified against the actual
    dev.jsonl / calibration.jsonl records this session (progress.md, session
    3): dev's 24 medication + 24 allergy names and calibration's 16 + 16 match
    these module constants exactly, disjointly, and the fixed templates cover
    100% of both partitions' medication/allergy lines. These tests pin the
    structural invariants that verification depended on, without requiring the
    (untracked, regenerable) dataset files at test time.
    """

    def test_value_arms_cover_the_calibration_pool_one_arm_per_name_pair(self):
        # BASELINE + one TRAIN arm per (medication, allergy) pair in the
        # calibration partition -- zip(..., strict=True) in surface_arms.py
        # would already raise at import time if the two pools disagreed in
        # length, so this also pins that they didn't.
        assert len(_CALIBRATION_MEDICATION_VALUES) == len(_CALIBRATION_ALLERGY_VALUES)
        assert len(VALUE_ARMS) == 1 + len(_CALIBRATION_MEDICATION_VALUES)
        assert sum(a.in_distribution for a in VALUE_ARMS) == len(_CALIBRATION_MEDICATION_VALUES)

    def test_template_arms_cover_the_calibration_template_count(self):
        assert len(TEMPLATE_ARMS) == 1 + len(_CALIBRATION_TEMPLATES)
        assert sum(a.in_distribution for a in TEMPLATE_ARMS) == len(_CALIBRATION_TEMPLATES)

    def test_dev_and_calibration_value_pools_are_disjoint(self):
        # The property the axis depends on: swapping to a TRAIN value never
        # coincidentally reproduces a development-partition name.
        assert set(_DEV_MEDICATION_VALUES) & set(_CALIBRATION_MEDICATION_VALUES) == set()
        assert set(_DEV_ALLERGY_VALUES) & set(_CALIBRATION_ALLERGY_VALUES) == set()

    def test_value_and_template_arms_target_only_medication_and_allergy(self):
        assert VALUE_TEMPLATE_FIELDS == ("medication", "allergy")

    def test_value_train_arms_cite_the_calibration_partition(self):
        for arm in VALUE_ARMS:
            if arm.label == "DEV":
                continue
            assert "calibration partition" in arm.provenance
            assert "development not opened" in arm.provenance

    def test_template_train_arms_cite_the_calibration_partition(self):
        for arm in TEMPLATE_ARMS:
            if arm.label == "DEV":
                continue
            assert "calibration partition" in arm.provenance
            assert "development not opened" in arm.provenance

    def test_value_arm_rewrites_the_name_and_leaves_the_frame_alone(self):
        # Development-style line: fixed frame ("Only {VALUE} so far."), one
        # of the 24 dev medication names.
        dev_value = _DEV_MEDICATION_VALUES[0]
        line = _DEV_MEDICATION_TEMPLATE.replace("{VALUE}", dev_value)
        arm = next(a for a in VALUE_ARMS if a.label == "TRAIN[0]")
        cal_value = _CALIBRATION_MEDICATION_VALUES[0]
        assert substitute(line, arm) == _DEV_MEDICATION_TEMPLATE.replace("{VALUE}", cal_value)

    def test_value_arm_maps_every_dev_name_to_the_same_arm_value(self):
        # Every arm carries one fixed calibration name per field; in any real
        # document only one of the 24 dev source phrases is actually present,
        # so the other 23 pairs are no-ops there -- but the mapping itself
        # must send *any* dev name to that arm's value, not just the first.
        arm = next(a for a in VALUE_ARMS if a.label == "TRAIN[0]")
        cal_value = _CALIBRATION_MEDICATION_VALUES[0]
        for dev_value in _DEV_MEDICATION_VALUES:
            line = _DEV_MEDICATION_TEMPLATE.replace("{VALUE}", dev_value)
            assert substitute(line, arm) == _DEV_MEDICATION_TEMPLATE.replace(
                "{VALUE}", cal_value
            )

    def test_template_arm_rewrites_the_frame_and_leaves_the_name_alone(self):
        dev_value = _DEV_ALLERGY_VALUES[0]
        line = _DEV_ALLERGY_TEMPLATE.replace("{VALUE}", dev_value)
        arm = next(a for a in TEMPLATE_ARMS if a.label == "TRAIN[2]")  # "[{VALUE}]"
        assert substitute(line, arm) == f"[{dev_value}]"

    def test_value_arm_mapping_has_no_ambiguous_substring_collisions(self):
        # substitute() raises SurfaceError on >1 occurrence of a source phrase;
        # a same-arm collision between two of its own source strings would
        # make every real document unusable for that arm. Exercise every
        # TRAIN arm against one full round of dev-template lines built from
        # every dev value, for both fields.
        for arm in VALUE_ARMS:
            if arm.label == "DEV":
                continue
            for dev_value in _DEV_MEDICATION_VALUES:
                substitute(_DEV_MEDICATION_TEMPLATE.replace("{VALUE}", dev_value), arm)
            for dev_value in _DEV_ALLERGY_VALUES:
                substitute(_DEV_ALLERGY_TEMPLATE.replace("{VALUE}", dev_value), arm)

    def test_template_arm_mapping_has_no_ambiguous_substring_collisions(self):
        for arm in TEMPLATE_ARMS:
            if arm.label == "DEV":
                continue
            for dev_value in _DEV_MEDICATION_VALUES:
                substitute(_DEV_MEDICATION_TEMPLATE.replace("{VALUE}", dev_value), arm)
            for dev_value in _DEV_ALLERGY_VALUES:
                substitute(_DEV_ALLERGY_TEMPLATE.replace("{VALUE}", dev_value), arm)


class TestApplyArm:
    """`apply_arm` is the pure entry point both the mapping path (denial,
    hedge, value, template) and the transform path (conflicting_*) share.
    """

    def test_mapping_arm_behaves_exactly_like_substitute_plus_target_rewrite(self):
        arm = SurfaceArm(
            label="A", axis="denial",
            mapping=(("Nothing at all.", "No, nothing."),),
            provenance="test",
        )
        transcript = "Doctor: Meds?\nPatient: Nothing at all."
        target = "MED:A[Nothing at all.]"
        out = apply_arm(transcript, target, arm)
        assert out == ("Doctor: Meds?\nPatient: No, nothing.", "MED:A[No, nothing.]")

    def test_mapping_arm_returns_none_on_ambiguous_rewrite(self):
        arm = SurfaceArm(
            label="A", axis="denial", mapping=(("old", "new"),), provenance="test",
        )
        assert apply_arm("old and old", "X:A[old]", arm) is None

    def test_transform_arm_delegates_to_the_transform(self):
        def transform(transcript, target):
            return transcript.upper(), target.upper()

        arm = SurfaceArm(
            label="T", axis="conflicting_structure", mapping=(), provenance="test",
            transform=transform,
        )
        assert apply_arm("hi", "x:s[hi]", arm) == ("HI", "X:S[HI]")

    def test_transform_arm_returning_none_is_a_drop(self):
        arm = SurfaceArm(
            label="T", axis="conflicting_structure", mapping=(), provenance="test",
            transform=lambda transcript, target: None,
        )
        assert apply_arm("hi", "x:s[hi]", arm) is None

    def test_an_arm_cannot_carry_both_a_mapping_and_a_transform(self):
        arm = SurfaceArm(
            label="Bad", axis="conflicting_structure",
            mapping=(("a", "b"),), provenance="test",
            transform=lambda transcript, target: (transcript, target),
        )
        with pytest.raises(SurfaceError, match="both a mapping and a transform"):
            apply_arm("a", "x:s[a]", arm)


class TestConflictingArms:
    """`conflicting` is the one state whose 2026-08-05/06 held-out drop (30.3
    points) had no disjoint phrase pool behind it, so it is the strongest
    candidate for a genuinely structural failure. These fixtures mirror the
    real dev.jsonl schema by hand (dev.jsonl itself is untracked, so tests
    must not depend on reading it -- same discipline as TestValueTemplateArms).
    """

    # A conflicting-duration document: the base five-turn block, then the
    # duration question repeated with a different answer at the end -- the
    # exact topology `state_span_data.py::_variant_lines` always produces.
    _DURATION_TRANSCRIPT = (
        "Doctor: What's the issue today?\n"
        "Patient: It started as knee pain and hasn't stopped.\n"
        "Doctor: How many days has it been?\n"
        "Patient: Started roughly 5 days prior.\n"
        "Doctor: On a scale from mild to severe, where is it?\n"
        "Patient: Pretty mild.\n"
        "Doctor: Did you try any medicine?\n"
        "Patient: Nothing at all.\n"
        "Doctor: Do you have any known allergies?\n"
        "Patient: None whatsoever.\n"
        "Doctor: How many days has it been?\n"
        "Patient: Started roughly 6 days prior."
    )
    _DURATION_TARGET = (
        "CC:S[knee pain]|DUR:C[5 days;6 days]|SEV:S[mild]"
        "|MED:A[Nothing at all.]|ALG:A[None whatsoever.]"
    )

    # A conflicting-medication document where one of the two conflicting
    # values IS the denial phrase, and the target's bracket order is the
    # REVERSE of physical transcript order -- the exact shape of the real
    # `dev-0008-conflicting` record that caught the first version of
    # `_make_conflicting_value_transform` guessing bracket order from
    # physical position instead of reading it.
    _MEDICATION_WITH_DENIAL_TRANSCRIPT = (
        "Doctor: What's the issue today?\n"
        "Patient: It started as knee pain and hasn't stopped.\n"
        "Doctor: How many days has it been?\n"
        "Patient: Started roughly 5 days prior.\n"
        "Doctor: On a scale from mild to severe, where is it?\n"
        "Patient: Pretty mild.\n"
        "Doctor: Did you try any medicine?\n"
        "Patient: Nothing at all.\n"
        "Doctor: Do you have any known allergies?\n"
        "Patient: None whatsoever.\n"
        "Doctor: Did you try any medicine?\n"
        "Patient: Only capsaicin cream so far."
    )
    _MEDICATION_WITH_DENIAL_TARGET = (
        "CC:S[knee pain]|DUR:S[5 days]|SEV:S[mild]"
        "|MED:C[capsaicin cream;Nothing at all.]|ALG:A[None whatsoever.]"
    )

    _NORMAL_TRANSCRIPT = (
        "Doctor: What's the issue today?\n"
        "Patient: It started as knee pain and hasn't stopped.\n"
        "Doctor: How many days has it been?\n"
        "Patient: Started roughly 5 days prior.\n"
        "Doctor: On a scale from mild to severe, where is it?\n"
        "Patient: Pretty mild.\n"
        "Doctor: Did you try any medicine?\n"
        "Patient: Nothing at all.\n"
        "Doctor: Do you have any known allergies?\n"
        "Patient: None whatsoever."
    )
    _NORMAL_TARGET = (
        "CC:S[knee pain]|DUR:S[5 days]|SEV:S[mild]"
        "|MED:A[Nothing at all.]|ALG:A[None whatsoever.]"
    )

    # -- conflicting_value ---------------------------------------------

    def test_value_arm_is_a_no_op_for_fields_without_an_open_pool(self):
        for arm in CONFLICTING_VALUE_ARMS:
            if arm.label == "DEV":
                continue
            assert apply_arm(self._DURATION_TRANSCRIPT, self._DURATION_TARGET, arm) is None

    def test_value_arm_handles_denial_as_one_of_the_two_conflicting_values(self):
        # Regression test: bracket order in `target` is the reverse of
        # physical transcript order here. The fix reads the field's own
        # C[...] segment directly instead of assuming an order.
        arm = next(a for a in CONFLICTING_VALUE_ARMS if a.label == "TRAIN[0]")
        med_a, med_b = _CONFLICTING_MEDICATION_PAIRS[0]
        out = apply_arm(
            self._MEDICATION_WITH_DENIAL_TRANSCRIPT,
            self._MEDICATION_WITH_DENIAL_TARGET,
            arm,
        )
        assert out is not None
        transcript, target = out
        assert f"Patient: {med_a}" in transcript  # replaced "Nothing at all."
        assert f"Only {med_b} so far." in transcript  # replaced "capsaicin cream"
        assert f"MED:C[{med_a};{med_b}]" in target
        # Re-parsing must locate both spans uniquely -- the same defense the
        # harness's `_apply` applies before ever scoring a rewritten example.
        proposals = parse_state_span_summary(target, transcript)
        conflicting = next(p for p in proposals if p.state.value == "conflicting")
        assert {s.text for s in conflicting.spans} == {med_a, med_b}

    def test_value_arm_is_a_no_op_on_a_document_with_no_conflicting_field(self):
        for arm in CONFLICTING_VALUE_ARMS:
            if arm.label == "DEV":
                continue
            assert apply_arm(self._NORMAL_TRANSCRIPT, self._NORMAL_TARGET, arm) is None

    def test_value_arms_cover_the_calibration_pool_and_pairs_are_distinct(self):
        assert len(CONFLICTING_VALUE_ARMS) == 1 + len(_CONFLICTING_MEDICATION_PAIRS)
        assert len(_CONFLICTING_MEDICATION_PAIRS) == len(_CONFLICTING_ALLERGY_PAIRS)
        for med_a, med_b in _CONFLICTING_MEDICATION_PAIRS:
            assert med_a != med_b
        for alg_a, alg_b in _CONFLICTING_ALLERGY_PAIRS:
            assert alg_a != alg_b

    # -- conflicting_structure -------------------------------------------

    def test_order_swap_exchanges_the_two_physical_positions(self):
        arm = next(a for a in CONFLICTING_STRUCTURE_ARMS if a.label == "ORDER")
        transcript, target = apply_arm(
            self._DURATION_TRANSCRIPT, self._DURATION_TARGET, arm
        )
        assert "Started roughly 6 days prior." in transcript.splitlines()[3]
        assert "Started roughly 5 days prior." in transcript.splitlines()[-1]
        # gold is a set -- the target string does not need to change
        assert target == self._DURATION_TARGET
        proposals = parse_state_span_summary(target, transcript)
        conflicting = next(p for p in proposals if p.state.value == "conflicting")
        assert {s.text for s in conflicting.spans} == {"5 days", "6 days"}

    def test_order_swap_applies_to_fields_without_an_open_value_pool(self):
        # Unlike conflicting_value, conflicting_structure works on ANY field
        # -- it only moves existing text, never invents new values.
        arm = next(a for a in CONFLICTING_STRUCTURE_ARMS if a.label == "ORDER")
        assert apply_arm(self._DURATION_TRANSCRIPT, self._DURATION_TARGET, arm) is not None

    def test_distance_arm_inserts_filler_and_preserves_values_and_target(self):
        arm = next(a for a in CONFLICTING_STRUCTURE_ARMS if a.label == "DISTANCE[3]")
        transcript, target = apply_arm(
            self._DURATION_TRANSCRIPT, self._DURATION_TARGET, arm
        )
        assert len(transcript) > len(self._DURATION_TRANSCRIPT)
        assert target == self._DURATION_TARGET
        proposals = parse_state_span_summary(target, transcript)
        conflicting = next(p for p in proposals if p.state.value == "conflicting")
        assert {s.text for s in conflicting.spans} == {"5 days", "6 days"}

    def test_distance_grows_with_the_requested_filler_count(self):
        lengths = {}
        for label in ("DISTANCE[1]", "DISTANCE[3]", "DISTANCE[6]"):
            arm = next(a for a in CONFLICTING_STRUCTURE_ARMS if a.label == label)
            transcript, _ = apply_arm(
                self._DURATION_TRANSCRIPT, self._DURATION_TARGET, arm
            )
            lengths[label] = len(transcript)
        assert lengths["DISTANCE[1]"] < lengths["DISTANCE[3]"] < lengths["DISTANCE[6]"]

    def test_structure_arms_are_a_no_op_on_a_document_with_no_conflicting_field(self):
        for arm in CONFLICTING_STRUCTURE_ARMS:
            if arm.label == "DEV":
                continue
            assert apply_arm(self._NORMAL_TRANSCRIPT, self._NORMAL_TARGET, arm) is None

    def test_dev_baseline_is_a_true_no_op_on_both_axes(self):
        for arms in (CONFLICTING_VALUE_ARMS, CONFLICTING_STRUCTURE_ARMS):
            arm = next(a for a in arms if a.label == "DEV")
            out = apply_arm(self._DURATION_TRANSCRIPT, self._DURATION_TARGET, arm)
            assert out == (self._DURATION_TRANSCRIPT, self._DURATION_TARGET)
