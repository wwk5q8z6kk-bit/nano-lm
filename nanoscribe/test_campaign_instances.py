"""Mechanical validation of the multi-instance campaign_v2 instrument.

Hand-authored surface values are exactly the kind of data that looks right and
is not. Every invariant the instrument depends on is asserted here rather than
inspected, because each failure mode is silent:

- a duplicated present value makes the constrained selector abstain on an
  ambiguous quote, which then reads as a model failure;
- an absent value that actually occurs turns the false-positive probe into a
  normal extraction slot;
- a value colliding with the neutral system-prompt examples re-opens the very
  leakage channel the C1-off cell exists to close.
"""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from nanoscribe.adapters import DEFAULT_BASELINE_LINES, default_baseline_specs
from nanoscribe.campaign_datasets import (
    CAMPAIGN_V2_BASE_ENCOUNTERS,
    campaign_cases,
    case_for,
    fixture_lines_for_encounter,
    instance_cases,
)
from nanoscribe.campaign_instances import (
    INSTANCE_IDS,
    INSTANCES,
    RESERVED_VALUES,
    split_encounter_id,
)
from nanoscribe.select import match_count

SLOTS_PER_INSTANCE = 16

_ORIGINAL_ENC1_TURNS = (
    "What brings you in today?",
    "My neck has been hurting.",
    "I think this is cervical strain.",
    "No allergies.",
    "I used to have migraines years ago.",
)


class InstanceInvariantTest(unittest.TestCase):
    def test_every_present_value_occurs_exactly_once(self) -> None:
        for values in INSTANCES:
            by_base = {
                split_encounter_id(case.encounter_id)[0]: case
                for case in instance_cases(values.instance_id)
            }
            for base, value in values.present_values():
                source = by_base[base].model_input.source
                self.assertEqual(
                    match_count(source, value),
                    1,
                    f"{values.instance_id}/{base}: {value!r} must occur exactly once",
                )

    def test_every_absent_value_occurs_zero_times(self) -> None:
        for values in INSTANCES:
            case = case_for("enc-4", values.instance_id)
            source = case.model_input.source
            for _atom_id, _atom_type, value in values.e4_absent:
                self.assertEqual(
                    match_count(source, value),
                    0,
                    f"{values.instance_id}/enc-4: {value!r} must not occur",
                )

    def test_no_value_collides_with_the_neutral_system_examples(self) -> None:
        reserved = {item.casefold() for item in RESERVED_VALUES}
        for values in INSTANCES:
            for value in values.all_values():
                self.assertNotIn(
                    value.casefold(),
                    reserved,
                    f"{values.instance_id}: {value!r} collides with a system example",
                )

    def test_instances_are_structurally_identical(self) -> None:
        shapes = set()
        for instance_id in INSTANCE_IDS:
            cases = instance_cases(instance_id)
            self.assertEqual(
                [split_encounter_id(case.encounter_id)[0] for case in cases],
                list(CAMPAIGN_V2_BASE_ENCOUNTERS),
            )
            slots = sum(len(case.atom_specs) for case in cases)
            types = Counter(
                spec.atom_type for case in cases for spec in case.atom_specs
            )
            states = Counter(
                atom.assertion_state for case in cases for atom in case.gold.atoms
            )
            unresolved = sum(len(case.gold.unresolved) for case in cases)
            shapes.add(
                (slots, tuple(sorted(types.items())), tuple(sorted(states.items())), unresolved)
            )
        self.assertEqual(len(shapes), 1, "instances differ in structure, not just values")
        self.assertEqual(next(iter(shapes))[0], SLOTS_PER_INSTANCE)

    def test_suite_is_five_instances_of_sixteen_slots(self) -> None:
        cases = campaign_cases("campaign_v2")
        self.assertEqual(len(cases), len(INSTANCE_IDS) * len(CAMPAIGN_V2_BASE_ENCOUNTERS))
        self.assertEqual(
            sum(len(case.atom_specs) for case in cases),
            len(INSTANCE_IDS) * SLOTS_PER_INSTANCE,
        )
        self.assertEqual(len({case.encounter_id for case in cases}), len(cases))

    def test_source_ids_are_unique_across_instances(self) -> None:
        source_ids = [
            source.source_id
            for case in campaign_cases("campaign_v2")
            for source in case.gold.sources
        ]
        self.assertEqual(len(set(source_ids)), len(source_ids))


class I0FidelityTest(unittest.TestCase):
    """i0 must reproduce the original hand-authored data exactly.

    Every claim previously scored on campaign_v1 was measured against these
    encounters; if i0 drifts, the historical partition stops being comparable
    and the reproduction check in the ablation is worthless.
    """

    def test_i0_reproduces_the_original_enc1_source(self) -> None:
        source = case_for("enc-1", "i0").model_input.source
        self.assertEqual(
            tuple(turn.text for turn in source.turns), _ORIGINAL_ENC1_TURNS
        )
        self.assertEqual(source.source_id, "src-1")

    def test_i0_specs_match_the_shipped_baseline_specs(self) -> None:
        self.assertEqual(case_for("enc-1", "i0").atom_specs, default_baseline_specs())

    def test_i0_fixture_lines_match_the_shipped_baseline_lines(self) -> None:
        self.assertEqual(fixture_lines_for_encounter("enc-1"), DEFAULT_BASELINE_LINES)

    def test_campaign_v1_partition_is_still_single_instance_i0(self) -> None:
        cases = campaign_cases("campaign_v1")
        self.assertEqual(
            [case.encounter_id for case in cases], ["enc-1", "enc-2", "enc-3"]
        )
        self.assertEqual(sum(len(case.atom_specs) for case in cases), 8)


class EnvVarSpellingTest(unittest.TestCase):
    """Pin the misspelled weights env var so a cleanup cannot break the runner.

    The env var name is missing an R, and that misspelling is what every reader
    and writer in the runner uses — consistently. The correctly-spelled
    occurrences elsewhere in the repo are references to the filename
    ``papers/NANOSCRIBE_VNEXT.md``, not environment variables, so this is a
    latent hazard rather than a live split: a well-meaning spell-fix in one
    file would silently stop the runner finding its weights and fall back to
    fixture lines. Behaviour is deliberately left alone; renaming is a separate
    change that must update every site at once and re-pin this test.
    """

    # Assembled, not written literally, so this file does not match its own scan.
    WEIGHTS_ENV = "NANOS" + "CIBE_QWEN_WEIGHTS"
    CORRECTED_ENV = "NANOS" + "CRIBE_QWEN_WEIGHTS"

    def _runner_sources(self):
        root = Path(_repo_root)
        return [
            path
            for path in sorted(root.glob("nanoscribe/*.py"))
            if path.name != Path(__file__).name
        ]

    def test_every_runner_site_uses_the_same_spelling(self) -> None:
        expected = {
            "nanoscribe/qwen_inference.py",
            "nanoscribe/run_eval.py",
            "nanoscribe/smoke_qwen_baseline.py",
            "nanoscribe/test_adapt.py",
        }
        root = Path(_repo_root)
        found = {
            str(path.relative_to(root))
            for path in self._runner_sources()
            if self.WEIGHTS_ENV in path.read_text()
        }
        self.assertEqual(found, expected)

    def test_no_site_uses_the_corrected_spelling(self) -> None:
        root = Path(_repo_root)
        offenders = [
            str(path.relative_to(root))
            for path in self._runner_sources()
            if self.CORRECTED_ENV in path.read_text()
        ]
        self.assertEqual(
            offenders,
            [],
            "a corrected spelling appeared without updating every reader/writer",
        )

    def test_resolver_reads_the_pinned_name(self) -> None:
        import os

        from nanoscribe.qwen_inference import resolve_weights_path

        saved = os.environ.pop(self.WEIGHTS_ENV, None)
        try:
            os.environ[self.WEIGHTS_ENV] = "/tmp/pinned-weights"
            self.assertEqual(resolve_weights_path(None), "/tmp/pinned-weights")
        finally:
            os.environ.pop(self.WEIGHTS_ENV, None)
            if saved is not None:
                os.environ[self.WEIGHTS_ENV] = saved


if __name__ == "__main__":
    unittest.main()
