"""Adversarial checks for the V6 temporal scorer; synthetic records only, no training.

Default execution is read-only for project artifacts. --out exclusively creates
a NEW check receipt and preserves any existing output.
"""
from __future__ import annotations

import argparse
from dataclasses import FrozenInstanceError, replace
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
from correction_protocol import Event, Manifest, WorldSpec
from temporal_revision_scoring_v6 import (
    Plan, Prediction, Time, contract, contract_sha256, early_adaptation_plan,
    final_truth_plan, phase_end, score, valid_intervals, write_report,
)


def fixture(rows=((0, 0, 0), (10, 0, 1), (20, 0, 2)), phases=40):
    """Independent rows: (phase, entity, value), with one relation."""
    events, latest = [], {}
    for phase, entity, value in rows:
        prior = latest.get((entity, 0))
        event = Event(f"e{len(events)}", phase, len(events), entity, 0, value,
                      None if prior is None else prior.event_id,
                      None if prior is None else prior.value)
        events.append(event)
        latest[event.key] = event
    return Manifest(7, WorldSpec(phases=phases, entities_per_phase=2, relations=1,
                                shown_relations=1, values=5, assignments_per_phase=1),
                    tuple(events))


def predict(plan, values):
    eligible = [case for case in plan.cases if case.status == "eligible"]
    assert len(eligible) == len(values)
    return tuple(Prediction(c.case_id, c.requested_time, value, plan.sha256)
                 for c, value in zip(eligible, values))


class TemporalRevisionTests(unittest.TestCase):
    def test_01_frozen_contract_identity_and_copy(self):
        self.assertEqual(contract_sha256(), "407fc15ceb89cc51662a28990bc7979c4bdb6325a46e4f07234fb5fb73cf45b1")
        copy = contract()
        copy["denominators"] = "tampered"
        self.assertNotEqual(copy, contract())

    def test_02_abc_final_truth_rewards_current_and_rejects_stale(self):
        manifest = fixture()
        plan = final_truth_plan(manifest, ("e1",), through_phase=39)
        self.assertEqual(score(plan, predict(plan, (2,)))["counts"]["accuracy"], 1)
        self.assertEqual(score(plan, predict(plan, (1,)))["counts"]["accuracy"], 0)
        case = plan.cases[0]
        self.assertEqual(case.anchor_event_id, "e1")
        self.assertEqual(case.target_provenance.source_id, "e2")
        self.assertEqual(case.target_provenance.source, "observed_target")
        self.assertEqual(case.target_validity.valid_from, Time(20, 2))

    def test_03_abc_early_adaptation_uses_requested_time(self):
        plan = early_adaptation_plan(fixture(), ("e1", "e2"), offset=1,
                                     through_phase=39, censor_policy="new_revision")
        self.assertEqual([c.requested_time.phase for c in plan.cases], [11, 21])
        self.assertEqual(score(plan, predict(plan, (1, 2)))["counts"]["accuracy"], 1)
        self.assertEqual(score(plan, predict(plan, (1, 1)))["counts"]["accuracy"], .5)

    def test_04_final_snapshot_cannot_masquerade_as_early_prediction(self):
        manifest = fixture()
        plan = early_adaptation_plan(manifest, ("e1",), offset=1,
                                     through_phase=39, censor_policy="new_revision")
        record = predict(plan, (2,))[0]
        with self.assertRaises(ValueError):
            score(plan, (replace(record, predicted_at=phase_end(manifest, 39)),))

    def test_05_supersession_censors_early_target_at_exact_boundary(self):
        plan = early_adaptation_plan(fixture(), ("e1",), offset=10,
                                     through_phase=39, censor_policy="new_revision")
        report = score(plan, ())
        self.assertEqual(report["counts"]["censored"], 1)
        self.assertEqual(report["counts"]["eligible"], 0)
        self.assertIsNone(report["counts"]["accuracy"])
        self.assertIsNone(plan.cases[0].target_provenance)

    def test_06_bcb_reversion_does_not_revive_an_old_event(self):
        manifest = fixture(((0, 0, 1), (10, 0, 2), (20, 0, 1)))
        intervals = valid_intervals(manifest, "value_change")
        self.assertEqual(intervals[0].valid_until, Time(10, 1))
        self.assertEqual(intervals[0].superseded_by, "e1")
        self.assertIsNone(intervals[2].valid_until)
        plan = early_adaptation_plan(manifest, ("e0", "e2"), offset=21,
                                     through_phase=39, censor_policy="value_change")
        self.assertEqual([c.status for c in plan.cases], ["censored", "not_yet_due"])
        final = final_truth_plan(manifest, ("e0",), through_phase=39)
        self.assertEqual(final.cases[0].target_provenance.source_id, "e2")
        self.assertEqual(score(final, predict(final, (1,)))["counts"]["accuracy"], 1)

    def test_07_same_value_assignments_have_explicit_two_policy_behavior(self):
        manifest = fixture(((0, 0, 0), (1, 0, 1), (2, 0, 1), (4, 0, 2)), phases=6)
        strict = early_adaptation_plan(manifest, ("e1",), offset=1,
                                       through_phase=5, censor_policy="new_revision")
        semantic = early_adaptation_plan(manifest, ("e1",), offset=1,
                                         through_phase=5, censor_policy="value_change")
        self.assertEqual(strict.cases[0].status, "censored")
        self.assertEqual(semantic.cases[0].status, "eligible")
        self.assertEqual(semantic.cases[0].target_provenance.source_id, "e2")
        self.assertEqual(semantic.cases[0].anchor_validity.valid_until, Time(4, 3))
        self.assertNotEqual(strict.sha256, semantic.sha256)
        self.assertEqual(valid_intervals(manifest, "new_revision")[2].kind, "same_value_assignment")

    def test_08_future_supersession_does_not_censor_present_report(self):
        manifest = fixture(((0, 0, 0), (1, 0, 1), (4, 0, 2)), phases=7)
        before = early_adaptation_plan(manifest, ("e1",), offset=5,
                                       through_phase=1, censor_policy="new_revision")
        after = early_adaptation_plan(manifest, ("e1",), offset=5,
                                      through_phase=4, censor_policy="new_revision")
        self.assertEqual(before.cases[0].status, "not_yet_due")
        self.assertEqual(after.cases[0].status, "censored")

    def test_09_same_phase_supersession_obeys_event_order(self):
        manifest = fixture(((0, 0, 0), (1, 0, 1), (1, 0, 2)), phases=3)
        plan = early_adaptation_plan(manifest, ("e1", "e2"), offset=0,
                                     through_phase=1, censor_policy="new_revision")
        self.assertEqual([c.status for c in plan.cases], ["censored", "eligible"])
        self.assertEqual(score(plan, predict(plan, (2,)))["counts"]["accuracy"], 1)

    def test_10_missing_eligible_predictions_count_as_wrong(self):
        plan = early_adaptation_plan(fixture(), ("e1", "e2"), offset=1,
                                     through_phase=39, censor_policy="new_revision")
        one = predict(plan, (1, 2))[:1]
        counts = score(plan, one)["counts"]
        self.assertEqual((counts["eligible"], counts["correct"], counts["missing"]), (2, 1, 1))
        self.assertEqual(counts["accuracy"], .5)
        self.assertEqual(score(plan, ())["counts"]["accuracy"], 0)

    def test_11_abstention_is_measured_wrong_not_missing(self):
        plan = final_truth_plan(fixture(), ("e0",), through_phase=39)
        counts = score(plan, predict(plan, (None,)))["counts"]
        self.assertEqual((counts["measured"], counts["abstained"], counts["missing"], counts["wrong"]),
                         (1, 1, 0, 1))

    def test_12_duplicate_unknown_and_foreign_predictions_rejected(self):
        plan = final_truth_plan(fixture(), ("e0",), through_phase=39)
        item = predict(plan, (2,))[0]
        invalid = [(item, item), (replace(item, case_id="unknown"),),
                   (replace(item, plan_sha256="0"*64),)]
        for records in invalid:
            with self.subTest(records=records), self.assertRaises(ValueError):
                score(plan, records)

    def test_13_invalid_timestamp_and_wrong_valid_timestamp_rejected(self):
        manifest = fixture()
        plan = final_truth_plan(manifest, ("e0",), through_phase=39)
        item = predict(plan, (2,))[0]
        for phase, order in ((True, 1), (1.0, 2), (-1, 0), (0, False), (0, -1)):
            with self.subTest(phase=phase, order=order), self.assertRaises(ValueError):
                Time(phase, order)
        for at in (Time(40, 3), Time(39, 99), Time(39, 0), Time(20, 2), phase_end(manifest, 38)):
            with self.subTest(at=at), self.assertRaises(ValueError):
                score(plan, (replace(item, predicted_at=at),))

    def test_14_ineligible_receipts_rejected(self):
        plan = early_adaptation_plan(fixture(), ("e1", "e2"), offset=10,
                                     through_phase=20, censor_policy="new_revision")
        for case in plan.cases:
            with self.subTest(case=case.status), self.assertRaises(ValueError):
                score(plan, (Prediction(case.case_id, case.requested_time, 1, plan.sha256),))

    def test_15_cohort_identity_and_policy_validation(self):
        manifest = fixture()
        for identities in (("e0", "e0"), ("ghost",), ["e0"]):
            with self.subTest(identities=identities), self.assertRaises(ValueError):
                final_truth_plan(manifest, identities, through_phase=39)
        with self.assertRaises(ValueError):
            final_truth_plan(manifest, ("e0", "e1"), through_phase=39)
        with self.assertRaises(ValueError):
            final_truth_plan(manifest, ("e2",), through_phase=10)
        for offset, policy in ((None, "new_revision"), (True, "new_revision"), (-1, "new_revision"), (1, "unspecified")):
            with self.subTest(offset=offset, policy=policy), self.assertRaises(ValueError):
                early_adaptation_plan(manifest, ("e0",), offset=offset,
                                      through_phase=39, censor_policy=policy)
        with self.assertRaises(ValueError):
            Plan(manifest, "invalid", ("e0",), 39, "latest_truth", None)

    def test_16_predictions_are_domain_checked_and_immutable(self):
        plan = final_truth_plan(fixture(), ("e0",), through_phase=39)
        item = predict(plan, (2,))[0]
        for value in (True, 1.5, -1, "2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(item, value=value)
        with self.assertRaises(ValueError):
            score(plan, (replace(item, value=5),))
        with self.assertRaises(ValueError):
            score(plan, [item])
        with self.assertRaises(FrozenInstanceError):
            item.value = 0
        with self.assertRaises(FrozenInstanceError):
            plan.through_phase = 0

    def test_17_empty_cohort_has_no_fabricated_rate(self):
        plan = final_truth_plan(fixture(), (), through_phase=39)
        counts = score(plan, ())["counts"]
        self.assertEqual(counts["total"], 0)
        self.assertIsNone(counts["accuracy"])

    def test_18_distinct_final_cohort_keys_preserve_denominator(self):
        manifest = fixture(((0, 0, 0), (0, 1, 3), (1, 0, 2)), phases=3)
        plan = final_truth_plan(manifest, ("e0", "e1"), through_phase=2)
        records = predict(plan, (2, 3))
        counts = score(plan, records[:1])["counts"]
        self.assertEqual(counts["accuracy"], .5)
        self.assertEqual(counts["eligible"], 2)

    def test_19_exhaustive_small_histories_against_independent_scan(self):
        # 27 histories, 2 policies, 5 cutoffs, 5 offsets = 1350 plans.
        for values in itertools.product(range(3), repeat=3):
            manifest = fixture(tuple(zip((0, 1, 3), (0, 0, 0), values)), phases=5)
            for policy, through, offset in itertools.product(("new_revision", "value_change"), range(5), range(5)):
                cohort = tuple(e for e in manifest.events if e.phase <= through)
                plan = early_adaptation_plan(manifest, tuple(e.event_id for e in cohort),
                                             offset=offset, through_phase=through, censor_policy=policy)
                expected = {"eligible": 0, "censored": 0, "not_yet_due": 0}
                outputs = []
                for event, case in zip(cohort, plan.cases):
                    due = event.phase + offset
                    superseding = [other for other in manifest.events if other.order > event.order
                                   and other.key == event.key
                                   and (policy == "new_revision" or other.value != event.value)]
                    if superseding and superseding[0].phase <= min(due, through):
                        status = "censored"
                    elif due > through:
                        status = "not_yet_due"
                    else:
                        status = "eligible"
                    self.assertEqual(case.status, status)
                    expected[status] += 1
                    if status == "eligible":
                        active = [other for other in manifest.events if other.key == event.key and other.phase <= due][-1]
                        self.assertEqual(case.target_provenance.source_id, active.event_id)
                        self.assertEqual(case.target_provenance.target, active.value)
                        outputs.append(Prediction(case.case_id, case.requested_time, active.value, plan.sha256))
                actual = score(plan, tuple(outputs))["counts"]
                for k,v in expected.items():
                    self.assertEqual(actual[k], v)
                self.assertEqual(actual["correct"], actual["eligible"])

    def test_20_outputs_exclusive_and_default_cli_read_only(self):
        script = Path(__file__).with_name("temporal_revision_scoring_v6.py")
        watched = [script, Path(__file__), script.with_name("correction_protocol.py")]
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            run = subprocess.run([sys.executable, "-B", str(script)],
                                 cwd=directory, capture_output=True, text=True, timeout=20)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(list(directory.iterdir()), [])
            self.assertEqual(json.loads(run.stdout)["contract_sha256"], contract_sha256())
            path = directory / "frozen.json"
            path.write_bytes(b"preserve")
            run = subprocess.run([sys.executable, "-B", str(script), "--out", str(path)],
                                 cwd=directory, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(run.returncode, 0)
            self.assertEqual(path.read_bytes(), b"preserve")
        self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in watched})


def example_receipts():
    manifest = fixture()
    early = early_adaptation_plan(manifest, ("e1", "e2"), offset=1,
                                  through_phase=39, censor_policy="new_revision")
    final = final_truth_plan(manifest, ("e1",), through_phase=39)
    return {
        "abc_perfect_at_each_requested_time": score(early, predict(early, (1, 2))),
        "abc_stale_B_at_each_early_time": score(early, predict(early, (1, 1))),
        "abc_perfect_final_C": score(final, predict(final, (2,))),
        "abc_stale_final_B": score(final, predict(final, (1,))),
        "abc_missing_eligible_records": score(early, ()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="Optional NEW result JSON; no overwrite")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TemporalRevisionTests)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    here = Path(__file__).resolve()
    sources = {"implementation": here.with_name("temporal_revision_scoring_v6.py"),
               "adversarial_checks": here, "event_manifest_dependency": here.with_name("correction_protocol.py")}
    report = {
        "suite": "Nano LM V6 temporal scoring repair", "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun,
        "passed": result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        "failures": len(result.failures), "errors": len(result.errors),
        "contract_sha256": contract_sha256(),
        "source_sha256": {k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in sources.items()},
        "independent_exhaustive_plans": 1350,
        "scope": "Synthetic temporal-scoring checks only; no training, no learned-model evidence, no G04/G06 raw-result rewrite.",
        "examples": example_receipts(),
    }
    write_report(report, args.out)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())

