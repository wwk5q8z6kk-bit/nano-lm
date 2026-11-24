"""Behavioral checks for the G03b measurement instrument, not model benchmarks."""
from __future__ import annotations

import argparse
from dataclasses import FrozenInstanceError, asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.dont_write_bytecode = True
from correction_protocol import (
    Event, EvaluationLog, LinearConstraints, Manifest, Measurement, TargetProvenance,
    WorldSpec, assess_linear_candidate, check_logit_margin, classify_solver_status,
    deliveries, generate_manifest, new_log, phase_end_queries, record_after_write,
    record_phase_end, replay_rng, summarize, write_report,
)


def fixture(rows, phases=8):
    """Rows are phase/entity/relation/value; every repeated key is a new revision."""
    latest, events = {}, []
    for phase, entity, relation, value in rows:
        prior = latest.get((entity, relation))
        event = Event(f"e{len(events)}", phase, len(events), entity, relation, value,
                      None if prior is None else prior.event_id,
                      None if prior is None else prior.value)
        latest[event.key] = event
        events.append(event)
    spec = WorldSpec(phases=phases, entities_per_phase=2, relations=3,
                     shown_relations=2, values=5, assignments_per_phase=2)
    return Manifest(7, spec, tuple(events))


def rows(report, horizon=0, cohort="all_assignments"):
    return report["cohorts"][cohort]["phase_end_horizons"][str(horizon)]


def constraints(matrix=((-1.0,),), upper=(-1.0,), source="new_assignment"):
    metadata = tuple(TargetProvenance(0, i, 1, source, f"source-{i}") for i in range(len(upper)))
    return LinearConstraints.from_arrays(matrix, upper, metadata)


class CorrectionProtocolTests(unittest.TestCase):
    def test_canonical_manifest_roundtrip_and_reproducibility(self):
        manifest = generate_manifest(8001)
        self.assertEqual(manifest.canonical_bytes(), generate_manifest(8001).canonical_bytes())
        self.assertNotEqual(manifest.sha256, generate_manifest(8002).sha256)
        self.assertEqual(len(manifest.events), 34)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest.write_new(path)
            restored = Manifest.read(path)
            self.assertEqual(restored, manifest)
            self.assertEqual(restored.sha256, manifest.sha256)

    def test_manifest_and_visible_values_are_immutable(self):
        manifest = generate_manifest()
        with self.assertRaises(FrozenInstanceError):
            manifest.events[0].value = 3
        with self.assertRaises(TypeError):
            manifest.events[0] = manifest.events[-1]
        visible = next(deliveries(manifest))
        self.assertIsInstance(visible.current_targets, tuple)
        with self.assertRaises(TypeError):
            visible.current_targets[0][2] = 4

    def test_original_shared_generator_failure_is_reproduced(self):
        def legacy(consume_replay):
            rng = np.random.default_rng(8001)
            stream = []
            for phase in range(4):
                stream.append(tuple(rng.integers(0, 1000, size=8).tolist()))
                if consume_replay:
                    rng.choice(100, 13, replace=False)
            return stream
        control, replay = legacy(False), legacy(True)
        self.assertEqual(control[0], replay[0])
        self.assertNotEqual(control[1:], replay[1:])

    def test_named_replay_streams_cannot_change_frozen_world(self):
        manifest = generate_manifest(8001)
        before = manifest.canonical_bytes()
        random = replay_rng(8001, "random")
        for amount in (1, 100, 10_000):
            random.integers(0, 100, size=amount)
        self.assertEqual(manifest.canonical_bytes(), before)
        self.assertEqual(generate_manifest(8001).sha256, manifest.sha256)
        a = replay_rng(8001, "random").integers(0, 2**31, size=10)
        b = replay_rng(8001, "random").integers(0, 2**31, size=10)
        c = replay_rng(8001, "targeted").integers(0, 2**31, size=10)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_learner_view_contains_only_delivered_targets(self):
        manifest = fixture([(0, 0, 0, 0), (1, 0, 0, 1), (2, 0, 0, 4)])
        views = list(deliveries(manifest))
        self.assertEqual([e.event_id for e in views[1].observed_events], ["e0", "e1"])
        self.assertEqual(views[1].current_targets, ((0, 0, 1, "e1"),))
        self.assertEqual(set(asdict(views[1])), {"current_event", "observed_events", "current_targets"})
        self.assertNotIn(4, [e.value for e in views[1].observed_events])
        self.assertEqual(views[2].current_targets, ((0, 0, 4, "e2"),))

    def test_invalid_revision_order_and_domain_are_rejected(self):
        base = fixture([(0, 0, 0, 0), (1, 0, 0, 1), (2, 0, 0, 2)])
        mutations = [
            (base.events[0], replace(base.events[1], previous_revision="missing"), base.events[2]),
            (base.events[0], replace(base.events[1], previous_value=4), base.events[2]),
            (base.events[0], replace(base.events[1], order=2), base.events[2]),
            (base.events[0], replace(base.events[1], value=999), base.events[2]),
            (base.events[0], replace(base.events[1], event_id="e0"), base.events[2]),
            (base.events[0], base.events[1], replace(base.events[2], phase=0)),
        ]
        for events in mutations:
            with self.subTest(events=events), self.assertRaises(ValueError):
                Manifest(base.seed, base.spec, events)
        with self.assertRaises(ValueError):
            WorldSpec(phases=0)
        with self.assertRaises(ValueError):
            WorldSpec(phases=10000, entities_per_phase=100)
        with self.assertRaises(ValueError):
            Event("bad", True, 0, 0, 0, 0)

    def test_same_subject_writes_separate_immediate_and_phase_end(self):
        # Two different keys on one subject; a later update damages the first.
        manifest = fixture([(0, 0, 0, 0), (0, 0, 1, 0),
                            (1, 0, 0, 1), (1, 0, 1, 2)])
        log = new_log(manifest)
        log = record_after_write(manifest, log, "e2", 1)
        log = record_after_write(manifest, log, "e3", 2)
        log = record_phase_end(manifest, log, 1, {(0, 0): 0, (0, 1): 2})
        report = summarize(manifest, log)
        self.assertEqual(report["cohorts"]["all_assignments"]["immediately_after_write"]["accuracy"], 1)
        self.assertEqual(rows(report)["accuracy"], .5)
        self.assertEqual(rows(report)["censored"], 0)

    def test_same_key_reassignment_within_phase_censors_old_version(self):
        manifest = fixture([(0, 0, 0, 0), (1, 0, 0, 1), (1, 0, 0, 2)])
        log = record_after_write(manifest, new_log(manifest), "e1", 1)
        log = record_after_write(manifest, log, "e2", 2)
        log = record_phase_end(manifest, log, 1, {(0, 0): 2})
        report = summarize(manifest, log)
        self.assertEqual(rows(report)["total_events"], 2)
        self.assertEqual(rows(report)["censored"], 1)
        self.assertEqual(rows(report)["eligible"], 1)
        self.assertEqual(rows(report)["accuracy"], 1)
        self.assertEqual(report["cohorts"]["all_assignments"]["immediately_after_write"]["measured"], 2)

    def test_same_value_assignments_are_retained_and_stratified(self):
        manifest = fixture([(0, 0, 0, 0), (1, 0, 0, 1), (2, 0, 0, 1)])
        self.assertEqual(manifest.events[-1].kind, "same_value_assignment")
        log = record_after_write(manifest, new_log(manifest), "e1", 1)
        log = record_phase_end(manifest, log, 1, {(0, 0): 1})
        log = record_after_write(manifest, log, "e2", 1)
        log = record_phase_end(manifest, log, 2, {(0, 0): 1})
        report = summarize(manifest, log)
        self.assertEqual(rows(report)["total_events"], 2)
        self.assertEqual(rows(report, cohort="semantic_value_changes")["total_events"], 1)
        self.assertEqual(rows(report, cohort="same_value_assignments")["total_events"], 1)
        self.assertEqual(rows(report, 1)["censored"], 1)
        self.assertEqual(rows(report, 1)["not_yet_due"], 1)

    def test_value_change_policy_keeps_trajectory_through_same_value_revision(self):
        manifest = fixture([(0, 0, 0, 0), (1, 0, 0, 1), (2, 0, 0, 1), (4, 0, 0, 2)])
        log = record_after_write(manifest, new_log(manifest, "value_change"), "e1", 1)
        log = record_phase_end(manifest, log, 1, {(0, 0): 1})
        log = record_after_write(manifest, log, "e2", 1)
        log = record_phase_end(manifest, log, 2, {(0, 0): 1})
        report = summarize(manifest, log)
        self.assertEqual(rows(report, 1)["eligible"], 1)
        self.assertEqual(rows(report, 1)["censored"], 0)
        self.assertEqual(rows(report, 1)["accuracy"], 1)
        # Later different-value events are not used before they are observed.
        self.assertEqual(rows(report, 20)["censored"], 0)
        log = record_after_write(manifest, log, "e3", 2)
        log = record_phase_end(manifest, log, 4, {(0, 0): 2})
        self.assertEqual(rows(summarize(manifest, log), 20)["censored"], 2)

    def test_failed_acquisition_stays_in_primary_denominator(self):
        manifest = fixture([(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 1), (1, 1, 0, 1)])
        log = record_after_write(manifest, new_log(manifest), "e2", 0)
        log = record_after_write(manifest, log, "e3", 1)
        log = record_phase_end(manifest, log, 1, {(0, 0): 0, (1, 0): 1})
        row = rows(summarize(manifest, log))
        self.assertEqual(row["accuracy"], .5)
        self.assertEqual(row["eligible"], 2)
        self.assertEqual(row["failed_initial_acquisition_eligible"], 1)
        self.assertEqual(row["acquired_only_accuracy"], 1)
        log = record_phase_end(manifest, log, 2, {(0, 0): 1, (1, 0): 1})
        delayed = rows(summarize(manifest, log), 1)
        self.assertEqual(delayed["accuracy"], 1)
        self.assertEqual(delayed["acquired_only_eligible"], 1)

    def test_unknown_initial_acquisition_does_not_create_selected_retention(self):
        manifest = fixture([(0, 0, 0, 0), (1, 0, 0, 1)])
        log = record_phase_end(manifest, new_log(manifest), 1, {(0, 0): 1})
        row = rows(summarize(manifest, log))
        self.assertEqual(row["accuracy"], 1)
        self.assertIsNone(row["acquired_only_accuracy"])
        self.assertEqual(row["initial_acquisition_unknown_eligible"], 1)

    def test_missing_due_measurement_makes_primary_rate_undefined(self):
        manifest = fixture([(0, 0, 0, 0), (1, 0, 0, 1)])
        log = record_after_write(manifest, new_log(manifest), "e1", 1)
        log = record_phase_end(manifest, log, 2, {(0, 0): 1})  # Phase1 snapshot missing.
        report = summarize(manifest, log)
        self.assertEqual(rows(report)["eligible"], 1)
        self.assertEqual(rows(report)["missing_measurement"], 1)
        self.assertIsNone(rows(report)["accuracy"])
        self.assertEqual(rows(report, 1)["accuracy"], 1)

    def test_not_yet_due_and_known_censoring_are_explicit(self):
        manifest = fixture([(0, 0, 0, 0), (1, 0, 0, 1), (2, 0, 0, 2)])
        log = record_phase_end(manifest, new_log(manifest), 2, {(0, 0): 2})
        row = rows(summarize(manifest, log), 20)
        self.assertEqual(row["total_events"], 2)
        self.assertEqual(row["censored"], 1)
        self.assertEqual(row["not_yet_due"], 1)
        self.assertEqual(row["eligible"], 0)
        self.assertIsNone(row["accuracy"])
        with self.assertRaises(ValueError):
            summarize(manifest, log, through_phase=3)

    def test_future_supersession_does_not_change_present_cohort(self):
        manifest = fixture([(0, 0, 0, 0), (1, 0, 0, 1), (5, 0, 0, 2)])
        log = record_phase_end(manifest, new_log(manifest), 1, {(0, 0): 1})
        report = summarize(manifest, log)
        self.assertEqual(rows(report)["eligible"], 1)
        self.assertEqual(rows(report, 5)["not_yet_due"], 1)
        self.assertEqual(rows(report, 5)["censored"], 0)

    def test_missing_or_future_query_keys_are_rejected(self):
        manifest = fixture([(0, 0, 0, 0), (1, 0, 0, 1), (3, 1, 0, 0), (4, 1, 0, 1)])
        log = new_log(manifest)
        self.assertEqual(phase_end_queries(manifest, 1), ((0, 0),))
        for predictions in ({}, {(0, 0): 1, (1, 0): 1}, {(0.0, 0): 1}, {(False, 0): 1}):
            with self.assertRaises(ValueError):
                record_phase_end(manifest, log, 1, predictions)
        log = record_phase_end(manifest, log, 1, {(0, 0): None})
        row = rows(summarize(manifest, log))
        self.assertEqual(row["measured"], 1)
        self.assertEqual(row["correct"], 0)
        self.assertEqual(row["accuracy"], 0)

    def test_append_records_preserve_old_log_and_reject_retroactive_samples(self):
        manifest = fixture([(0, 0, 0, 0), (1, 0, 0, 1)])
        original = new_log(manifest)
        original_bytes = json.dumps(asdict(original), sort_keys=True)
        appended = record_after_write(manifest, original, "e1", 1)
        self.assertEqual(json.dumps(asdict(original), sort_keys=True), original_bytes)
        self.assertEqual(appended.measurements[:len(original.measurements)], original.measurements)
        with self.assertRaises(ValueError):
            record_after_write(manifest, appended, "e1", 1)
        finished = record_phase_end(manifest, appended, 1, {(0, 0): 1})
        with self.assertRaises(ValueError):
            record_after_write(manifest, finished, "e1", 1)
        with self.assertRaises(ValueError):
            record_phase_end(manifest, finished, 1, {(0, 0): 1})

    def test_foreign_manifest_and_fabricated_record_identity_are_rejected(self):
        manifest = fixture([(0, 0, 0, 0), (1, 0, 0, 1)])
        with self.assertRaises(ValueError):
            record_phase_end(generate_manifest(9), new_log(manifest), 1, {})
        bad = replace(new_log(manifest),
                      measurements=(Measurement("ghost", "after_write", 1, 1),),
                      last_position=(1, 1))
        with self.assertRaises(ValueError):
            summarize(manifest, bad)
        bad = replace(new_log(manifest),
                      measurements=(Measurement("e1", "after_write", 2, 1),),
                      last_position=(2, 1))
        with self.assertRaises(ValueError):
            summarize(manifest, bad)

    def test_solver_statuses_do_not_conflate_reported_infeasibility_and_numerics(self):
        self.assertEqual(classify_solver_status(0), "candidate")
        self.assertEqual(classify_solver_status(2), "reported_infeasible")
        self.assertEqual(classify_solver_status(4), "unknown_failure")
        for status in (1, 3, 4, 99):
            self.assertEqual(classify_solver_status(status), "unknown_failure")
        report2 = assess_linear_candidate(2, constraints())
        report4 = assess_linear_candidate(4, constraints())
        self.assertNotEqual(report2["verdict"], report4["verdict"])
        self.assertFalse(report2["validated"])
        self.assertFalse(report4["validated"])

    def test_solver_success_requires_candidate_and_actual_deployed_residuals(self):
        c = constraints()
        self.assertEqual(assess_linear_candidate(0, c, np.array([0.]))["verdict"], "invalid_candidate")
        only = assess_linear_candidate(0, c, np.array([1.]))
        self.assertEqual(only["verdict"], "candidate_only")
        self.assertFalse(only["validated"])
        good = assess_linear_candidate(0, c, np.array([1.]), np.array([1.], dtype=np.float32))
        self.assertTrue(good["validated"])
        self.assertEqual(good["deployed_value"]["minimum_slack"], 0)
        for bad in (np.array([np.nan]), np.array([np.inf]), np.array([1], dtype=np.int64)):
            self.assertFalse(assess_linear_candidate(0, c, np.array([1.]), bad)["validated"])

    def test_float32_cast_is_checked_against_original_threshold(self):
        threshold = 1.0 + 2.0 ** -25
        c = constraints(upper=(-threshold,))
        candidate = np.array([threshold], dtype=np.float64)
        deployed = candidate.astype(np.float32)
        report = assess_linear_candidate(0, c, candidate, deployed)
        self.assertTrue(report["solver_candidate"]["satisfies"])
        self.assertFalse(report["validated"])
        self.assertEqual(report["verdict"], "invalid_deployed_value")
        self.assertGreater(report["deployed_value"]["maximum_violation"], 0)
        # A caller may choose tolerance, but its effect is explicit in the receipt.
        tolerant = assess_linear_candidate(0, c, candidate, deployed, atol=1e-7)
        self.assertTrue(tolerant["validated"])
        self.assertEqual(tolerant["atol"], 1e-7)

    def test_applied_increment_includes_addition_rounding(self):
        original = np.array([1e8], dtype=np.float32)
        requested_delta = np.array([1.], dtype=np.float64)
        actual_state = original + requested_delta.astype(np.float32)
        actual_delta = actual_state.astype(np.float64) - original.astype(np.float64)
        self.assertEqual(float(actual_delta[0]), 0.0)
        result = assess_linear_candidate(0, constraints(), requested_delta, actual_delta)
        self.assertFalse(result["validated"])
        self.assertEqual(result["deployed_value"]["maximum_violation"], 1.0)

    def test_constraint_provenance_is_explicit_and_immutable(self):
        provenance = (
            TargetProvenance(0, 0, 1, "new_assignment", "e1"),
            TargetProvenance(0, 1, 2, "observed_target", "e0"),
            TargetProvenance(0, 2, 3, "model_prediction", "checkpoint-abc"),
        )
        a = np.array([[-1.], [-1.], [-1.]])
        c = LinearConstraints.from_arrays(a, [-1., -1., -1.], provenance)
        a[:] = 99  # Original caller array cannot rewrite the constraints.
        self.assertEqual(c.matrix[0], (-1.,))
        receipt = assess_linear_candidate(0, c, np.array([1.]), np.array([1.]))
        self.assertEqual({x["source"] for x in receipt["target_provenance"]},
                         {"new_assignment", "observed_target", "model_prediction"})
        self.assertEqual(receipt["constraints_sha256"], c.sha256)
        self.assertEqual(len(receipt["deployed_value"]["value_sha256"]), 64)
        with self.assertRaises(ValueError):
            TargetProvenance(0, 0, 1, "unspecified", "x")
        with self.assertRaises(ValueError):
            LinearConstraints.from_arrays([[-1.]], [-1.], ())

    def test_actual_logits_distinguish_argmax_from_required_margin(self):
        result = check_logit_margin(np.array([1.0000001, 1.0], dtype=np.float32), 0, 2e-7)
        self.assertTrue(result["argmax_correct"])
        self.assertFalse(result["satisfies"])
        tied = check_logit_margin(np.array([0., 0.]), 1, 0)
        self.assertTrue(tied["satisfies"])
        self.assertFalse(tied["argmax_correct"])
        for values in (np.array([np.nan, 1.]), np.array([1, 0]),
                       np.array([1e308, -1e308])):
            with self.assertRaises(ValueError):
                check_logit_margin(values, 0, 1)

    def test_invalid_tolerances_and_duplicate_manifest_json_rejected(self):
        for tolerance in (True, -1, float("nan"), "0", 1j):
            with self.assertRaises(ValueError):
                assess_linear_candidate(0, constraints(), np.array([1.]), np.array([1.]), atol=tolerance)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for content in ('{"seed":1,"seed":2}', '[]', 'null'):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    Manifest.read(path)

    def test_existing_outputs_are_preserved_and_default_cli_is_read_only(self):
        script = Path(__file__).with_name("correction_protocol.py")
        def inventory():
            return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in script.parent.iterdir() if p.is_file()}
        before = inventory()
        run = subprocess.run([sys.executable, "-B", str(script)],
                             capture_output=True, text=True, timeout=20)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("no model training", json.loads(run.stdout)["scope"])
        self.assertEqual(before, inventory())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            path.write_bytes(b"frozen bytes")
            with self.assertRaises(FileExistsError):
                generate_manifest().write_new(path)
            with self.assertRaises(FileExistsError):
                write_report({"test": True}, path)
            self.assertEqual(path.read_bytes(), b"frozen bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="Optional NEW result JSON; no overwrite")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CorrectionProtocolTests)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    report = {"suite": "G03b correction measurement contracts",
              "tests_run": result.testsRun,
              "passed": result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
              "failures": len(result.failures), "errors": len(result.errors),
              "scope": "Measurement, chronology, provenance and numerical-validation fixtures only; no model training or architecture comparison."}
    write_report(report, args.out)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
