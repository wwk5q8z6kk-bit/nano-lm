"""Integration assertions concern execution and accounting, never high accuracy."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.dont_write_bytecode = True
from native_correction_smoke import (
    AdamW, HORIZONS, Manifest, NativeCorrectionLearner, STEPS_PER_EVENT,
    TrainableNative, fixed_manifest, query_tokens, run_smoke,
)


class NativeCorrectionSmokeTests(unittest.TestCase):
    def test_fixture_contains_initial_change_same_value_and_within_phase_revision(self):
        manifest = fixed_manifest()
        self.assertEqual(len(manifest.events), 8)
        self.assertEqual([e.kind for e in manifest.events].count("initial_observation"), 3)
        self.assertEqual([e.kind for e in manifest.events].count("value_change"), 4)
        self.assertEqual([e.kind for e in manifest.events].count("same_value_assignment"), 1)
        first, superseding = manifest.events[3], manifest.events[5]
        self.assertEqual(first.key, superseding.key)
        self.assertEqual(first.phase, superseding.phase)
        self.assertEqual(superseding.previous_revision, first.event_id)

    def test_actual_weights_and_optimizer_state_change_with_exact_exposures(self):
        report = run_smoke()
        self.assertNotEqual(report["initial_checkpoint"]["model_sha256"],
                            report["final_checkpoint"]["model_sha256"])
        self.assertNotEqual(report["initial_checkpoint"]["optimizer_state_sha256"],
                            report["final_checkpoint"]["optimizer_state_sha256"])
        counts = report["counts"]
        self.assertEqual(counts["optimizer_steps"], 16)
        self.assertEqual(counts["scored_target_token_exposures"], 16)
        self.assertEqual(counts["training_input_tokens"], 48)
        self.assertEqual(counts["masked_out_target_positions"], 32)
        self.assertEqual(counts["assignment_events"], 5)
        updates = [s for e in report["events"] for s in e["optimizer_updates"]]
        self.assertEqual([s["step"] for s in updates], list(range(1, 17)))
        self.assertTrue(all(s["actual_displacement_l2"] > 0 for s in updates))
        for row in report["events"]:
            self.assertEqual(row["target_token_exposures"], STEPS_PER_EVENT)
            self.assertEqual(row["checkpoint_after"]["optimizer_step"]
                             - row["checkpoint_before"]["optimizer_step"], STEPS_PER_EVENT)

    def test_same_runtime_execution_is_deterministic_and_manifest_unchanged(self):
        manifest = fixed_manifest()
        original = manifest.canonical_bytes()
        self.assertEqual(run_smoke(manifest), run_smoke(manifest))
        self.assertEqual(original, manifest.canonical_bytes())

    def test_deliveries_and_future_mutation_cannot_change_earlier_states(self):
        observed = []
        original_observe = NativeCorrectionLearner.observe

        def spy(learner, visible):
            observed.append(asdict(visible))
            self.assertFalse(hasattr(learner, "manifest"))
            return original_observe(learner, visible)

        manifest = fixed_manifest()
        with patch.object(NativeCorrectionLearner, "observe", spy):
            original = run_smoke(manifest)
        for index, view in enumerate(observed):
            self.assertEqual(set(view), {"current_event", "observed_events", "current_targets"})
            self.assertEqual([e["event_id"] for e in view["observed_events"]],
                             [e.event_id for e in manifest.events[:index + 1]])
            self.assertTrue(all(e["order"] <= index for e in view["observed_events"]))
        events = manifest.events[:-1] + (replace(manifest.events[-1], value=1),)
        changed = run_smoke(Manifest(manifest.seed, manifest.spec, events))
        self.assertNotEqual(original["manifest_sha256"], changed["manifest_sha256"])
        self.assertEqual(original["initial_checkpoint"], changed["initial_checkpoint"])
        self.assertEqual(original["events"][:-1], changed["events"][:-1])
        self.assertEqual(original["phase_ends"][:4], changed["phase_ends"][:4])

    def test_predictions_are_from_native_forward_and_use_the_full_head(self):
        calls = []
        forward = TrainableNative.forward

        def spy(model, tokens):
            logits = forward(model, tokens)
            calls.append((tokens.copy(), logits[0, -1].copy()))
            return logits

        with patch.object(TrainableNative, "forward", spy):
            report = run_smoke()
        predictions = []
        for phase in report["phase_ends"]:
            predictions.extend(e["prediction_after_write"] for e in report["events"]
                               if e["phase"] == phase["phase"])
            predictions.extend(phase["predictions"])
        self.assertEqual(len(calls), report["counts"]["inference_forward_calls"])
        self.assertEqual(len(calls), len(predictions))
        for (tokens, logits), receipt in zip(calls, predictions):
            np.testing.assert_array_equal(tokens[0], receipt["query_tokens"])
            np.testing.assert_array_equal(logits, receipt["logits"])
            token = int(np.argmax(logits))
            self.assertEqual(token, receipt["raw_argmax_token"])
            self.assertEqual(receipt["prediction"], token - 5 if token in (5, 6, 7) else None)
            self.assertTrue(np.isfinite(logits).all())

    def test_training_calls_score_only_the_delivered_current_target(self):
        calls = []
        train_step = AdamW.train_step

        def spy(optimizer, documents, valid_mask=None):
            calls.append((documents.copy(), valid_mask.copy()))
            return train_step(optimizer, documents, valid_mask)

        with patch.object(AdamW, "train_step", spy):
            report = run_smoke()
        self.assertEqual(len(calls), report["counts"]["optimizer_steps"])
        for index, (documents, mask) in enumerate(calls):
            event = fixed_manifest().events[index // STEPS_PER_EVENT]
            self.assertEqual(documents.dtype.kind, "i")
            self.assertEqual(documents.tolist(), [list(query_tokens(event.key)) + [5 + event.value]])
            self.assertEqual(mask.tolist(), [[False, False, True]])
            self.assertEqual(int(mask.sum()), 1)
            self.assertFalse(any(token in (5, 6, 7) for token in documents[0, :-1]))

    def test_every_horizon_has_complete_denominators_and_censor_timing(self):
        report = run_smoke()
        for cohort in report["metrics"]["cohorts"].values():
            immediate = cohort["immediately_after_write"]
            self.assertEqual(immediate["events"], immediate["measured"] + immediate["missing"])
            self.assertEqual(immediate["missing"], 0)
            for horizon in HORIZONS:
                row = cohort["phase_end_horizons"][str(horizon)]
                self.assertEqual(row["total_events"], row["eligible"] + row["censored"] + row["not_yet_due"])
                self.assertEqual(row["eligible"], row["measured"] + row["missing_measurement"])
                self.assertEqual(row["missing_measurement"], 0)
                self.assertEqual(row["eligible"], row["acquired_only_eligible"]
                                 + row["failed_initial_acquisition_eligible"]
                                 + row["initial_acquisition_unknown_eligible"])
        cohort = report["metrics"]["cohorts"]["all_assignments"]
        self.assertEqual(cohort["immediately_after_write"]["events"], 5)
        expected = {0: (4, 1, 0), 1: (3, 2, 0), 5: (1, 2, 2),
                    10: (0, 2, 3), 20: (0, 2, 3)}
        for horizon, partition in expected.items():
            row = cohort["phase_end_horizons"][str(horizon)]
            self.assertEqual((row["eligible"], row["censored"], row["not_yet_due"]), partition)

    def test_prediction_receipts_identify_the_causal_checkpoint(self):
        report = run_smoke()
        previous = report["initial_checkpoint"]
        for row in report["events"]:
            self.assertEqual(row["checkpoint_before"], previous)
            self.assertEqual(row["prediction_after_write"]["checkpoint"], row["checkpoint_after"])
            self.assertEqual(row["checkpoint_after"]["last_delivered_event_id"], row["event_id"])
            self.assertEqual(row["delivered_event_ids"][-1], row["event_id"])
            previous = row["checkpoint_after"]
        self.assertEqual(previous, report["final_checkpoint"])
        for phase in report["phase_ends"]:
            preceding = [e for e in report["events"] if e["phase"] <= phase["phase"]]
            expected = preceding[-1]["checkpoint_after"] if preceding else report["initial_checkpoint"]
            self.assertEqual(phase["checkpoint"], expected)
            self.assertTrue(all(p["checkpoint"] == expected for p in phase["predictions"]))

    def test_cli_defaults_to_stdout_and_refuses_to_overwrite(self):
        script = Path(__file__).with_name("native_correction_smoke.py").resolve()
        with tempfile.TemporaryDirectory() as directory:
            cmd = [sys.executable, "-B", str(script)]
            first = subprocess.run(cmd, cwd=directory, capture_output=True, text=True, check=True)
            parsed = json.loads(first.stdout)
            self.assertEqual(parsed["counts"]["optimizer_steps"], 16)
            self.assertEqual(list(Path(directory).iterdir()), [])
            output = Path(directory) / "new-receipt.json"
            subprocess.run(cmd + ["--out", str(output)], cwd=directory,
                           capture_output=True, text=True, check=True)
            saved = output.read_bytes()
            second = subprocess.run(cmd + ["--out", str(output)], cwd=directory,
                                    capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("FileExistsError", second.stderr)
            self.assertEqual(output.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main(verbosity=2)
