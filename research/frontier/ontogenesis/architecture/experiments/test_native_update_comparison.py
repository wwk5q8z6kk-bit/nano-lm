"""Tests exercise causal native execution and accounting, never score thresholds."""
from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

sys.dont_write_bytecode = True
import native_update_comparison as comparison
from correction_protocol import Event, Manifest, VisiblePrefix, deliveries, generate_manifest


class NativeUpdateComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training_calls, cls.forward_calls = [], []
        real_train = comparison.AdamW.train_step
        real_forward = comparison.TrainableNative.forward

        def train_spy(optimizer, documents, valid_mask=None):
            cls.training_calls.append((documents.copy(), valid_mask.copy()))
            return real_train(optimizer, documents, valid_mask)

        def forward_spy(model, tokens):
            logits = real_forward(model, tokens)
            cls.forward_calls.append((tokens.copy(), logits.copy()))
            return logits

        started = time.perf_counter()
        with patch.object(comparison.AdamW, "train_step", train_spy), \
                patch.object(comparison.TrainableNative, "forward", forward_spy):
            cls.report = comparison.run_comparison()
        cls.comparison_elapsed_seconds = time.perf_counter() - started

    def test_frozen_recipe_and_all_seed_worlds_are_unselected(self):
        report = self.report
        self.assertEqual(hashlib.sha256(comparison.PROTOCOL_PATH.read_bytes()).hexdigest(),
                         comparison.EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(report["protocol_sha256"], comparison.EXPECTED_PROTOCOL_SHA256)
        self.assertEqual([row["seed"] for row in report["runs"]], [9201, 9202, 9203])
        for run in report["runs"]:
            manifest = generate_manifest(run["seed"], comparison.WORLD_SPEC)
            self.assertEqual(run["manifest_sha256"], manifest.sha256)
            self.assertEqual(run["manifest"], json.loads(manifest.canonical_bytes()))
            self.assertEqual(len(manifest.events), 40)
            self.assertEqual(sum(e.previous_revision is not None for e in manifest.events), 12)
            self.assertEqual(set(run["arms"]), set(comparison.ARM_NAMES))
            observed = {event.key for event in manifest.events}
            hidden = {tuple(row["key"]) for row in run["negative_control"]["targets"]}
            self.assertEqual(len(hidden), 14)
            self.assertFalse(hidden & observed)
            self.assertEqual(observed | hidden, {(e, r) for e in range(14) for r in range(3)})

    def test_identical_common_branch_includes_adam_moments_and_no_array_aliases(self):
        for run in self.report["runs"]:
            common = run["common_prefix_training"]
            self.assertEqual(common["branch_checkpoint"]["optimizer_step"], 40)
            for arm in run["arms"].values():
                self.assertEqual(arm["branch_checkpoint"], common["branch_checkpoint"])
                self.assertEqual(arm["phase_ends"][0], run["arms"]["frozen"]["phase_ends"][0])
        original = comparison.AdamW(comparison.TrainableNative(comparison.MODEL_CONFIG, seed=2),
                                    comparison.OPTIMIZER_CONFIG)
        event = generate_manifest(comparison.SEEDS[0], comparison.WORLD_SPEC).events[0]
        original.train_step(comparison._document(event), np.array([[False, False, True]]))
        model, copied = comparison.clone_training_state(original)
        self.assertEqual(copied.state_digest(), original.state_digest())
        self.assertEqual(copied.step_number, 1)
        for left, right in ((original.model.parameters, model.parameters),
                            (original.first, copied.first), (original.second, copied.second)):
            for name in left:
                np.testing.assert_array_equal(left[name], right[name])
                self.assertFalse(np.shares_memory(left[name], right[name]))
        original_digest = original.state_digest()
        model.parameters["embedding"][0, 0] += 0.1
        self.assertEqual(original.state_digest(), original_digest)
        self.assertNotEqual(copied.state_digest(), original_digest)

    def test_update_target_input_and_query_counts_match_actual_calls(self):
        self.assertEqual(self.report["total_executed"], {
            "common_pretraining_optimizer_steps": 120, "post_branch_optimizer_steps": 756,
            "common_pretraining_scored_targets": 480, "post_branch_scored_targets": 756})
        self.assertEqual(len(self.training_calls), 876)
        self.assertEqual(sum(int(mask.sum()) for _, mask in self.training_calls), 1236)
        expected = {
            "current_only": (72, 72, 0),
            "replay_matched_total": (72, 36, 36),
            "replay_additive": (108, 72, 36),
            "frozen": (0, 0, 0)}
        for run in self.report["runs"]:
            for arm, (steps, current, replay) in expected.items():
                counts = run["arms"][arm]["counts"]
                self.assertEqual(counts["post_prefix_events"], 36)
                self.assertEqual(counts["optimizer_steps"], steps)
                self.assertEqual(counts["current_target_presentations"], current)
                self.assertEqual(counts["replay_target_presentations"], replay)
                self.assertEqual(counts["scored_target_presentations"], current + replay)
                self.assertEqual(counts["training_documents"], steps)
                self.assertEqual(counts["training_input_tokens"], 3 * steps)
                self.assertEqual(counts["masked_out_target_positions"], 2 * steps)
                self.assertEqual(counts["inference_forward_calls"], 204)
                self.assertEqual(counts["evaluation_query_tokens"], 612)
                self.assertEqual(run["arms"][arm]["final_checkpoint"]["optimizer_step"], 40 + steps)
            self.assertEqual(run["arms"]["replay_additive"]["counts"]["optimizer_steps"],
                             1.5 * run["arms"]["current_only"]["counts"]["optimizer_steps"])
        self.assertEqual(len(self.forward_calls), 2448 + 6)

    def test_training_documents_use_only_declared_current_or_latest_replay_labels(self):
        expected_calls = []
        for run in self.report["runs"]:
            prefix = run["common_prefix_training"]
            expected_calls.extend([(prefix["training_documents"], prefix["scored_target_mask"])] * 40)
            manifest = generate_manifest(run["seed"], comparison.WORLD_SPEC)
            views = tuple(deliveries(manifest))
            for arm_name, arm in run["arms"].items():
                for receipt in arm["events"]:
                    visible = views[receipt["order"]]
                    current = visible.current_event
                    source = {event.event_id: event for event in visible.observed_events}
                    current_steps = sum(row["role"] == "current" for row in receipt["optimizer_updates"])
                    replay_steps = sum(row["role"] == "replay" for row in receipt["optimizer_updates"])
                    self.assertEqual(current_steps, comparison.PROTOCOL["arms"][arm_name]["current_steps_per_event"])
                    self.assertEqual(replay_steps, comparison.PROTOCOL["arms"][arm_name]["replay_steps_per_event"])
                    if replay_steps:
                        selected, category = comparison.select_replay(visible)
                        self.assertEqual(receipt["replay_event_id"], selected.event_id)
                        self.assertEqual(receipt["replay_category"], category)
                    for update in receipt["optimizer_updates"]:
                        actual = source[update["source_event_id"]]
                        self.assertLessEqual(actual.order, current.order)
                        self.assertEqual(actual, current if update["role"] == "current"
                                         else comparison._latest(visible.observed_events)[actual.key])
                        self.assertEqual(update["key"], list(actual.key))
                        self.assertEqual(update["target_value"], actual.value)
                        self.assertEqual(update["training_document_tokens"],
                                         list(comparison.query_tokens(actual.key)) + [18 + actual.value])
                        self.assertEqual(update["scored_target_mask"], [False, False, True])
                        expected_calls.append(([update["training_document_tokens"]],
                                               [update["scored_target_mask"]]))
        self.assertEqual(len(expected_calls), len(self.training_calls))
        for (documents, mask), (expected_documents, expected_mask) in zip(self.training_calls, expected_calls):
            self.assertEqual(documents.tolist(), expected_documents)
            self.assertEqual(mask.tolist(), expected_mask)
            self.assertTrue(np.all(documents[:, :-1] < 18))

    def test_prediction_receipts_are_actual_native_full_head_outputs(self):
        expected = []
        for run in self.report["runs"]:
            for arm in run["arms"].values():
                for phase in arm["phase_ends"]:
                    expected.extend(event["prediction_after_write"] for event in arm["events"]
                                    if event["phase"] == phase["phase"])
                    expected.extend(phase["known_predictions"])
                    expected.extend(phase["negative_control_predictions"])
        calls = [(tokens, logits) for tokens, logits in self.forward_calls if tokens.shape[0] == 1]
        self.assertEqual(len(calls), len(expected))
        self.assertEqual(len(calls), 2448)
        for (tokens, logits), receipt in zip(calls, expected):
            np.testing.assert_array_equal(tokens[0], receipt["query_tokens"])
            np.testing.assert_array_equal(logits[0, -1], receipt["logits"])
            self.assertEqual(logits.shape, (1, 3, 22))
            self.assertTrue(np.isfinite(logits).all())
            self.assertTrue(np.all(tokens < 18))
            token = int(np.argmax(logits[0, -1]))
            self.assertEqual(receipt["raw_argmax_token"], token)
            self.assertEqual(receipt["prediction"], token - 18 if 18 <= token < 22 else None)

    def test_frozen_state_stays_fixed_and_trainable_arms_make_real_updates(self):
        for run in self.report["runs"]:
            for arm_name, arm in run["arms"].items():
                branch = arm["branch_checkpoint"]
                updates = [update for event in arm["events"] for update in event["optimizer_updates"]]
                if arm_name == "frozen":
                    self.assertEqual(updates, [])
                    for checkpoint in [event["checkpoint_after"] for event in arm["events"]]:
                        self.assertEqual(checkpoint["model_sha256"], branch["model_sha256"])
                        self.assertEqual(checkpoint["optimizer_state_sha256"], branch["optimizer_state_sha256"])
                    self.assertEqual(arm["final_checkpoint"]["optimizer_step"], 40)
                else:
                    self.assertNotEqual(arm["final_checkpoint"]["model_sha256"], branch["model_sha256"])
                    self.assertTrue(all(update["optimizer"]["actual_displacement_l2"] > 0 for update in updates))
                    self.assertEqual([u["optimizer"]["step"] for u in updates],
                                     list(range(41, 41 + len(updates))))
                previous = branch
                for event in arm["events"]:
                    self.assertEqual(event["checkpoint_before"], previous)
                    self.assertEqual(event["prediction_after_write"]["model_sha256"],
                                     event["checkpoint_after"]["model_sha256"])
                    self.assertEqual(event["prediction_after_write"]["optimizer_step"],
                                     event["checkpoint_after"]["optimizer_step"])
                    previous = event["checkpoint_after"]
                self.assertEqual(previous, arm["final_checkpoint"])

    def test_all_assignment_horizons_retain_failed_acquisition_and_missing_partitions(self):
        for run in self.report["runs"]:
            reference = run["arms"]["frozen"]["correction_metrics"]
            for arm in run["arms"].values():
                metrics = arm["correction_metrics"]
                self.assertEqual(metrics["cohorts"]["all_assignments"]["immediately_after_write"]["events"], 12)
                for cohort_name, cohort in metrics["cohorts"].items():
                    immediate = cohort["immediately_after_write"]
                    self.assertEqual(immediate["events"], immediate["measured"])
                    self.assertEqual(immediate["missing"], 0)
                    for h in comparison.HORIZONS:
                        row = cohort["phase_end_horizons"][str(h)]
                        ref = reference["cohorts"][cohort_name]["phase_end_horizons"][str(h)]
                        for field in ("total_events", "eligible", "censored", "not_yet_due"):
                            self.assertEqual(row[field], ref[field])
                        self.assertEqual(row["total_events"], row["eligible"] + row["censored"] + row["not_yet_due"])
                        self.assertEqual(row["eligible"], row["measured"] + row["missing_measurement"])
                        self.assertEqual(row["missing_measurement"], 0)
                        self.assertEqual(row["eligible"], row["acquired_only_eligible"]
                                         + row["failed_initial_acquisition_eligible"]
                                         + row["initial_acquisition_unknown_eligible"])
                        self.assertEqual(row["accuracy"], row["correct"] / row["eligible"] if row["eligible"] else None)
                self.assertEqual(arm["post_prefix_new_fact_acquisition"]["total"], 24)
                for phase in arm["phase_ends"]:
                    self.assertEqual(phase["current_observed_facts"]["total"], 4 * (phase["phase"] + 1))
                    self.assertEqual(phase["unobserved_random_label_control"]["total"], 2 * (phase["phase"] + 1))

    def test_replay_priority_and_fallbacks_have_explicit_source_accounting(self):
        first = Event("a", 0, 0, 0, 0, 0)
        other = Event("b", 0, 1, 1, 0, 1)
        sibling = Event("c", 0, 2, 0, 1, 2)
        revised = Event("d", 0, 3, 0, 0, 3, "a", 0)
        examples = [
            ((first,), first, "current_event_fallback_no_other_key"),
            ((first, other), first, "other_entity_fallback"),
            ((first, other, sibling), first, "same_entity_other_key"),
            ((first, other, sibling, revised), sibling, "same_entity_other_key"),
        ]
        for events, expected, category in examples:
            targets = tuple((e.entity, e.relation, e.value, e.event_id)
                            for _, e in sorted(comparison._latest(events).items()))
            selected = comparison.select_replay(VisiblePrefix(events[-1], events, targets))
            self.assertEqual(selected, (expected, category))
        for run in self.report["runs"]:
            for name in ("replay_matched_total", "replay_additive"):
                arm = run["arms"][name]
                counts = arm["counts"]
                self.assertEqual(counts["same_entity_replay_steps"] + counts["other_entity_replay_steps"]
                                 + counts["current_event_replay_fallback_steps"], 36)
                self.assertEqual(counts["other_entity_replay_steps"], 12)
                self.assertEqual(counts["same_entity_replay_steps"], 24)
                self.assertEqual(counts["current_event_replay_fallback_steps"], 0)
                self.assertEqual(arm["events"][0]["replay_category"], "other_entity_fallback")
                self.assertEqual(arm["events"][1]["replay_category"], "same_entity_other_key")
                self.assertGreater(counts["maximum_algorithmic_replay_payload_json_bytes"], 0)
            for name in ("current_only", "frozen"):
                self.assertEqual(run["arms"][name]["counts"]["maximum_algorithmic_replay_payload_json_bytes"], 0)

    def test_invalid_visible_target_table_rejected_before_any_state_mutation(self):
        manifest = generate_manifest(9201, comparison.WORLD_SPEC)
        common = manifest.events[:4]
        optimizer = comparison.AdamW(comparison.TrainableNative(comparison.MODEL_CONFIG, seed=1),
                                     comparison.OPTIMIZER_CONFIG)
        learner = comparison.NativeAllocationLearner(optimizer, common, "replay_matched_total")
        view = tuple(deliveries(manifest))[4]
        bad = replace(view, current_targets=view.current_targets + ((13, 2, 3, "future"),))
        before = learner.checkpoint()
        with self.assertRaises(ValueError):
            learner.observe(bad)
        self.assertEqual(learner.checkpoint(), before)
        self.assertEqual(learner.counts["optimizer_steps"], 0)
        with self.assertRaises(ValueError):
            learner.observe(tuple(deliveries(manifest))[5])
        self.assertEqual(learner.checkpoint(), before)

    def test_future_value_counterfactual_preserves_all_earlier_states(self):
        manifest = generate_manifest(9201, comparison.WORLD_SPEC)
        last = manifest.events[-1]
        changed = Manifest(manifest.seed, manifest.spec,
                           manifest.events[:-1] + (replace(last, value=(last.value + 1) % 4),))
        baseline = self.report["runs"][0]
        counterfactual = comparison.run_seed(9201, changed, ("replay_matched_total",))
        self.assertNotEqual(baseline["manifest_sha256"], counterfactual["manifest_sha256"])
        self.assertEqual(baseline["common_prefix_training"], counterfactual["common_prefix_training"])
        self.assertEqual(baseline["negative_control"], counterfactual["negative_control"])
        left, right = baseline["arms"]["replay_matched_total"], counterfactual["arms"]["replay_matched_total"]
        self.assertEqual(left["events"][:-1], right["events"][:-1])
        self.assertEqual(left["phase_ends"][:-1], right["phase_ends"][:-1])

    def test_negative_control_label_counterfactual_cannot_change_training_or_predictions(self):
        baseline = self.report["runs"][0]
        make_targets = comparison.negative_control_targets

        def alternate(manifest):
            return {key: (value + 1) % 4 for key, value in make_targets(manifest).items()}

        with patch.object(comparison, "negative_control_targets", alternate):
            counterfactual = comparison.run_seed(9201, arm_names=("replay_matched_total",))
        self.assertEqual(baseline["manifest_sha256"], counterfactual["manifest_sha256"])
        self.assertEqual(baseline["common_prefix_training"], counterfactual["common_prefix_training"])
        self.assertNotEqual(baseline["negative_control"]["targets_sha256"],
                            counterfactual["negative_control"]["targets_sha256"])
        left, right = baseline["arms"]["replay_matched_total"], counterfactual["arms"]["replay_matched_total"]
        for field in ("events", "final_checkpoint", "counts", "correction_metrics", "evaluation_log"):
            self.assertEqual(left[field], right[field])
        for a, b in zip(left["phase_ends"], right["phase_ends"]):
            self.assertEqual({k: v for k, v in a.items() if k != "unobserved_random_label_control"},
                             {k: v for k, v in b.items() if k != "unobserved_random_label_control"})

    def test_same_runtime_repeat_is_identical_for_one_seed_and_arm(self):
        repeated = comparison.run_seed(9202, arm_names=("current_only",))
        baseline = self.report["runs"][1]
        self.assertEqual(repeated, {**baseline, "arms": {"current_only": baseline["arms"]["current_only"]}})

    def test_paired_summaries_preserve_undefined_seed_endpoints(self):
        self.assertEqual(comparison._summary([0.5, None, 0.0]),
                         {"values": [0.5, None, 0.0], "available_seed_count": 2,
                          "mean": None, "min": None, "max": None, "complete": False})
        for name, row in self.report["paired_seed_differences"].items():
            arm_name = name.removesuffix("_minus_current_only")
            for metric, summary in row.items():
                expected = []
                for run in self.report["runs"]:
                    left = comparison._endpoints(run["arms"][arm_name])[metric]
                    right = comparison._endpoints(run["arms"]["current_only"])[metric]
                    expected.append(None if left is None or right is None else left - right)
                self.assertEqual(summary, comparison._summary(expected))

    def test_cli_stdout_and_exclusive_output_use_identical_receipt(self):
        # The cached real report avoids repeating the whole experiment for I/O assertions.
        json_report = json.loads(json.dumps(self.report, allow_nan=False))
        with patch.object(comparison, "run_comparison", return_value=self.report):
            stream = io.StringIO()
            with patch.object(sys, "argv", ["native_update_comparison.py"]), redirect_stdout(stream):
                comparison.main()
            stdout_report = json.loads(stream.getvalue())
            self.assertEqual(stdout_report, json_report)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "new_receipt.json"
                with patch.object(sys, "argv", ["native_update_comparison.py", "--out", str(path)]):
                    comparison.main()
                saved = path.read_bytes()
                self.assertEqual(json.loads(saved), json_report)
                with patch.object(sys, "argv", ["native_update_comparison.py", "--out", str(path)]), \
                        patch.object(comparison, "run_comparison", side_effect=AssertionError("must refuse before running")):
                    with self.assertRaises(FileExistsError):
                        comparison.main()
                self.assertEqual(path.read_bytes(), saved)
                dangling = Path(directory) / "dangling.json"
                dangling.symlink_to(Path(directory) / "missing.json")
                with patch.object(sys, "argv", ["native_update_comparison.py", "--out", str(dangling)]):
                    with self.assertRaises(FileExistsError):
                        comparison.main()
                self.assertFalse((Path(directory) / "missing.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
