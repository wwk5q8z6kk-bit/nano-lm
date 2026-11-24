"""Risk-based checks for V6's evaluation instrument; no learned model is run."""
import copy
from dataclasses import fields
from pathlib import Path
import shutil
import tempfile
import unittest

import generative_reasoning_v6 as g


class InstrumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="nanolm-v6-instrument-")
        cls.directory = Path(cls.temp.name) / "dataset"
        cls.dataset = g.generate_dataset(cls.directory)
        cls.public = {r["id"]: r for r in cls.dataset.public}
        cls.oracle = {r["id"]: r for r in cls.dataset.oracle}
        cls.control_run = g.make_run_manifest(cls.dataset, "symbolic-instrument-test")
        cls.symbolic = g.control_predictions(cls.dataset, g.symbolic_candidate)
        cls.perfect = g.score(cls.dataset, cls.symbolic, cls.control_run)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def model_manifest(self, context_limit=512):
        # Deliberately synthetic identities: tests validate receipt mechanics only.
        identity = {
            "checkpoint_sha256": "0" * 64, "tokenizer_sha256": "1" * 64,
            "model_config_sha256": "2" * 64, "adapter_source_sha256": "3" * 64,
            "context_limit": context_limit,
        }
        return g.make_run_manifest(
            self.dataset, "synthetic_adapter_fixture_not_a_model_run",
            "model_checkpoint_evaluation", identity)

    def model_prediction(self, oracle_row, raw=None, stop="eos"):
        run = self.model_manifest()
        public = self.public[oracle_row["id"]]
        outputs = list(range(32)) if stop == "max_new_tokens" else [5, 6]
        receipt = g.model_receipt(public["prompt"], run, [7, 8, 9], outputs, stop)
        return run, {
            "id": oracle_row["id"], "status": "generated",
            "raw_text": oracle_row["expected_answer"] if raw is None else raw,
            "receipt": receipt,
        }

    def refresh_row(self, public, oracle_row):
        prompt = g.render_prompt(oracle_row["facts"], oracle_row["family"],
                                 oracle_row["query_entity"])
        public[oracle_row["id"]]["prompt"] = prompt
        oracle_row["expected_answer"] = g.semantic_answer(
            oracle_row["facts"], oracle_row["family"], oracle_row["query_entity"])
        oracle_row["prompt_sha256"] = g.text_sha(prompt)
        oracle_row["prompt_utf8_bytes"] = len(prompt.encode())
        oracle_row["context_utf8_bytes"] = len(g.context_text(oracle_row["facts"]).encode())

    def test_01_freeze_and_deterministic_generation(self):
        self.assertEqual(g.sha(g.PROTOCOL_PATH.read_bytes()), g.FROZEN_PROTOCOL_SHA256)
        public, oracle = g.build_dataset()
        self.assertEqual(public, self.dataset.public)
        self.assertEqual(oracle, self.dataset.oracle)
        self.assertEqual(self.dataset.manifest["summary"]["worlds"], 96)
        self.assertEqual(self.dataset.manifest["summary"]["records"], 480)
        self.assertEqual(self.dataset.manifest["contamination_status"], "NOT_CHECKED")

    def test_02_independent_parser_matches_visible_gold(self):
        for row in self.dataset.oracle:
            prompt = self.public[row["id"]]["prompt"]
            self.assertEqual(g.symbolic_candidate(g.CandidateView(prompt)), row["expected_answer"])
        for split in g.SPLITS:
            for family in ("all", *g.FAMILIES):
                metric = self.perfect["metrics"][split][family]
                self.assertEqual(metric["primary_counterfactual_joint"]["rate"], 1)
                self.assertEqual(metric["permutation_joint_correct"]["rate"], 1)
                self.assertEqual(metric["unknown_abstention"]["rate"], 1)

    def test_03_no_hidden_metadata_in_candidate_view(self):
        self.assertEqual([f.name for f in fields(g.CandidateView)], ["prompt"])
        view = g.CandidateView(self.dataset.public[0]["prompt"])
        self.assertFalse(hasattr(view, "__dict__"))
        for name in ("condition", "seed", "oracle", "expected_answer", "world_id", "id"):
            self.assertFalse(hasattr(view, name))
        for row in self.dataset.oracle:
            prompt = self.public[row["id"]]["prompt"]
            self.assertNotIn(row["world_id"], prompt)
            self.assertNotIn(row["condition"], prompt)

    def test_04_pair_same_query_different_required_answers(self):
        groups = {}
        for row in self.dataset.oracle:
            groups.setdefault(row["world_id"], {})[row["condition"]] = row
        for group in groups.values():
            a, b = group["full_a"], group["full_b"]
            self.assertEqual(a["query_entity"], b["query_entity"])
            self.assertNotEqual(a["expected_answer"], b["expected_answer"])
            self.assertEqual(sum(x != y for x, y in zip(a["facts"], b["facts"])), 1)

    def test_05_same_answer_shortcuts_cannot_pass_primary(self):
        for name, candidate in (
            ("unknown", lambda v: "UNKNOWN"),
            ("constant", lambda v: "value_000000000000"),
            ("first", lambda v: g.value_control(v)),
            ("last", lambda v: g.value_control(v, last=True)),
        ):
            result = g.score(self.dataset, g.control_predictions(self.dataset, candidate), self.control_run)
            for split in g.SPLITS:
                primary = result["metrics"][split]["all"]["primary_counterfactual_joint"]["rate"]
                self.assertLess(primary, 1, name)
                if name in ("unknown", "constant"):
                    self.assertEqual(primary, 0, name)

    def test_06_missing_outputs_remain_in_all_denominators(self):
        row = next(r for r in self.dataset.oracle if r["condition"] == "full_b")
        predictions = [p for p in self.symbolic if p["id"] != row["id"]]
        result = g.score(self.dataset, predictions, self.control_run)
        metric = result["metrics"][row["split"]][row["family"]]
        self.assertEqual(metric["primary_counterfactual_joint"]["correct"], metric["worlds"] - 1)
        self.assertEqual(metric["primary_counterfactual_joint"]["denominator"], metric["worlds"])
        self.assertEqual(result["denominators"]["registered_records"], 480)
        self.assertEqual(result["denominators"]["status_counts"]["missing"], 1)

    def test_07_all_missing_is_zero_not_filtered(self):
        result = g.score(self.dataset, [], self.control_run)
        self.assertEqual(result["denominators"]["status_counts"]["missing"], 480)
        for split in g.SPLITS:
            self.assertEqual(result["metrics"][split]["all"]["primary_counterfactual_joint"]["rate"], 0)
            self.assertEqual(result["metrics"][split]["all"]["unknown_abstention"]["rate"], 0)

    def test_08_original_target_gain_is_separate_from_abstention(self):
        predictions = copy.deepcopy(self.symbolic)
        for prediction in predictions:
            row = self.oracle[prediction["id"]]
            if row["condition"] == "no_context":
                prediction["raw_text"] = row["a_target"]  # Deliberate private-gold scoring fixture.
        result = g.score(self.dataset, predictions, self.control_run)
        for split in g.SPLITS:
            metric = result["metrics"][split]["all"]
            self.assertEqual(metric["original_target_contextual_gain"]["paired_difference"], 0)
            self.assertEqual(metric["condition_policy_accuracy"]["no_context"]["rate"], 0)
            self.assertEqual(metric["condition_policy_accuracy"]["full_a"]["rate"], 1)
            self.assertEqual(self.perfect["metrics"][split]["all"]["original_target_contextual_gain"]["paired_difference"], 1)

    def test_09_duplicate_and_unknown_prediction_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate prediction"):
            g.score(self.dataset, [self.symbolic[0], self.symbolic[0]], self.control_run)
        bad = copy.deepcopy(self.symbolic[0])
        bad["id"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unknown prediction"):
            g.score(self.dataset, [bad], self.control_run)

    def test_10_private_gold_tampering_rejected(self):
        oracle = copy.deepcopy(self.dataset.oracle)
        oracle[0]["expected_answer"] = "value_000000000000"
        with self.assertRaisesRegex(ValueError, "gold does not follow"):
            g.validate_dataset(self.dataset.public, oracle)

    def test_11_planted_cross_split_overlap_detected(self):
        public = copy.deepcopy(self.dataset.public)
        by_id = {r["id"]: r for r in public}
        oracle = copy.deepcopy(self.dataset.oracle)
        world_ids = {}
        for split in g.SPLITS:
            world_ids[split] = next(r["world_id"] for r in oracle
                                   if r["split"] == split and r["family"] == "two_hop")
        development = {r["condition"]: r for r in oracle if r["world_id"] == world_ids["development"]}
        for row in oracle:
            if row["world_id"] == world_ids["evaluation"]:
                source = development[row["condition"]]
                for key in ("facts", "query_entity", "a_target"):
                    row[key] = copy.deepcopy(source[key])
                self.refresh_row(by_id, row)
        with self.assertRaisesRegex(ValueError, "cross-split"):
            g.validate_dataset(public, oracle)

    def test_12_counterfactual_collapsed_to_a_rejected(self):
        public = copy.deepcopy(self.dataset.public)
        by_id = {r["id"]: r for r in public}
        oracle = copy.deepcopy(self.dataset.oracle)
        target = next(r for r in oracle if r["condition"] == "full_b")
        a = next(r for r in oracle if r["world_id"] == target["world_id"] and r["condition"] == "full_a")
        target["facts"] = copy.deepcopy(a["facts"])
        self.refresh_row(by_id, target)
        with self.assertRaisesRegex(ValueError, "counterfactual targets"):
            g.validate_dataset(public, oracle)

    def test_13_duplicate_public_and_incomplete_world_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate public"):
            g.validate_dataset(self.dataset.public + [self.dataset.public[0]], self.dataset.oracle)
        remove_id = self.dataset.public[0]["id"]
        with self.assertRaisesRegex(ValueError, "incomplete world"):
            g.validate_dataset([r for r in self.dataset.public if r["id"] != remove_id],
                               [r for r in self.dataset.oracle if r["id"] != remove_id])

    def test_14_output_policy_is_strict_ascii_boundary_only(self):
        value = "value_0123456789ab"
        self.assertEqual(g.normalized_answer(" \t" + value + "\r\n"), value)
        self.assertEqual(g.normalized_answer(" UNKNOWN\n"), "UNKNOWN")
        for invalid in (
            value + ".", "Answer: " + value, value.upper(), value + "\u00a0",
            "\u200b" + value, value + "\nUNKNOWN", "", "unknown", "value_012 456789ab",
        ):
            self.assertIsNone(g.normalized_answer(invalid), repr(invalid))

    def test_15_nonfinite_and_duplicate_json_keys_rejected(self):
        for raw in ('{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', '{"x":1,"x":2}'):
            with self.assertRaises(ValueError):
                g.strict_loads(raw)
        bad = copy.deepcopy(self.control_run)
        bad["run_id"] = float("nan")
        with self.assertRaises(ValueError):
            g.score(self.dataset, [], bad)

    def test_16_immutable_generation_and_artifact_hashes(self):
        with self.assertRaisesRegex(ValueError, "destination exists"):
            g.generate_dataset(self.directory)
        with tempfile.TemporaryDirectory() as root:
            copied = Path(root) / "copied"
            shutil.copytree(self.directory, copied)
            with (copied / "public.jsonl").open("ab") as out:
                out.write(b"\n")
            with self.assertRaisesRegex(ValueError, "hash/byte"):
                g.load_dataset(copied)

    def test_17_run_protocol_and_dataset_binding_rejected(self):
        for field in ("protocol_sha256", "dataset_manifest_sha256"):
            bad = copy.deepcopy(self.control_run)
            bad[field] = "f" * 64
            with self.assertRaisesRegex(ValueError, "binding"):
                g.score(self.dataset, [], bad)

    def test_18_unknown_actual_model_token_counts_are_not_eligible(self):
        row = next(r for r in self.dataset.oracle if r["condition"] == "full_a")
        run, prediction = self.model_prediction(row)
        prediction["receipt"]["input_token_ids"] = None
        prediction["receipt"]["input_tokens"] = None
        prediction["receipt"]["actual_model_tokenizer"] = False
        with self.assertRaisesRegex(ValueError, "actual tokenizer counts"):
            g.score(self.dataset, [prediction], run)

    def test_19_budget_exhaustion_is_censored_even_if_text_matches(self):
        row = next(r for r in self.dataset.oracle if r["condition"] == "full_a")
        run, prediction = self.model_prediction(row, stop="max_new_tokens")
        result = g.score(self.dataset, [prediction], run)
        self.assertEqual(result["denominators"]["budget_exhausted"], 1)
        metric = result["metrics"][row["split"]][row["family"]]
        self.assertEqual(metric["condition_policy_accuracy"]["full_a"]["correct"], 0)
        self.assertEqual(result["denominators"]["registered_records"], 480)

    def test_20_model_receipt_identity_token_and_truncation_errors(self):
        row = self.dataset.oracle[0]
        for field, value in (
            ("input_tokens", 999), ("truncated", True), ("reset_state_per_case", False),
            ("context_limit", 999), ("output_token_ids", [float("nan")]),
        ):
            run, prediction = self.model_prediction(row)
            prediction["receipt"][field] = value
            with self.assertRaises(ValueError, msg=field):
                g.score(self.dataset, [prediction], run)
        run, prediction = self.model_prediction(row)
        prediction["receipt"]["model"]["tokenizer_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "identity"):
            g.score(self.dataset, [prediction], run)

    def test_21_control_receipt_cannot_be_relabelled_as_model(self):
        with self.assertRaises(ValueError):
            g.score(self.dataset, self.symbolic, self.model_manifest())
        self.assertEqual(self.perfect["model_execution_status"], "NOT_RUN")
        self.assertIsNone(self.perfect["exposure"]["actual_model_input_tokens"])
        self.assertEqual(self.perfect["exposure"]["model_tokenizer_context_eligibility"], "NOT_CHECKED")

    def make_fake_adapter(self, ids=None, mismatch=False, truncated=False, context_limit=100):
        run = self.model_manifest(context_limit)
        class FakeAdapter:
            identity = run["model"]
            reset_calls = 0
            generation_calls = 0
            views = []
            def reset(self):
                self.reset_calls += 1
            def encode_prompt(self, view):
                self.views.append(view)
                return [1, 2] if ids is None else list(ids)
            def generate(self, view):
                self.generation_calls += 1
                self.views.append(view)
                actual = [1, 2] if ids is None else list(ids)
                return g.Completion(g.symbolic_candidate(view),
                                    actual + [3] if mismatch else actual, [5, 6], "eos", truncated)
        return run, FakeAdapter()

    def test_22_runner_prompt_only_boundary_and_state_reset(self):
        run, adapter = self.make_fake_adapter()
        small = g.Dataset(self.dataset.public[:3], [], self.dataset.manifest, self.dataset.manifest_sha256)
        predictions = g.run_candidate(small, adapter, run)
        self.assertEqual(adapter.reset_calls, 3)
        self.assertEqual(adapter.generation_calls, 3)
        self.assertTrue(all(p["status"] == "generated" for p in predictions))
        for view in adapter.views:
            self.assertIsInstance(view, g.CandidateView)
            self.assertFalse(hasattr(view, "id"))
            self.assertFalse(hasattr(view, "expected_answer"))

    def test_23_runner_context_preflight_never_generates_or_crops(self):
        run, adapter = self.make_fake_adapter(ids=list(range(69)), context_limit=100)
        small = g.Dataset(self.dataset.public[:1], [], self.dataset.manifest, self.dataset.manifest_sha256)
        predictions = g.run_candidate(small, adapter, run)
        self.assertEqual(adapter.generation_calls, 0)
        self.assertEqual(predictions[0]["status"], "context_ineligible")
        self.assertEqual(len(predictions[0]["receipt"]["input_token_ids"]), 69)
        result = g.score(self.dataset, predictions, run)
        self.assertEqual(result["denominators"]["status_counts"]["context_ineligible"], 1)
        self.assertEqual(result["denominators"]["status_counts"]["missing"], 479)
        self.assertEqual(result["exposure"]["actual_model_input_tokens"], 0)

    def test_24_runner_encoding_mismatch_or_crop_becomes_retained_error(self):
        for kwargs in ({"mismatch": True}, {"truncated": True}):
            run, adapter = self.make_fake_adapter(**kwargs)
            small = g.Dataset(self.dataset.public[:1], [], self.dataset.manifest, self.dataset.manifest_sha256)
            predictions = g.run_candidate(small, adapter, run)
            self.assertEqual(predictions[0]["status"], "error")
            result = g.score(self.dataset, predictions, run)
            self.assertEqual(result["denominators"]["status_counts"]["error"], 1)
            self.assertEqual(result["denominators"]["registered_records"], 480)
            self.assertEqual(result["exposure"]["actual_model_input_tokens"], 2)
            self.assertEqual(result["exposure"]["failed_generation_output_token_counts_unknown"], 1)

    def test_25_first_newline_stop_and_wrong_prompt_hash_rejected(self):
        row = self.dataset.oracle[0]
        run, prediction = self.model_prediction(row, raw="UNKNOWN\nextra", stop="newline")
        with self.assertRaisesRegex(ValueError, "first-newline"):
            g.score(self.dataset, [prediction], run)
        prediction = copy.deepcopy(self.symbolic[0])
        prediction["receipt"]["prompt_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "prompt hash"):
            g.score(self.dataset, [prediction], self.control_run)

    def test_26_wrong_format_is_in_denominator(self):
        bad = copy.deepcopy(self.symbolic)
        row = next(r for r in self.dataset.oracle if r["condition"] == "full_a")
        prediction = next(p for p in bad if p["id"] == row["id"])
        prediction["raw_text"] += "."
        result = g.score(self.dataset, bad, self.control_run)
        self.assertEqual(result["denominators"]["format_invalid"], 1)
        metric = result["metrics"][row["split"]][row["family"]]
        self.assertEqual(metric["primary_counterfactual_joint"]["correct"], metric["worlds"] - 1)

    def test_27_invalid_records_and_conflicting_facts_rejected(self):
        bad = copy.deepcopy(self.symbolic[0])
        bad["oracle_answer"] = "forbidden"
        with self.assertRaisesRegex(ValueError, "invalid keys"):
            g.score(self.dataset, [bad], self.control_run)
        row = next(r for r in self.dataset.oracle if r["condition"] == "full_a")
        fact = row["facts"][0]
        with self.assertRaisesRegex(ValueError, "duplicate or conflicting"):
            g.semantic_answer(row["facts"] + [fact], row["family"], row["query_entity"])



    def test_28_coherent_in_memory_world_edit_cannot_keep_old_dataset_identity(self):
        altered = copy.deepcopy(self.dataset)
        public = {r["id"]: r for r in altered.public}
        world = next(r["world_id"] for r in altered.oracle if r["family"] == "direct_retrieval")
        old = next(r["expected_answer"] for r in altered.oracle
                   if r["world_id"] == world and r["condition"] == "full_a")
        replacement = "value_deadbeefcafe"
        for row in altered.oracle:
            if row["world_id"] != world:
                continue
            row["a_target"] = replacement
            for fact in row["facts"]:
                if fact["object"] == old:
                    fact["object"] = replacement
            self.refresh_row(public, row)
        g.validate_dataset(altered.public, altered.oracle)  # Coherent semantics still pass.
        with self.assertRaisesRegex(ValueError, "content binding"):
            g.score(altered, g.control_predictions(altered, g.symbolic_candidate), self.control_run)
        altered = copy.deepcopy(self.dataset)
        altered.manifest["generated_at_utc"] = "forged"
        with self.assertRaisesRegex(ValueError, "manifest identity"):
            g.score(altered, self.symbolic, self.control_run)

    def test_29_eos_or_budget_label_cannot_bypass_first_newline(self):
        row = next(r for r in self.dataset.oracle if r["condition"] == "full_a")
        target = row["expected_answer"]
        for raw in ("\n" + target, target + "\n", target + "\n\n", "\n" + target + "\n"):
            for stop in ("eos", "max_new_tokens"):
                run, prediction = self.model_prediction(row, raw=raw, stop=stop)
                with self.assertRaisesRegex(ValueError, "first-newline"):
                    g.score(self.dataset, [prediction], run)
        run, prediction = self.model_prediction(row, raw=target + "\n", stop="newline")
        g.validate_prediction(prediction, self.public[row["id"]], run)

    def test_30_mutating_encoder_list_cannot_change_preflight_snapshot(self):
        run = self.model_manifest(context_limit=100)
        class MutatingAdapter:
            identity = run["model"]
            def reset(self):
                self.ids = [7, 8, 9]
            def encode_prompt(self, view):
                return self.ids
            def generate(self, view):
                self.ids.pop()
                return g.Completion(g.symbolic_candidate(view), self.ids, [5, 6], "eos")
        small = g.Dataset(self.dataset.public[:1], [], self.dataset.manifest, self.dataset.manifest_sha256)
        predictions = g.run_candidate(small, MutatingAdapter(), run)
        self.assertEqual(predictions[0]["status"], "error")
        self.assertEqual(predictions[0]["receipt"]["input_token_ids"], [7, 8, 9])
        result = g.score(self.dataset, predictions, run)
        self.assertEqual(result["exposure"]["actual_model_input_tokens"], 3)

    def test_31_reused_output_and_input_buffers_do_not_rewrite_previous_receipts(self):
        run = self.model_manifest(context_limit=100)
        class ReusingAdapter:
            identity = run["model"]
            def __init__(self):
                self.inputs, self.outputs, self.count = [7, 8, 9], [5, 6], 0
            def reset(self):
                pass
            def encode_prompt(self, view):
                self.inputs[:] = [7, 8, 9 + self.count]
                return self.inputs
            def generate(self, view):
                self.count += 1
                self.outputs[:] = [5, 5 + self.count]
                return g.Completion(g.symbolic_candidate(view), self.inputs, self.outputs, "eos")
        small = g.Dataset(self.dataset.public[:2], [], self.dataset.manifest, self.dataset.manifest_sha256)
        adapter = ReusingAdapter()
        predictions = g.run_candidate(small, adapter, run)
        self.assertEqual([p["receipt"]["input_token_ids"] for p in predictions], [[7, 8, 9], [7, 8, 10]])
        self.assertEqual([p["receipt"]["output_token_ids"] for p in predictions], [[5, 6], [5, 7]])
        adapter.inputs[:] = [0, 0, 0]
        adapter.outputs[:] = [0, 0]
        self.assertEqual(predictions[0]["receipt"]["input_token_ids"], [7, 8, 9])
        self.assertEqual(predictions[0]["receipt"]["output_token_ids"], [5, 6])
        g.score(self.dataset, predictions, run)


if __name__ == "__main__":
    unittest.main(verbosity=2)

