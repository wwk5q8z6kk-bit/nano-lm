"""Targeted checks for evidence boundaries and the optimizer reanalysis arithmetic."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).parents[1] / "analysis/reconcile_optimizer_factorial.py"
SPEC = importlib.util.spec_from_file_location("optimizer_audit", MODULE)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def full_cells(value=.5):
    return {f"{a}/{o}": {s: value for s in audit.FACTORIAL_SEEDS} for a, o in audit.CELLS}


def synthetic_run():
    log = [{"phase": phase, "acquire": .75 if phase >= 45 else .5,
            "retain": .625, "slots_used": 100, "final_loss": 1.}
           for phase in range(60)]
    return {"address": "original", "opt": "adam", "seed": 3101, "lr": .001,
            "log": log, "late_acquire": .75, "late_retain": .625, "early_acquire": .5}


class OptimizerAuditChecks(unittest.TestCase):
    def test_late_window_uses_phases_46_to_60(self):
        row = synthetic_run()
        result = audit.summarize_factorial_run(row, "original", "adam", 3101, .001)
        self.assertEqual(result["late_acquire"], .75)
        self.assertEqual(result["early_acquire"], .5)

    def test_summary_or_phase_identity_mismatch_is_rejected(self):
        for mutation in ("summary", "phase", "seed", "nonfinite"):
            row = synthetic_run()
            if mutation == "summary": row["late_acquire"] = .7
            if mutation == "phase": row["log"][-1]["phase"] = 58
            if mutation == "seed": row["seed"] = 3102
            if mutation == "nonfinite": row["log"][0]["final_loss"] = float("nan")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                audit.summarize_factorial_run(row, "original", "adam", 3101, .001)

    def test_registered_strict_and_inclusive_boundaries(self):
        d = full_cells()
        d["original/adam"] = dict.fromkeys(audit.FACTORIAL_SEEDS, .2)
        d["hash/adam"] = dict.fromkeys(audit.FACTORIAL_SEEDS, .9)
        d["hash/adam_reset"] = dict.fromkeys(audit.FACTORIAL_SEEDS, .9)
        d["original/adam_reset"] = dict.fromkeys(audit.FACTORIAL_SEEDS, .7)
        d["original/adam_b2_09"] = dict.fromkeys(audit.FACTORIAL_SEEDS, .3)
        rules = audit.prediction_counts(d, True)
        self.assertEqual(rules["counts"]["F1"], 0)
        self.assertEqual(rules["counts"]["F2"], 0)
        self.assertEqual(rules["verdicts"]["F3"], "holds")
        self.assertEqual(rules["verdicts"]["F5"], "partial")

    def test_f4_requires_every_seed_and_missing_seed_is_rejected(self):
        d = full_cells()
        d["hash/adam_reset"][3105] = .56
        self.assertEqual(audit.prediction_counts(d, True)["verdicts"]["F4"], "fails")
        del d["original/adam"][3105]
        with self.assertRaises(ValueError): audit.prediction_counts(d, True)

    def test_sgd_selection_tie_and_uninformative_gate(self):
        s = audit.choose_sgd_lr([(1, .995), (3, 1.), (10, .98), (30, .7), (100, .6), (300, .4)])
        self.assertEqual(s["chosen_lr"], 1)
        self.assertTrue(s["informative"])
        self.assertEqual(audit.prediction_counts(full_cells(.8), False)["verdicts"]["F6"], "uninformative")
        with self.assertRaises(ValueError): audit.choose_sgd_lr([(1, 1.)] * 6)

    def test_conditional_recovery_and_interaction_are_distinct(self):
        d = full_cells()
        for name, value in [("original/adam", .2), ("original/adam_reset", .6),
                            ("hash/adam", 1.), ("hash/adam_reset", 1.)]:
            d[name] = dict.fromkeys(audit.FACTORIAL_SEEDS, value)
        result = audit.paired_contrasts(d)
        self.assertAlmostEqual(result["interaction_reset_original_minus_hash"]["mean"], .4)
        self.assertAlmostEqual(result["reset_recovery_ratio_of_means"], .5)

    def test_detached_candidates_do_not_certify_joint_commit(self):
        result = audit.independent_candidate_counterexample()
        self.assertTrue(all(x >= 1 for x in result["individual_target_margins"]))
        self.assertTrue(all(not all(x >= 1 for x in candidate)
                            for candidate in result["joint_target_margins_for_each_candidate"]))
        self.assertTrue(all(x >= 1 for x in result["jointly_feasible_state"]))

    def test_report_creation_is_exclusive_and_nonfinite_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "report.json"
            audit.write_report({"value": 1}, p)
            with self.assertRaises(FileExistsError): audit.write_report({"value": 2}, p)
            self.assertEqual(json.loads(p.read_text()), {"value": 1})
            q = Path(directory) / "nonfinite.json"
            with self.assertRaises(ValueError): audit.write_report({"value": float("nan")}, q)
            self.assertFalse(q.exists())


if __name__ == "__main__":
    unittest.main()
