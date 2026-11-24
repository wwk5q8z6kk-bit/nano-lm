"""Known-answer and independent LP checks for the scoped edit geometry."""
from fractions import Fraction as F
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from update_feasibility_geometry import l1_from_initial, margins, protected_optimum, run


class GeometryChecks(unittest.TestCase):
    def test_initial_and_ideal_hitting_point(self):
        self.assertEqual(margins((2, 1)), {"target_B_minus_A": F(-1), "protected_C_minus_D": F(8, 5)})
        self.assertEqual(margins((0, 3)), {"target_B_minus_A": F(3), "protected_C_minus_D": F(-6, 5)})

    def test_entire_target_only_minimum_segment_samples(self):
        for i in range(-20, 21):
            x = F(i, 10)
            point = (x, x + 3)
            self.assertEqual(l1_from_initial(point), 4)
            self.assertEqual(margins(point)["target_B_minus_A"], 3)
            self.assertLessEqual(margins(point)["protected_C_minus_D"], 0)
        self.assertGreater(l1_from_initial((F(21, 10), F(51, 10))), 4)
        self.assertGreater(l1_from_initial((F(-21, 10), F(9, 10))), 4)

    def test_tie_depends_on_explicit_answer_order(self):
        scores = {"C": F(2), "D": F(2)}
        self.assertEqual(max(["C", "D"], key=scores.get), "C")
        self.assertEqual(max(["D", "C"], key=scores.get), "D")
        self.assertFalse(run()["target_only_minimum_l1"]["tie_is_argmax_error_independent_of_tie_breaking"])

    def test_joint_known_answers(self):
        self.assertEqual(protected_optimum(F(1)), (F(11, 3), F(20, 3)))
        self.assertEqual(l1_from_initial(protected_optimum(F(1))), F(22, 3))
        self.assertEqual(protected_optimum(F(8, 5)), (F(14, 3), F(23, 3)))
        self.assertEqual(l1_from_initial(protected_optimum(F(8, 5))), F(28, 3))

    def test_arbitrarily_small_positive_margin_has_positive_excess(self):
        for denominator in [2, 10, 1000, 10**12]:
            mu = F(1, denominator)
            point = protected_optimum(mu)
            self.assertEqual(margins(point)["protected_C_minus_D"], mu)
            self.assertGreater(l1_from_initial(point), 4)
            self.assertEqual(l1_from_initial(point) - 4, F(10, 3 * denominator))

    def test_independent_epigraph_linear_program(self):
        # Variables x,y,p,q; p>=|x-2|, q>=|y-1|. No closed-form
        # expression is used to construct the objective or constraints.
        from scipy.optimize import linprog
        for mu in [0, .1, 1, 1.6, 7]:
            result = linprog([0, 0, 1, 1], A_ub=[
                [1, -1, 0, 0], [-1, .4, 0, 0],
                [1, 0, -1, 0], [-1, 0, -1, 0],
                [0, 1, 0, -1], [0, -1, 0, -1]],
                b_ub=[-3, -mu, 2, -2, 1, -1],
                bounds=[(None, None), (None, None), (0, None), (0, None)], method="highs")
            self.assertEqual(result.status, 0)
            self.assertAlmostEqual(result.fun, float(l1_from_initial(protected_optimum(F(str(mu))))), places=10)

    def test_contract_rejects_inexact_or_out_of_scope_margin(self):
        for bad in [True, 1.0, "1", None]:
            with self.assertRaises(TypeError):
                protected_optimum(bad)
        with self.assertRaises(ValueError):
            protected_optimum(-1)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(GeometryChecks)
    result = unittest.TextTestRunner(stream=sys.stderr).run(suite)
    report = {"status": "PASS" if result.wasSuccessful() else "FAIL", "tests_run": result.testsRun,
              "passed": result.testsRun - len(result.failures) - len(result.errors),
              "failures": len(result.failures), "errors": len(result.errors),
              "scope": "Exact fixture checks and independent floating-point LP cross-checks, not formal proof verification"}
    print(json.dumps(report, indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
