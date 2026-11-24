"""Independent small cases for the bilinear margin argument; no LM claim."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
from margin_durability import margin_drift, trajectory_margin_drift


class MarginTests(unittest.TestCase):
    def test_joint_drift_decomposition_known_values(self):
        r = margin_drift([[2.,0.],[0.,0.]], [1.,1.], [[3.,2.],[0.,0.]], [2.,3.], 0)
        self.assertEqual(r['margins_before'], [2.])
        self.assertEqual(r['margins_after'], [12.])
        self.assertEqual(r['signed_terms_state_decoder_interaction'], [[2.,3.,5.]])
        self.assertEqual(r['decomposition_max_abs_residual'], 0.)

    def test_small_joint_update_has_positive_bound(self):
        r = margin_drift([[2.,0.],[-1.,0.]], [1.,0.], [[2.,.01],[-1.,-.01]], [1.,.01], 0)
        self.assertTrue(r['positive_lower_bound'])
        self.assertTrue(r['actual_strict_target_winner_after'])

    def test_large_margin_alone_does_not_bound_future_drift(self):
        r = margin_drift([[10.],[-10.]], [1.], [[-10.],[10.]], [1.], 0)
        self.assertEqual(r['margins_before'], [20.])
        self.assertEqual(r['margins_after'], [-20.])
        self.assertFalse(r['positive_lower_bound'])

    def test_nonzero_orthogonal_update_needs_no_rehearsal(self):
        r = margin_drift([[1.,0.],[-1.,0.]], [1.,0.], [[1.,0.],[-1.,0.]], [1.,100.], 0)
        self.assertEqual(r['margins_after'], [2.])
        self.assertTrue(r['actual_strict_target_winner_after'])
        self.assertFalse(r['positive_lower_bound'])

    def test_coherent_transport_can_make_norm_bound_inconclusive(self):
        r = margin_drift([[1.,0.],[-1.,0.]], [1.,0.], [[.01,0.],[-.01,0.]], [100.,0.], 0)
        self.assertEqual(r['margins_after'], [2.])
        self.assertFalse(r['positive_lower_bound'])
        self.assertTrue(r['actual_strict_target_winner_after'])

    def test_telescoping_bound_applies_to_every_supplied_prefix(self):
        states = [(np.array([[2.,x],[-1.,-x]]), np.array([1.,x])) for x in [0.,.01,.02,.03]]
        r = trajectory_margin_drift(states, 0)
        self.assertTrue(r['positive_lower_bounds_through_horizon'])
        self.assertTrue(np.all(np.asarray(r['actual_margins']) >= np.asarray(r['margin_lower_bounds']) - 1e-12))

    def test_random_joint_updates_obey_bound(self):
        rng = np.random.default_rng(417)
        for _ in range(30):
            w, u = rng.normal(size=(5,4)), rng.normal(size=4)
            r = margin_drift(w, u, w+rng.normal(size=w.shape), u+rng.normal(size=u.shape), 2)
            change = np.abs(np.array(r['margins_after'])-r['margins_before'])
            self.assertTrue(np.all(change <= np.array(r['total_bound']) + 1e-10))
            self.assertLess(r['decomposition_max_abs_residual'], 1e-12)

    def test_ties_and_invalid_inputs(self):
        r = margin_drift([[1.],[1.]], [1.], [[1.],[1.]], [1.], 0)
        self.assertFalse(r['positive_lower_bound'])
        for w,u,target in [([[1.],[0.]],[float('nan')],0), ([[1.],[0.]],[1.],True), ([[1.],[0.]],[1.],2)]:
            with self.assertRaises(ValueError): margin_drift(w,u,w,u,target)
        with self.assertRaises(ValueError): trajectory_margin_drift([],0)

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'frozen.json'; p.write_bytes(b'previous result')
            script = Path(__file__).resolve().parents[1]/'analysis/margin_durability.py'
            result = subprocess.run([sys.executable,'-B',str(script),'--out',str(p)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(p.read_bytes(),b'previous result')


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MarginTests)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    print(json.dumps({'suite':'bilinear_margin_durability','tests_run':result.testsRun,
                      'passed':result.testsRun-len(result.failures)-len(result.errors),
                      'failures':len(result.failures),'errors':len(result.errors),
                      'scope':'Known-answer cases, counterexamples and finite numerical examples; no interval verification or neural training.'},indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
