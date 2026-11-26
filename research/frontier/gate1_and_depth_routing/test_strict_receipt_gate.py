import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from strict_receipt_gate import (SCHEMA, SEEDS, PARAMS, ORDERS, RECIPE, STEPS, METRICS,
                                 audit, load_receipt, validate_record, snapshot)


def fixture(seed):
    arms = {name: {'params': size, 'skipped_amp_steps': 0,
                   'curve': [dict({k: 1.0 for k in METRICS}, step=s,
                                  tokens=s*64*512, train_sec=float(s)) for s in STEPS]}
            for name, size in PARAMS.items()}
    return {'schema': SCHEMA, 'seed': seed, 'completed': True, 'gpu': 'Tesla T4',
            'recipe': RECIPE.copy(), 'arm_order': ORDERS[seed].copy(), 'arms': arms}


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.paths = [Path(self.tmp.name) / f'seed_{n}.json' for n in SEEDS]
        for p,n in zip(self.paths, SEEDS): p.write_text(json.dumps(fixture(n)))

    def test_valid_snapshot_and_reorder(self):
        data, hashes = snapshot(self.paths[::-1])
        self.assertEqual(set(data),set(SEEDS)); self.assertEqual(len(hashes),3)
        self.assertEqual(snapshot(self.paths,hashes)[1],hashes)

    def test_partial_duplicate_or_wrong_seed(self):
        with self.assertRaises(ValueError): snapshot(self.paths[:2])
        with self.assertRaises(ValueError): snapshot([self.paths[0],self.paths[0],self.paths[2]])
        d=fixture(1000); d['seed']=1003
        with self.assertRaises(ValueError): validate_record(d)

    def test_bool_seed_and_nonfinite_duplicate_json(self):
        d=fixture(1000); d['seed']=True
        with self.assertRaises(ValueError): validate_record(d)
        for data in (b'{"x":NaN}',b'{"x":Infinity}',b'{"x":1,"x":2}'):
            with self.assertRaises(ValueError): load_receipt(data)

    def test_incomplete_recipe_order_or_arm(self):
        for key, value in [('completed',False),('recipe',{}),('arm_order',ORDERS[1001]),('arms',{})]:
            d=fixture(1000); d[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): validate_record(d)

    def test_truncated_steps_and_interior_token_count(self):
        for change in ('short','tokens','steps','time','nan','params','skip','bool'):
            d=fixture(1000); a=d['arms']['dense']; c=a['curve']
            if change=='short': c.pop()
            elif change=='tokens': c[1]['tokens']+=1
            elif change=='steps': c[1]['step']=500
            elif change=='time': c[1]['train_sec']=c[0]['train_sec']
            elif change=='nan': c[1]['mix']=float('nan')
            elif change=='params': a['params']+=1
            elif change=='skip': a['skipped_amp_steps']=1
            elif change=='bool': c[1]['mix']=True
            with self.subTest(change=change),self.assertRaises(ValueError): validate_record(d)

    def test_manifest_mismatch(self):
        _,hashes=snapshot(self.paths); hashes['seed_1000.json']='0'*64
        with self.assertRaisesRegex(ValueError,'manifest mismatch'): snapshot(self.paths,hashes)

    def test_scoring_uses_snapshots_not_mutable_originals(self):
        def fake_score(copies):
            self.paths[0].write_text('{}')  # original changes after validation
            hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in copies}
            self.assertEqual(validate_record(load_receipt(copies[0].read_bytes())),1000)
            return {'provenance':{'receipt_sha256':hashes,'validated_seeds':list(SEEDS)},'verdict':'fixture only'}
        out=audit(self.paths,scorer=fake_score)
        self.assertEqual(out['disposition'],'REVIEW_ONLY_NO_ADOPTION')
        self.assertEqual(out['training_provenance'],'NOT_ATTESTED')

    def test_wrong_scorer_provenance_rejected(self):
        with self.assertRaisesRegex(ValueError,'provenance'):
            audit(self.paths,scorer=lambda _: {'provenance':{'receipt_sha256':{}}})


if __name__ == '__main__': unittest.main()
