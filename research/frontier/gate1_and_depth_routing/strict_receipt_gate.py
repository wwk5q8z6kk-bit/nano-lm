"""Fail-closed custody gate for the fixed, registered 14M tournament only.

This is a prospective wrapper around audit_and_strict_score.py, not a change
of the historical scorer/protocol/JSON. Validate a byte snapshot before scoring
it. A receipt hash identifies the reported result, not the original training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

SCHEMA = 'nanolm.h13_h10_muon_tournament.v1'
SEEDS = (1000, 1001, 1002)
PARAMS = {'dense': 14292224, 'value_res': 14292231, 'block_attnres': 14300928,
          'coda_refine': 14292480, 'dense10': 15768064}
ORDERS = {1000: ['dense','value_res','block_attnres','coda_refine','dense10'],
          1001: ['block_attnres','coda_refine','dense10','dense','value_res'],
          1002: ['coda_refine','dense10','dense','value_res','block_attnres']}
RECIPE = {'opt': 'OptaxParityMuon', 'lr': .004, 'lr_floor': .01, 'wd': .1,
          'warmup': 300, 'bs': 64, 'seq': 512, 'micro': 4, 'steps': 3000,
          'tokens_per_arm': 98304000}
STEPS = list(range(500, 3001, 500))
METRICS = ('mix','web','textbook','math','code','train_sec','tokens_per_second',
           'train_loss','grad_norm','peak_gpu_gb')


def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f'duplicate JSON key: {key}')
        out[key] = value
    return out


def _bad_constant(value):
    raise ValueError(f'nonfinite JSON constant: {value}')


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


def _number(v):
    return type(v) in (int, float) and math.isfinite(v)


def load_receipt(data: bytes):
    return json.loads(data, object_pairs_hook=_unique_object, parse_constant=_bad_constant)


def validate_record(record):
    _require(isinstance(record, dict), 'receipt must be an object')
    seed = record.get('seed')
    _require(type(seed) is int and seed in SEEDS, 'seed outside registered set')
    _require(record.get('schema') == SCHEMA and record.get('completed') is True,
             'wrong schema or incomplete receipt')
    recipe = record.get('recipe')
    _require(isinstance(recipe, dict), 'missing recipe')
    for name, expected in RECIPE.items():
        _require(type(recipe.get(name)) is type(expected) and recipe[name] == expected,
                 f'seed {seed}: recipe mismatch: {name}')
    _require(record.get('gpu') == 'Tesla T4', f'seed {seed}: unexpected device')
    _require(record.get('arm_order') == ORDERS[seed], f'seed {seed}: wrong arm order')
    arms = record.get('arms')
    _require(isinstance(arms, dict) and set(arms) == set(PARAMS), f'seed {seed}: wrong arms')
    for name, arm in arms.items():
        tag = f'seed {seed}/{name}'
        _require(isinstance(arm, dict) and type(arm.get('params')) is int
                 and arm['params'] == PARAMS[name], f'{tag}: parameter mismatch')
        skipped = arm.get('skipped_amp_steps')
        _require(type(skipped) is int and skipped == 0, f'{tag}: skipped/unknown AMP updates; review exposure')
        curve = arm.get('curve')
        _require(isinstance(curve, list) and len(curve) == len(STEPS), f'{tag}: incomplete curve')
        elapsed = 0.
        for step, row in zip(STEPS, curve):
            _require(isinstance(row, dict) and type(row.get('step')) is int and row['step'] == step,
                     f'{tag}: wrong evaluation step')
            _require(type(row.get('tokens')) is int and row['tokens'] == step * 64 * 512,
                     f'{tag}: wrong token count')
            for metric in METRICS:
                value = row.get(metric)
                _require(_number(value) and value >= 0, f'{tag}: invalid {metric}')
            _require(row['train_sec'] > elapsed and row['tokens_per_second'] > 0,
                     f'{tag}: non-monotone time or zero throughput')
            elapsed = row['train_sec']
    return seed


def snapshot(paths, expected_hashes=None):
    """Read once, reject duplicate/missing/unexpected seeds and hash mismatch."""
    if len(paths) != len(SEEDS):
        raise ValueError('exactly three registered seed files required')
    by_seed = {}
    digests = {}
    for path in map(Path, paths):
        data = path.read_bytes()
        seed = validate_record(load_receipt(data))
        if seed in by_seed:
            raise ValueError(f'duplicate seed {seed}')
        name = f'seed_{seed}.json'
        digest = hashlib.sha256(data).hexdigest()
        by_seed[seed] = data
        digests[name] = digest
    _require(set(by_seed) == set(SEEDS), 'missing registered seed')
    if expected_hashes is not None:
        _require(isinstance(expected_hashes, dict) and digests == expected_hashes,
                 'receipt SHA256 manifest mismatch')
    return by_seed, digests


def audit(paths, expected_hashes=None, scorer=None):
    """Use private byte copies so validation and scoring see the same records.

    `scorer` is injectable for unit tests only; CLI uses the existing additive
    strict scorer. No outputs overwrite any historical artifact.
    """
    records, hashes = snapshot(paths, expected_hashes)
    if scorer is None:
        from audit_and_strict_score import strict_score_files
        scorer = strict_score_files
    with tempfile.TemporaryDirectory(prefix='nanolm-receipts-') as directory:
        copies = []
        for seed in SEEDS:
            dest = Path(directory) / f'seed_{seed}.json'
            dest.write_bytes(records[seed])
            copies.append(dest)
        result = scorer(copies)
    _require(result.get('provenance', {}).get('receipt_sha256') == hashes,
             'scorer provenance differs from validated byte snapshot')
    _require(result.get('provenance', {}).get('validated_seeds') == list(SEEDS),
             'scorer seed set differs from validated snapshot')
    return {'schema': 'nanolm.receipt_gate.v2', 'validated_seeds': list(SEEDS),
            'receipt_sha256': hashes, 'expected_manifest_matched': expected_hashes is not None,
            'training_provenance': 'NOT_ATTESTED', 'disposition': 'REVIEW_ONLY_NO_ADOPTION',
            'scorer_report': result}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs=3, type=Path)
    parser.add_argument('--expected-receipt', type=Path,
                        help='Prior strict_audit_receipt.json to bind report bytes')
    args = parser.parse_args()
    expected = None
    if args.expected_receipt is not None:
        expected = load_receipt(args.expected_receipt.read_bytes())['provenance']['receipt_sha256']
    print(json.dumps(audit(args.files, expected), indent=2, allow_nan=False))
