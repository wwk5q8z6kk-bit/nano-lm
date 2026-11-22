"""Recompute reported summaries and audit control-flow risks without fitting a model.

Frozen inputs are copies of upstream results/source, not reruns. The RNG probe
reproduces only the NumPy world/correction draws in correction_interference.py.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'evidence' / 'upstream'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def world_constants():
    """Read literal constants without importing the Torch training harness."""
    tree = ast.parse((UPSTREAM / 'cab_s.py.txt').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
            names = [x.id for x in node.targets[0].elts]
            if names == ['Z', 'R', 'V', 'EPS', 'K']:
                return dict(zip(names, ast.literal_eval(node.value)))
    raise ValueError('Upstream constant contract changed')


def correction_schedule(seed, random_replay, phases=40, new=64, fix=16):
    """Faithful draw-order diagnostic, with training and predictions omitted."""
    c = world_constants()
    Z, R, V, K = (c[k] for k in ('Z', 'R', 'V', 'K'))
    rng = np.random.default_rng(seed)
    n = phases * new
    rule = rng.integers(V, size=(Z, R))
    typ = rng.integers(Z, size=n)
    truth = rule[typ].copy()
    shown = np.zeros((n, R), bool)
    for e in range(n):
        shown[e, rng.choice(R, K, replace=False)] = True
    schedules = []
    for ph in range(phases):
        if not ph:
            continue
        oe, orr = np.nonzero(shown[:ph * new])
        pick = rng.choice(len(oe), fix, replace=False)
        fe, fr = oe[pick], orr[pick]
        truth[fe, fr] = rng.integers(V, size=fix)
        corr = [(int(e), int(r), int(truth[e, r])) for e, r in zip(fe, fr)]
        schedules.append(np.asarray(corr, dtype='<i8'))
        if random_replay:
            ce = sorted({e for e, _, _ in corr})
            k = sum(int(shown[e].sum()) for e in ce) - len(corr)
            if k > 0:
                oe2, or2 = np.nonzero(shown[:ph * new])
                ok = ~np.isin(oe2, ce)
                oe2, or2 = oe2[ok], or2[ok]
                rng.choice(len(oe2), k, replace=False)
    return schedules


def solver_status_audit():
    """Exercise the actual saved function with controlled solver statuses.

    This does not establish what happened in any historical solver invocation.
    It demonstrates that the old artifact cannot distinguish the failure kinds.
    """
    tree = ast.parse((UPSTREAM / 'correction_interference.py.txt').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'address_edit')
    module = ast.Module(body=[fn], type_ignores=[])
    c = world_constants()
    outcomes = {}
    for status in (2, 4):
        env = {'np': np, 'R': c['R'],
               'linprog': lambda *args, _status=status, **kwargs: SimpleNamespace(status=_status)}
        exec(compile(module, 'frozen_upstream_address_edit', 'exec'), env)
        delta, ok = env['address_edit'](np.zeros((c['R'], c['V'], 2)), np.zeros(2), 0, 1)
        outcomes[str(status)] = {'success': bool(ok), 'delta_is_none': delta is None}
    return {'returned_outcomes': outcomes, 'distinguishes_status_2_and_4': outcomes['2'] != outcomes['4'],
            'scope': 'Control-flow diagnostic only; actual historical solver statuses were not saved.'}


def report():
    records = {}
    seeds = [8001, 8002, 8003]
    for path in sorted((UPSTREAM / 'results' / 'correction_interference').glob('*.json')):
        rows = json.loads(path.read_text())
        if [r['seed'] for r in rows] != seeds or any(r['arm'] != path.stem for r in rows):
            raise ValueError('Unexpected correction result identity: ' + str(path))
        keys = [k for k in rows[0] if k not in ('arm', 'seed', 'in_store')]
        records[path.stem] = {
            'seed_count': len(rows),
            'means': {k: float(np.mean([r[k] for r in rows])) for k in keys},
            'ranges': {k: [min(r[k] for r in rows), max(r[k] for r in rows)] for k in keys},
            'source_sha256': digest(path.read_bytes()),
        }
    store_path = UPSTREAM / 'results' / 'lm' / 'store-pilot.json'
    store = json.loads(store_path.read_text())
    s = {r['arm']: r for r in store['rows']}
    probes = []
    for seed in seeds:
        base = correction_schedule(seed, False)
        replay = correction_schedule(seed, True)
        equal = [bool(np.array_equal(a, b)) for a, b in zip(base, replay)]
        probes.append({'seed': seed, 'correction_phases': len(equal),
                       'first_different_phase_zero_based': next((i + 1 for i, same in enumerate(equal) if not same), None),
                       'different_phases': sum(not x for x in equal),
                       'single_stream_sha256': digest(np.concatenate(base).tobytes()),
                       'random_replay_stream_sha256': digest(np.concatenate(replay).tobytes())})
    return {
        'scope': 'Reanalysis of pinned repository summaries and source diagnostics; no neural training rerun.',
        'upstream_commit': '2ef9ce5827cb8ea1fb6c8946df32fbcf5e06dec0',
        'correction_results': records,
        'store_pilot': {
            'seed_count_reported': 1, 'store_bytes': store['store_bytes'],
            'post8_minus_base8_bpb': s['post8']['mix'] - s['base8']['mix'],
            'joint8_minus_post8_bpb': s['joint8']['mix'] - s['post8']['mix'],
            'joint8_minus_base8_bpb': s['joint8']['mix'] - s['base8']['mix'],
            'post4_minus_base8_bpb': s['post4']['mix'] - s['base8']['mix'],
            'timing_valid_for_efficiency': False,
            'source_sha256': digest(store_path.read_bytes()),
        },
        'random_replay_rng_probe': probes,
        'solver_status_probe': solver_status_audit(),
        'not_available_in_raw_summaries': ['per-event paired durability trajectory', 'per-metric denominators',
                                           'all solver return statuses and primal residuals',
                                           'valid isolated wall time or energy', 'independent fresh-seed replication'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = report()
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.out:
        with args.out.open('x', encoding='utf-8') as f:
            f.write(text)
    print(text, end='')


if __name__ == '__main__':
    main()
