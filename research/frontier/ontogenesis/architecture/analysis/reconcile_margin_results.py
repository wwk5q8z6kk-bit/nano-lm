"""Read-only audit of pinned margin summaries and exogenous event generation.

No Torch import, learned weights or neural training. The LP counterexample is
constructed, not a reconstruction of any historical model prediction.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def constants(path):
    namespace = {}
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Tuple)
               and {x.id for x in target.elts if isinstance(x, ast.Name)} == {'Z', 'R', 'V', 'EPS', 'K'}
               for target in node.targets):
            # Execute only the literal constants assignment, never module imports.
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return {key: namespace[key] for key in ['Z', 'R', 'V', 'K']}


def event_stream(seed, c, saved_immediate):
    phases, new, fix = 40, 64, 16
    n = phases * new
    rng = np.random.default_rng(seed)
    rule = rng.integers(c['V'], size=(c['Z'], c['R']))
    typ = rng.integers(c['Z'], size=n)
    truth = rule[typ].copy()
    shown = np.zeros((n, c['R']), bool)
    for entity in range(n):
        shown[entity, rng.choice(c['R'], c['K'], replace=False)] = True
    corrected = np.zeros((n, c['R']), bool)
    when = np.full((n, c['R']), -1)
    manifest, collisions, seen = [], [], set()
    noops = repeats = 0
    for phase in range(1, phases):
        old_entities, old_relations = np.nonzero(shown[:phase * new])
        selected = rng.choice(len(old_entities), fix, replace=False)
        entities, relations = old_entities[selected], old_relations[selected]
        prior = truth[entities, relations].copy()
        replacement = rng.integers(c['V'], size=fix)
        noops += int(np.sum(prior == replacement))
        truth[entities, relations] = replacement
        corrected[entities, relations] = True
        when[entities, relations] = phase
        counts = Counter(map(int, entities))
        nonterminal = [index for index, entity in enumerate(entities)
                       if int(entity) in set(map(int, entities[index + 1:]))]
        if nonterminal:
            collisions.append({
                'phase': phase,
                'repeated_entities': {entity: count for entity, count in counts.items() if count > 1},
                'nonterminal_event_indices': nonterminal,
                'events': [(int(e), int(r), int(y)) for e, r, y in zip(entities, relations, replacement)],
            })
        for e, r, y in zip(entities, relations, replacement):
            key = (int(e), int(r))
            repeats += int(key in seen)
            seen.add(key)
            manifest.append({'phase': phase, 'entity': key[0], 'relation': key[1], 'value': int(y)})
    e, r = np.nonzero(shown)
    age = (phases - 1) - np.repeat(np.arange(phases), new)[e]
    regular = ~corrected[e, r]
    hit = corrected.any(1)[e]
    cold = corrected[e, r] & ((phases - 1) - when[e, r] >= 20)
    error_candidates = [errors for errors in range(len(manifest) + 1)
                        if round(1 - errors / len(manifest), 4) == saved_immediate]
    return {
        'seed': seed,
        'correction_events': len(manifest),
        'correction_phases': phases - 1,
        'phases_with_same_entity_multi_relation_writes': len(collisions),
        'nonterminal_same_entity_write_events': sum(len(item['nonterminal_event_indices']) for item in collisions),
        'same_value_assignment_events': noops,
        'repeat_key_correction_events': repeats,
        'unique_corrected_keys': len(seen),
        'final_old_correction_denominator': int(cold.sum()),
        'final_old_regular_corrected_entity_denominator': int((regular & (age >= 20) & hit).sum()),
        'phase_end_error_counts_consistent_with_rounded_none_m1': error_candidates,
        'stream_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'collision_details': collisions,
    }


def counterexample(source):
    node = next(node for node in ast.parse(source.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == 'edit')
    namespace = {'np': np, 'linprog': linprog, 'R': 2}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    # Class-1 logit differences are u0 and -2*u0+u1 respectively.
    weights = np.array([[[0., 0.], [1., 0.]], [[0., 0.], [-2., 1.]]])
    initial = np.zeros(2)
    first_delta, first_ok = namespace['edit'](weights, initial, 0, 1, 1.0, 'none')
    first = initial + first_delta
    second_delta, second_ok = namespace['edit'](weights, first, 1, 1, 1.0, 'none')
    second = first + second_delta
    joint = np.array([1., 3.])
    result = {
        'scope': 'Constructed example using the extracted minimum-L1 edit function; does not identify historical error causes.',
        'first_solver_success': bool(first_ok),
        'second_solver_success': bool(second_ok),
        'state_after_first_write': first.tolist(),
        'state_after_second_write': second.tolist(),
        'first_target_correct_after_first_write': int((weights[0] @ first).argmax()) == 1,
        'first_target_correct_after_second_write': int((weights[0] @ second).argmax()) == 1,
        'second_target_correct_after_second_write': int((weights[1] @ second).argmax()) == 1,
        'jointly_feasible_state': joint.tolist(),
    }
    assert first_ok and second_ok
    assert result['first_target_correct_after_first_write']
    assert not result['first_target_correct_after_second_write']
    assert result['second_target_correct_after_second_write']
    assert ((weights @ joint).argmax(-1) == 1).all()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path(__file__).parent / 'research/frontier/ontogenesis')
    parser.add_argument('--prior-results', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    source_dir = args.source_dir
    c = constants(source_dir / 'cab_s.py')
    arms, raw = {}, {}
    for path in sorted((source_dir / 'results/correction_interference').glob('margins-*.json')):
        data = json.loads(path.read_text())
        arm = data[0]['arm']
        raw[arm] = data
        arms[arm] = {
            'seed_count': len(data),
            'source_sha256': sha256(path),
            'metrics': {
                key: {'mean': float(np.mean([x[key] for x in data])),
                      'min': min(x[key] for x in data), 'max': max(x[key] for x in data)}
                for key in data[0] if key not in ['arm', 'seed']
            },
        }
    assert set(raw) == {'half_m1', 'half_m3', 'half_m6', 'keep_m3', 'none_m1'}
    stream_rows = [event_stream(row['seed'], c, row['corrected_now']) for row in raw['none_m1']]
    prior_match = None
    if args.prior_results:
        old = sorted(json.loads(args.prior_results.read_text()), key=lambda x: x['seed'])
        new = sorted(raw['half_m1'], key=lambda x: x['seed'])
        common = ['corrected_now', 'collateral_now', 'rule_old_corrected', 'corrected_old']
        prior_match = all(a['seed'] == b['seed']
                          and all(a[key] == b[key] for key in common)
                          and a['lp_feasible'] == b['feasible']
                          for a, b in zip(old, new, strict=True))
    report = {
        'source_commit': '9d88a2c20233f42b4a49c4430264b20f842895ad',
        'scope': 'Committed rounded-summary arithmetic, exogenous stream reconstruction and a constructed LP example. No neural training or recovered model predictions.',
        'source_hashes': {name: sha256(source_dir / name) for name in ['cab_s.py', 'correction_margins.py']},
        'numpy_version': np.__version__,
        'constants': {**c, 'phases': 40, 'new_entities_per_phase': 64, 'corrections_per_noninitial_phase': 16},
        'arms': arms,
        'half_m1_matches_prior_four_common_rounded_metrics_and_success_flags': prior_match,
        'reproduction_match_scope': 'Rounded summaries only, not weights, trajectories, residuals or independent fresh seeds.',
        'exogenous_event_diagnostics': stream_rows,
        'within_phase_counterexample': counterexample(source_dir / 'correction_margins.py'),
        'causal_limit': 'Collision counts can accommodate the observed phase-end errors but cannot establish which historical errors were caused by collisions or numerical residuals.',
    }
    if args.out is not None:
        with args.out.open('x') as target:
            json.dump(report, target, indent=2)
            target.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
