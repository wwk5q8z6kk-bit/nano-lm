#!/usr/bin/env python3
"""Capture completed H7 output without loading models or opening screen inputs.

Sources are read-only; destination is create-only. Failure leaves extracted
evidence and CAPTURE_FAILURE.json in place. Never retry an evaluation because
capture fails. This helper verifies recorded invocation evidence, not provider
execution history beyond the operator's canonical reservation.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile

ROOT = Path('/Users/mac/Projects/nano-lm')
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import run_h7_kaggle as op
import reconcile_p1_comparison as comparison

RUNNER_SHA256 = 'f0d7eed99f6b1eaf28e214c0e73924585ec91b92333379a460f2d4d53f8d8c86'
RUNTIME = {'python': '3.13.15', 'torch': '2.11.0+cpu', 'torch_cuda': None,
           'cuda_available': False, 'tokenizers': '0.22.2', 'pytest': '9.0.2'}
CONTRACT = {
    'schema_version': 'nano.h7-boundary-screen-evaluation.v1',
    'experiment_id': 'nano-p1-h7-boundary-margin-screen-v1', 'records': 1000,
    'screen_manifest_sha256': 'a766a41ed99bcffcd8e078c652c8a85fe6d796fe44268e03e0e309d9070cd6a6',
    'screen_records_sha256': '9b18c5eef105d0fa2ff0a1e575e764c0014a4ff62e40e8b8205dc12674c436f2',
    'control_checkpoint_sha256': '07547ec5c01b46fa87e8d53a40497a7c4b4a6bc92b596dddd67f4be4d18c89e0',
    'treatment_seeds': [20260805, 20260806],
}
RESULT_FILES = {'results/SHA256SUMS', 'results/h7_screen_evaluation.json',
                'results/h7_screen_evaluation.log'}
EVIDENCE_FILES = {'started.json', 'reservation.json', 'package-validation.json',
                  'RUN_H7_KAGGLE_OPERATIONAL.sh', 'runtime_probe.py', 'event_marker.py',
                  'runner.log', 'runtime.json', 'screen_evaluation.started.json', 'terminal.json'}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_json(payload: bytes):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    return json.loads(payload, object_pairs_hook=unique, parse_constant=invalid)


def read_json(path: Path):
    op.digest(path)  # Enforce a regular, nonsymlink source.
    return parse_json(path.read_bytes())


def sidecar_digest(path: Path) -> str:
    sha = op.digest(path)
    sidecar = path.with_name(path.name + '.sha256')
    op.digest(sidecar)
    require(sidecar.read_text().strip().split() == [sha, path.name],
            f'Archive sidecar mismatch: {path.name}')
    return sha


def extract_files(archive: Path, target: Path, *, results_only=False) -> set[str]:
    """Reject traversal, duplicates, links, specials, and unexpected roots."""
    target.mkdir(parents=True, exist_ok=False)
    files, seen = set(), set()
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        for member in members:
            name = member.name
            require(op.safe_name(name) and name not in seen, 'Unsafe or duplicate archive member')
            seen.add(name)
            if results_only:
                require(PurePosixPath(name).parts[0] == 'results', 'Unexpected result archive root')
            require(member.isfile() or (results_only and member.isdir()), 'Link or non-file archive member')
            if member.isfile():
                files.add(name)
        if results_only:
            require(files == RESULT_FILES, 'Unexpected completed result archive inventory')
            require(seen - files <= {'results'}, 'Unexpected result archive directories')
        for member in members:
            dest = target / member.name
            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, dest.open('xb') as output:
                    shutil.copyfileobj(source, output)
    return files


def verify_inventory(root: Path, checksum_name: str, files: set[str]) -> None:
    seen = set()
    for line in (root / checksum_name).read_text().splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  (.+)', line)
        require(match is not None, 'Malformed checksum line')
        sha, name = match.groups()
        require(op.safe_name(name) and name not in seen, 'Unsafe or duplicate checksum path')
        require(name in files and op.digest(root / name) == sha, f'Member digest mismatch: {name}')
        seen.add(name)
    require(seen == files - {checksum_name}, 'Incomplete checksum inventory')


def expected_wrapper(archive: Path) -> str:
    """Read only the manifest and shell-script members of the frozen package."""
    package = op.PACKAGES['evaluation']
    manifest_name = package['root'] + '/BUNDLE_MANIFEST.json'
    script_name = package['root'] + '/' + package['script']
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        require(sum(m.name == manifest_name for m in members) == 1 and
                sum(m.name == script_name for m in members) == 1, 'Ambiguous frozen package members')
        manifest_member, script_member = tar.getmember(manifest_name), tar.getmember(script_name)
        require(manifest_member.isfile() and script_member.isfile(), 'Non-file frozen script/manifest')
        manifest_bytes = tar.extractfile(manifest_member).read()
        require(hashlib.sha256(manifest_bytes).hexdigest() == package['manifest_sha256'],
                'Frozen package manifest changed')
        manifest = parse_json(manifest_bytes)
        script_bytes = tar.extractfile(script_member).read()
    record = manifest['members'][package['script']]
    require(len(script_bytes) == record['bytes'] and hashlib.sha256(script_bytes).hexdigest() == record['sha256'],
            'Frozen evaluation shell script changed')
    return op.derive_wrapper(script_bytes.decode(), 'evaluation')


def timestamp(value: str) -> datetime:
    return datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ')


def validate_recorded_rows(report: dict, package_path: Path) -> dict:
    """Recompute saved-output arithmetic only; never load models or screen inputs."""
    package = op.PACKAGES['evaluation']
    with tarfile.open(package_path, 'r:gz') as tar:
        manifest_bytes = tar.extractfile(package['root'] + '/BUNDLE_MANIFEST.json').read()
    require(hashlib.sha256(manifest_bytes).hexdigest() == package['manifest_sha256'],
            'Frozen manifest changed before recorded-row validation')
    manifest = parse_json(manifest_bytes)
    for relative in ('nano_ai/contract.py', 'nano_ai/training/diagnose_h6.py',
                     'nano_ai/training/evaluate_pointer.py', 'nano_ai/training/evaluate_evidence_query_h7.py'):
        require(op.digest(ROOT / relative) == manifest['members'][relative]['sha256'],
                f'Recorded-row arithmetic source differs from frozen package: {relative}')
    # Imports define models but perform no model construction, checkpoint reads,
    # screen access, inference, or training. Only the pure helpers below are used.
    from nano_ai.contract import FIELD_ORDER, FieldState
    from nano_ai.training import diagnose_h6 as taxonomy
    from nano_ai.training import evaluate_evidence_query_h7 as frozen
    from nano_ai.training.evaluate_pointer import _empty_raw_bucket, _finish_raw_bucket

    field_names = [name.value for name in FIELD_ORDER]
    states = {state.value for state in FieldState}
    arm_rows, counts = [], []
    require(len(report['arms']) == 3, 'Recorded diagnostics require three arms')
    for arm_index, arm in enumerate(report['arms']):
        raw = arm['raw']
        rows = raw['item_diagnostics']
        require(isinstance(rows, list) and len(rows) == CONTRACT['records'],
                'Recorded diagnostic item population differs from 1000')
        aggregate = _empty_raw_bucket()
        by_field = {name: _empty_raw_bucket() for name in field_names}
        aggregate['total'] = 5 * len(rows)
        for bucket in by_field.values():
            bucket['total'] = len(rows)
        identities, failures = set(), 0
        for index, row in enumerate(rows):
            example_id = row['example_id']
            require(isinstance(example_id, str) and example_id and example_id not in identities,
                    'Recorded item identity is empty or duplicated')
            identities.add(example_id)
            require(isinstance(row['variant'], str) and row['variant'] and row['target_field'] in field_names and
                    (row['target_state'] is None or row['target_state'] in states), 'Recorded item metadata invalid')
            fields = row['fields']
            if fields is None:
                require(row['decode_status'] == 'failure' and isinstance(row['decode_error'], str) and row['decode_error'],
                        'Recorded decode failure metadata invalid')
                failures += 1
                continue
            require(row['decode_status'] == 'decoded' and row['decode_error'] is None and
                    isinstance(fields, list) and [field['field'] for field in fields] == field_names,
                    'Recorded decoded field order/population invalid')
            for field in fields:
                role = f'arm[{arm_index}].row[{index}].{field["field"]}'
                require(field['raw_state'] in states and field['gold_state'] in states, 'Recorded field state invalid')
                taxonomy._classify_field(field, role=role)
                raw_spans = taxonomy._validated_spans(field['raw_spans'], role=role + '.raw_spans')
                gold_spans = taxonomy._validated_spans(field['gold_spans'], role=role + '.gold_spans')
                span_exact = {taxonomy._span_identity(s) for s in raw_spans} == {taxonomy._span_identity(s) for s in gold_spans}
                require(field['state_correct'] is (field['raw_state'] == field['gold_state']) and
                        field['span_exact'] is span_exact, 'Recorded state/span flags disagree with stored states/spans')
                presented = field['raw_state'] in {'supported', 'absent'}
                require(field['presented'] is presented and field['wrong_presented'] is (presented and not field['joint_exact']),
                        'Recorded presented flags disagree with stored state/joint flags')
                for bucket in (aggregate, by_field[field['field']]):
                    bucket['decoded'] += 1
                    for key in ('state_correct', 'span_exact', 'joint_exact', 'presented', 'wrong_presented'):
                        bucket[key] += int(field[key])
        require(aggregate['total'] == 5000 and aggregate['decoded'] + 5 * failures == 5000,
                'Recorded field population differs from 5000')
        require(raw['items'] == len(rows) and raw['decoded_items'] == len(rows) - failures and
                raw['decode_failure_items'] == failures and raw['decode_failure_rate'] == failures / len(rows),
                'Recorded raw item/decode counts disagree with rows')
        expected_fields = _finish_raw_bucket(aggregate)
        require(raw['fields'] == expected_fields and
                raw['by_field'] == {name: _finish_raw_bucket(bucket) for name, bucket in by_field.items()} and
                raw['wrong_presented_field_count'] == aggregate['wrong_presented'],
                'Recorded raw aggregate/per-field metrics disagree with rows')
        metrics = {'joint_exact_accuracy': expected_fields['joint_exact_accuracy'],
                   'state_accuracy': expected_fields['state_accuracy'], 'decode_failure_items': failures}
        require(arm['metrics'] == metrics, 'Recorded arm metrics disagree with rows')
        mechanism = frozen._mechanism_summary(raw)
        require(arm['mechanism'] == mechanism and mechanism['field_total'] == aggregate['decoded'],
                'Recorded mechanism summary disagrees with rows')
        arm_rows.append(rows)
        counts.append({'seed': arm['candidate']['seed'], 'items': len(rows), 'field_denominator': 5000,
                       'decoded_fields': aggregate['decoded'], 'decode_failure_items': failures})
    for aligned in zip(*arm_rows, strict=True):
        metadata = ('example_id', 'variant', 'target_field', 'target_state')
        require(all(tuple(row[key] for key in metadata) == tuple(aligned[0][key] for key in metadata) for row in aligned),
                'Recorded arm item identities/metadata differ')
        available = [row['fields'] for row in aligned if row['fields'] is not None]
        if available:
            gold = lambda fields: [(field['field'], field['gold_state'], field['gold_spans']) for field in fields]
            require(all(gold(fields) == gold(available[0]) for fields in available), 'Recorded gold diagnostics differ across arms')
    expected_comparisons = [frozen._comparison(report['arms'][0], arm) for arm in report['arms'][1:]]
    require(report['comparisons'] == expected_comparisons, 'Recorded shared-state comparisons disagree with rows')
    require(report['decision'] == frozen._decision(expected_comparisons), 'Recorded decision disagrees with rows')
    return {'arms': counts, 'aggregate_and_per_field_metrics_verified': True,
            'mechanism_and_shared_state_comparisons_verified': True, 'decision_verified': True,
            'limitation': 'Stored diagnostics only: no gold-truth or omitted normalized gold.value verification; by-gold-state, target-challenge and auxiliary acceptance summaries are not independently recomputed.'}


def validate(args) -> dict:
    require(op.digest(Path(op.__file__)) == RUNNER_SHA256, 'Frozen operational runner changed')
    archive_sha = sidecar_digest(args.archive)
    files = extract_files(args.archive, args.destination)
    verify_inventory(args.destination, 'ATTEMPT_SHA256SUMS', files)
    package = op.PACKAGES['evaluation']
    expected_files = {'ATTEMPT_SHA256SUMS', package['result'], package['result'] + '.sha256'}
    expected_files |= {'evidence/' + name for name in EVIDENCE_FILES}
    expected_files |= {package['root'] + '/' + name for name in RESULT_FILES}
    require(files == expected_files, 'Unexpected completed attempt archive inventory')
    evidence = args.destination / 'evidence'
    reservation = read_json(args.reservation)
    require(read_json(args.authority / 'evaluation.reservation.json') == reservation,
            'Evaluation reservation differs from canonical operator authority')
    require(op.validate_reservation(reservation) == 'evaluation', 'Wrong stage')
    require(reservation['canonical_authority'] == str(args.authority.resolve()), 'Wrong authority directory')
    require(read_json(evidence / 'reservation.json') == reservation, 'Captured reservation differs')
    training = read_json(args.training_capture)
    require(training['all_attempt_members_verified'] is True and training['all_treatment_members_verified'] is True,
            'Training capture did not validate all artifacts')
    require(training['terminal'] == reservation['training_terminal'], 'Training terminal differs from admission')
    require(training['treatment_archive_sha256'] == reservation['treatment_archive_sha256'],
            'Evaluation treatment archive differs from validated training output')
    training_reservation = read_json(args.authority / 'training.reservation.json')
    require(training['terminal']['attempt_id'] == training_reservation['attempt_id'],
            'Training attempt differs from canonical authority')
    require(training['terminal']['reservation_sha256'] == op.digest(args.authority / 'training.reservation.json'),
            'Training reservation hash differs from canonical authority')
    started, terminal = read_json(evidence / 'started.json'), read_json(evidence / 'terminal.json')
    require(set(started) == {'attempt_id', 'stage', 'started_at_utc', 'reservation_sha256'}, 'Unexpected start marker')
    require(started['attempt_id'] == reservation['attempt_id'] and started['stage'] == 'evaluation' and
            started['reservation_sha256'] == op.digest(args.reservation), 'Attempt identity mismatch')
    require(all(terminal.get(key) == value for key, value in started.items()), 'Terminal/start mismatch')
    require(terminal['status'] == 'completed' and terminal['phase'] == 'completed' and
            terminal['exit_code'] == 0 and terminal['error'] is None and terminal['retry_authorized'] is False,
            'Evaluation incomplete, failed, or interrupted; no retry authorized')
    require(type(terminal['elapsed_seconds']) in (int, float) and
            0 < terminal['elapsed_seconds'] <= reservation['max_runtime_seconds'], 'Evaluation runtime cap exceeded')
    total_runtime = terminal['elapsed_seconds'] + training['terminal']['elapsed_seconds']
    require(total_runtime <= op.MAX_SECONDS, 'Total campaign runtime cap exceeded')
    require(read_json(evidence / 'runtime.json') == RUNTIME, 'Runtime differs from pinned training/evaluation runtime')
    require(training['runtime'] == RUNTIME, 'Training capture runtime differs')
    event = read_json(evidence / 'screen_evaluation.started.json')
    require(set(event) == {'event', 'started_at_utc'} and event['event'] == 'screen_evaluation',
            'Wrong scientific invocation marker')
    require(timestamp(started['started_at_utc']) <= timestamp(event['started_at_utc']) <= timestamp(terminal['finished_at_utc']),
            'Scientific invocation timestamp outside attempt')
    provenance = read_json(evidence / 'package-validation.json')
    require(provenance['manifest_sha256'] == package['manifest_sha256'] and provenance['full_tree_verified'] is True and
            provenance['frozen_archive_sha256_provenance'] == package['archive_sha256'], 'Frozen input provenance mismatch')
    wrapper = (evidence / 'RUN_H7_KAGGLE_OPERATIONAL.sh').read_text()
    require(wrapper == expected_wrapper(args.evaluation_package), 'Captured operational shell differs from frozen derivation')
    require(sum(line.startswith('python -m nano_ai.training.') for line in wrapper.splitlines()) == 1 and
            wrapper.count('python "$H7_EVENT_MARKER" screen_evaluation\n') == 1, 'Wrong scientific invocation count')
    require((evidence / 'runtime_probe.py').read_text() == op.RUNTIME_PROBE and
            (evidence / 'event_marker.py').read_text() == op.EVENT_MARKER, 'Captured operational probes differ')
    result_archive = args.destination / package['result']
    result_archive_sha = sidecar_digest(result_archive)
    result_root = args.destination / 'verified-results'
    result_files = extract_files(result_archive, result_root, results_only=True)
    verify_inventory(result_root, 'results/SHA256SUMS', result_files)
    for name in result_files:
        require(op.digest(result_root / name) == op.digest(args.destination / package['root'] / name),
                'Nested result differs from separately preserved attempt result')
    result_path = result_root / 'results/h7_screen_evaluation.json'
    result_sha = op.digest(result_path)
    report = read_json(result_path)
    require(report['runtime'] == {'device': 'cpu', 'batch_size': 32, 'python': '3.13.15',
                                 'torch': '2.11.0+cpu', 'cuda': None, 'gpu': None}, 'Result runtime mismatch')
    candidates = training['candidates']
    require(len(candidates) == 2 and {c['seed'] for c in candidates} == set(CONTRACT['treatment_seeds']) and
            len({c['sha256'] for c in candidates}) == 2, 'Training capture candidates invalid')
    require(report['arms'][0]['candidate']['checkpoint_bytes'] == 13168916, 'Control checkpoint size mismatch')
    for arm in report['arms'][1:]:
        candidate = arm['candidate']
        matching = [c for c in candidates if c['seed'] == candidate['seed']]
        require(len(matching) == 1, 'Unexpected result candidate')
        prior = matching[0]
        for field, training_field in (('label', 'label'), ('epoch', 'epoch'), ('checkpoint_sha256', 'sha256'),
                                      ('checkpoint_bytes', 'artifact_bytes'), ('training_report_sha256', 'report_sha256')):
            require(candidate[field] == prior[training_field], f'Result candidate differs from admitted training: {field}')
    validated = comparison.h7_result(result_path, result_sha, CONTRACT)
    row_validation = validate_recorded_rows(report, args.evaluation_package)
    return {
        'schema': 'nano.h7-evaluation-capture-validation.v1',
        'attempt_archive_sha256': archive_sha, 'attempt_archive_bytes': args.archive.stat().st_size,
        'result_archive_sha256': result_archive_sha, 'result_archive_bytes': result_archive.stat().st_size,
        'result_path': str(result_path.resolve()), 'result_sha256': result_sha,
        'training_capture_sha256': op.digest(args.training_capture),
        'canonical_reservation_verified': True, 'all_attempt_members_verified': True,
        'all_result_members_verified': True, 'training_candidate_identities_verified': True,
        'recorded_evaluator_invocations': 1, 'runtime': RUNTIME, 'terminal': terminal,
        'total_training_and_evaluation_runtime_seconds': total_runtime,
        'stored_result_validation': validated,
        'recorded_row_validation': row_validation,
        'verification_limit': 'No model replay or screen input access. Invocation evidence is checked against the canonical reservation; independent provider execution history is not audited.',
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--training-capture', type=Path, required=True, help='Verified training CAPTURE_VALIDATION.json')
    parser.add_argument('--reservation', type=Path, default=ROOT / 'artifacts/runops/h7-20261022/evaluation.reservation.json')
    parser.add_argument('--authority', type=Path, default=op.AUTHORITY)
    parser.add_argument('--evaluation-package', type=Path, default=ROOT / 'artifacts/nano_h7/execution-package-v1/nano-h7-kaggle-evaluation.tar.gz')
    args = parser.parse_args(argv)
    require(not args.destination.exists() and not args.destination.is_symlink(), 'Capture destination must be new')
    try:
        summary = validate(args)
    except Exception as exc:
        failure = {'status': 'CAPTURE_VALIDATION_FAILED', 'error': f'{type(exc).__name__}: {exc}',
                   'retry_authorized': False, 'attempt_archive': str(args.archive)}
        if args.destination.is_dir():
            with (args.destination / 'CAPTURE_FAILURE.json').open('x') as handle:
                json.dump(failure, handle, indent=2, allow_nan=False)
                handle.write('\n')
        print(json.dumps(failure), file=sys.stderr)
        return 1
    with (args.destination / 'CAPTURE_VALIDATION.json').open('x') as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
