"""Collect local diagnostic receipts; no training or writes without --out NEW.

Records file identity and command results, not remote authenticity or the
scientific meaning of a passing check.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def snapshot():
    return {
        p.relative_to(ROOT).as_posix(): digest(p.read_bytes())
        for p in sorted(ROOT.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ledger', type=Path)
    parser.add_argument('--baseline-package', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.out is not None and args.out.exists():
        raise FileExistsError(f'Existing receipt is preserved: {args.out}')
    started = datetime.now(timezone.utc).isoformat()
    before = snapshot()
    issues = []
    commands = [
        ('package_regressions', ['checks/test_research_package.py']),
        ('mathematical_examples', ['checks/repo_lightweight_checks.py']),
        ('numpy_examples', ['checks/formal_sanity.py']),
        ('transport_anchors', ['checks/transport_anchor_pilot.py']),
        ('native_forward_reference', ['prototype/test_native_reference.py']),
        ('repository_reanalysis', ['analysis/reconcile_repository_evidence.py']),
        ('package_validator', ['checks/validate_research_package.py']),
    ]
    if args.baseline_ledger is not None:
        commands[-1][1].extend(['--baseline-ledger', str(args.baseline_ledger.resolve())])
    receipts = {}
    for name, arguments in commands:
        began = time.perf_counter()
        command = [sys.executable, '-B', *arguments]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                timeout=60,
                                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        receipt = {'command': command, 'returncode': result.returncode,
                   'elapsed_seconds_diagnostic_only': round(time.perf_counter() - began, 4),
                   'stderr': result.stderr}
        try:
            receipt['output'] = json.loads(result.stdout)
        except json.JSONDecodeError:
            receipt['stdout'] = result.stdout
            issues.append(f'{name}: stdout was not a JSON receipt')
        if result.returncode != 0:
            issues.append(f'{name}: nonzero return code {result.returncode}')
        receipts[name] = receipt
    after = snapshot()
    unchanged = before == after
    if not unchanged:
        issues.append('Default commands changed package files')
    for name, field, expected in [
        ('package_regressions', 'passed', 35),
        ('mathematical_examples', 'passed', 12),
        ('numpy_examples', 'n_passed', 5),
        ('native_forward_reference', 'passed', 13),
        ('package_validator', 'status', 'PASS'),
    ]:
        if receipts[name].get('output', {}).get(field) != expected:
            issues.append(f'{name}: expected {field}={expected!r}')
    comparisons = {}
    for name, path, subkey in [
        ('transport_anchors', 'results/transport_anchor_pilot_v0.json', 'summary'),
        ('native_forward_reference', 'results/native_reference_v3_final.json', None),
        ('repository_reanalysis', 'results/repository_reanalysis_v3.json', None),
    ]:
        expected = json.loads((ROOT / path).read_text())
        if subkey is not None:
            expected = expected[subkey]
        equal = receipts[name].get('output') == expected
        comparisons[name] = {'artifact': path, 'equal_to_saved_result': equal}
        if not equal:
            issues.append(f'{name}: current output differs from preserved result')
    source_manifest = json.loads((ROOT / 'evidence/source_manifest.json').read_text())
    evidence = []
    for entry in source_manifest['files']:
        data = (ROOT / entry['local_path']).read_bytes()
        git_blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        matched = (digest(data) == entry['sha256'] and git_blob == entry['git_blob_sha']
                   and len(data) == entry['bytes'])
        evidence.append({'path': entry['local_path'], 'sha256': digest(data),
                         'git_blob_sha': git_blob, 'matches_source_manifest': matched})
        if not matched:
            issues.append(f"Upstream copy mismatch: {entry['local_path']}")
    historical = {'status': 'NOT_CHECKED'}
    if args.baseline_package is not None:
        historical = {}
        for name in ['formal_sanity_20261121.json', 'transport_anchor_pilot_v0.json']:
            local = ROOT / 'results' / name
            baseline = args.baseline_package.resolve() / 'results' / name
            matched = local.read_bytes() == baseline.read_bytes()
            historical[name] = {'byte_identical': matched, 'sha256': digest(local.read_bytes())}
            if not matched:
                issues.append(f'Historical result rewritten: {name}')
    baseline = None
    if args.baseline_ledger is not None:
        data = args.baseline_ledger.read_bytes()
        baseline = {'sha256': digest(data), 'bytes': len(data),
                    'current_log_starts_with_baseline': (ROOT / 'ledger_events.jsonl').read_bytes().startswith(data)}
        if not baseline['current_log_starts_with_baseline']:
            issues.append('Previous full ledger prefix was changed')
    report = {
        'status': 'PASS' if not issues else 'FAIL', 'errors': issues,
        'scope': 'Local deterministic checks and artifact identities; no neural training, formal verification, benchmark replication or remote CI.',
        'started_at_utc': started,
        'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'python_version': platform.python_version(),
        'dependencies': {name: importlib.metadata.version(name) for name in ['numpy', 'jsonschema']},
        'inspected_prior_pr_head': '3f9ef1a1b5643acda9ee4cc2a780d4fdcbf4bce8',
        'inspected_research_base': source_manifest['source_commit'],
        'commands': receipts,
        'saved_result_comparisons': comparisons,
        'upstream_copies': evidence,
        'historical_result_preservation': historical,
        'baseline_ledger': baseline,
        'default_commands_left_package_unchanged': unchanged,
        'before_after_paths_changed': sorted(set(before) ^ set(after)),
        'before_after_bytes_changed': sorted(p for p in set(before) & set(after) if before[p] != after[p]),
        'source_hashes_before_receipt_creation': before,
        'source_hash_scope': 'All package files except Python bytecode; this newly created receipt is intentionally excluded from its own hashes.',
    }
    if args.out is not None:
        with args.out.open('x', encoding='utf-8') as target:
            json.dump(report, target, indent=2)
            target.write('\n')
    print(json.dumps(report, indent=2))
    return 0 if not issues else 1


if __name__ == '__main__':
    sys.exit(main())
