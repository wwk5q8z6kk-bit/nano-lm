"""Read-only integrity verification; writes a new report beside this script."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPAIR = Path(__file__).resolve().parent
VALIDATION = ROOT / 'artifacts/p1_retention_validation_20261026'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def inspect(files, base):
    changed = []
    for name, expected in files.items():
        path = base / name
        if not path.is_file() or sha(path) != expected:
            changed.append(name)
    return {'files': len(files), 'mismatches': changed}

protected = json.loads((REPAIR / 'protected_before.json').read_text())
code = json.loads((VALIDATION / 'implementation_lock.json').read_text())['files']
annotations = json.loads((VALIDATION / 'annotation_lock.json').read_text())['files']
result = {
    'checked_utc': datetime.now(timezone.utc).isoformat(),
    'protected': inspect(protected, ROOT),
    'implementation': inspect(code, ROOT),
    'annotations': inspect(annotations, VALIDATION),
    'implementation_lock_copies_identical': sha(REPAIR / 'implementation_lock.json') == sha(VALIDATION / 'implementation_lock.json'),
    'runs': {},
}
for mode in ('free', 'constrained'):
    folder = VALIDATION / mode
    provenance = json.loads((folder / 'provenance.json').read_text())
    result['runs'][mode] = {
        'status': provenance['status'],
        'code_hashes_equal_implementation_lock': provenance['code_hashes'] == code,
        'source_snapshot': inspect(provenance['code_hashes'], folder / 'source_snapshot'),
        'current_inputs': inspect(provenance['input_hashes'], VALIDATION),
        'input_snapshot': inspect(provenance['input_hashes'], folder / 'input_snapshot'),
        'checkpoint': provenance['checkpoint'],
        'checkpoint_hashes': provenance['checkpoint_hashes'],
        'cases': len(json.loads((folder / 'encounters.json').read_text())),
    }
checkpoint = Path(result['runs']['free']['checkpoint'])
result['checkpoint_files'] = inspect(result['runs']['free']['checkpoint_hashes'], checkpoint)
result['same_checkpoint_both_runs'] = result['runs']['free']['checkpoint_hashes'] == result['runs']['constrained']['checkpoint_hashes']
checks = [result[k]['mismatches'] == [] for k in ('protected', 'implementation', 'annotations', 'checkpoint_files')]
checks += [result['implementation_lock_copies_identical'], result['same_checkpoint_both_runs']]
for mode, run in result['runs'].items():
    checks += [run['status'] == 'complete', run['code_hashes_equal_implementation_lock'], run['cases'] == 24]
    checks += [run[k]['mismatches'] == [] for k in ('source_snapshot', 'current_inputs', 'input_snapshot')]
result['passed'] = all(checks)
output = REPAIR / 'integrity.json'
with output.open('x') as f:
    f.write(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'runs'}, indent=2))
for mode, run in result['runs'].items():
    print(mode, {k:v for k,v in run.items() if k not in ('checkpoint_hashes','checkpoint')})
assert result['passed'], 'Integrity mismatch; inspect report'
