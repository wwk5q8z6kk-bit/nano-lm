"""Replay saved model proposals through the current checked path; no inference."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nanoscribe.compact import compact_solver
from nanoscribe.scribe import scribe

HERE = Path(__file__).parent
run = HERE / 'run-09'
rows = json.loads((HERE / 'dialogues.json').read_text())
records = json.loads((run / 'encounters.json').read_text())
provenance = json.loads((run / 'provenance.json').read_text())
assert len(rows) == len(records) == 20
lookup = {row['transcript']: saved['generation']['raw'] for row, saved in zip(rows, records, strict=True)}
solver = compact_solver(lookup.__getitem__, version=provenance['checkpoint_revision'], binder='context')
checks = []
for row, saved in zip(rows, records, strict=True):
    draft = scribe(row['transcript'], encounter_id=row['example_id'], solver=solver, context_policy='context')
    assert draft == saved['draft'], row['example_id']
    checks.append({'example_id': row['example_id'], 'identical_draft': True})
sources = ['nanoscribe/context_solver.py', 'nanoscribe/compact.py', 'nanoscribe/context.py',
           'nanoscribe/scribe.py', 'nanoscribe/check_note.py', 'nano_ai/adapters/state_span.py']
result = {'generation_source': 'run-09/encounters.json', 'new_model_inference': False,
          'incremental_paid_compute_usd': 0, 'checks': checks,
          'source_sha256': hashlib.sha256((run / 'encounters.json').read_bytes()).hexdigest(),
          'current_code_hashes': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}}
(HERE / 'current_path_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print('CURRENT_PATH_IDENTICAL', len(checks))
