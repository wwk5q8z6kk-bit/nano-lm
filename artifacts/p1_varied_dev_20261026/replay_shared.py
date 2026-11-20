"""Development integration replay only; no generation or new model scores."""
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.run_p1_independent_dev import ROOT, admit, collect, compare, source_snapshot

root = Path(__file__).resolve().parent
output = root / 'shared-replay-01'
output.mkdir(exist_ok=False)
hashes = source_snapshot(output)
rows, cases, _ = admit(root)
saved = json.loads((root/'constrained-02/encounters.json').read_text())
raw = {row['transcript']: encounter['generation']['raw'] for row, encounter in zip(rows, saved, strict=True)}
encounters = collect(cases, lambda transcript: raw[transcript], output, version='saved-generation-integration-replay')
comparison, details = compare(cases, encounters)
assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == sha for path, sha in hashes.items())
(output/'comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
(output/'paired_fields.json').write_text(json.dumps(details, indent=2)+'\n')
(output/'provenance.json').write_text(json.dumps({'status':'complete', 'new_model_inference':False,'source_run':'constrained-02','code_hashes':hashes},indent=2)+'\n')
print(comparison['metrics']['checked']['grounded_exact_correct'])
