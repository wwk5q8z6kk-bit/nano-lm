"""Rescore saved outputs only; no inference, annotation edits, or product tuning."""
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.run_p1_independent_dev import admit, compare, _field_annotation, LABELS

DATA = Path(__file__).resolve().parent


def read(name):
    return json.loads((DATA / name).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    _, cases, hashes = admit(DATA)
    gold = read('gold.json')
    annotations_a = read('annotations_a.json')
    annotations_b = read('annotations_b.json')
    variants = {'adjudicated': gold, 'original_a': annotations_a, 'original_b': annotations_b}
    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'method': 'Prespecified annotation sensitivity; same captured predictions, unchanged scorer. No new inference.',
              'input_hashes': hashes, 'analysis_sha256': digest(Path(__file__)),
              'variants': {}, 'run_hashes': {}}
    for mode in ('free', 'constrained'):
        run = DATA / mode
        provenance = read(f'{mode}/provenance.json')
        assert provenance['status'] == 'complete'
        encounters = read(f'{mode}/encounters.json')
        result['run_hashes'][mode] = {name: digest(run / name) for name in (
            'provenance.json', 'encounters.json', 'comparison.json', 'paired_fields.json')}
        for variant, records in variants.items():
            by_id = {r['case_id']: r['fields'] for r in records}
            variant_cases = tuple(replace(c, fields=tuple(
                _field_annotation(label, by_id[c.case_id][label], c.request.transcript)
                for label in LABELS), fixture=None) for c in cases)
            score, pairs = compare(variant_cases, encounters)
            if variant == 'adjudicated':
                assert score == read(f'{mode}/comparison.json')
            normalized = Counter()
            for pair in pairs:
                if pair['gold']['state'] not in ('supported', 'absent'):
                    continue
                raw, checked = pair['raw']['normalized_content_correct'], pair['checked']['normalized_content_correct']
                normalized['retained'] += raw and checked
                normalized['lost'] += raw and not checked
                normalized['salvaged'] += not raw and checked
            result['variants'].setdefault(variant, {})[mode] = {
                'comparison': score, 'normalized_correct_assertion_transitions': dict(normalized)}
    output = DATA / 'annotation_sensitivity.json'
    assert not output.exists(), 'Do not overwrite an existing analysis'
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    for variant, modes in result['variants'].items():
        for mode, metrics in modes.items():
            comparison = metrics['comparison']
            print(variant, mode, {p: {
                k: comparison['metrics'][p][k] for k in (
                    'state_correct', 'normalized_content_correct', 'grounded_exact_correct',
                    'correct_asserted_fields', 'asserted_state_or_content_mismatch')
            } for p in ('raw', 'checked')})


if __name__ == '__main__':
    main()
