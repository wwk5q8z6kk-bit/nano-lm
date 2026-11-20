"""Read-only audit of locked inputs and saved predictions. Never runs inference.

Run from repository root with PYTHONDONTWRITEBYTECODE=1 python3
artifacts/p1_retention_validation_20261026/independent_audit_recompute.py.
The scorer below independently implements the published metric definitions.
Product imports are used only to reparse saved text and check legacy transport.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

DATA = Path(__file__).resolve().parent
ROOT = DATA.parents[1]
sys.path.insert(0, str(ROOT))
from nano_ai.adapters.state_span import parse_state_span_summary
from nano_ai.contract import FieldOutput

LABELS = dict(zip(('CC', 'DUR', 'SEV', 'MED', 'ALG'),
                  ('chief_complaint', 'duration', 'severity', 'medication', 'allergy')))
STATES = dict(zip('SAMUC', ('supported', 'absent', 'missing', 'uncertain', 'conflicting')))
ASSERTED = {'supported', 'absent'}
METRICS = ('state_correct', 'normalized_content_correct', 'exact_evidence_aligned',
           'grounded_exact_correct', 'complete_evidence_exact_correct')


def read(name):
    return json.loads((DATA / name).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(text):
    punctuation = " \t\r\n.,;:!?\"'`\u2018\u2019"
    text = ' '.join(unicodedata.normalize('NFKC', text).casefold().split()).strip(punctuation)
    for article in ('a ', 'an ', 'the '):
        if text.startswith(article):
            return text[len(article):].strip(punctuation)
    return text


def annotate(records, sources):
    result = {}
    for record in records:
        text = sources[record['case_id']]['transcript']
        fields = []
        for label, field in LABELS.items():
            annotation = record['fields'][label]
            evidence = []
            for span in annotation['spans']:
                if isinstance(span, str):
                    positions = [i for i in range(len(text)) if text.startswith(span, i)]
                    assert len(positions) == 1
                    value, start = span, positions[0]
                else:
                    value, start = span['text'], span['start']
                end = start + len(value)
                assert text[start:end] == value
                line_start = text.rfind('\n', 0, start) + 1
                assert text[line_start:start].startswith('Patient: ')
                assert '\n' not in value
                evidence.append(dict(start=start, end=end, text=value, speaker='patient'))
            state = STATES[annotation['state']]
            fields.append(dict(field=field, state=state,
                               value=norm(evidence[0]['text']) if state == 'supported' else None,
                               evidence=evidence))
        result[record['case_id']] = fields
    return result


def score(pred, gold):
    if pred is None:
        return dict.fromkeys(METRICS, False)
    same = pred['state'] == gold['state']
    content = same and (gold['state'] != 'supported' or norm(pred['value']) == norm(gold['value']))
    key = lambda f: {(x['start'], x['end'], x['text'], x['speaker']) for x in f['evidence']}
    actual, expected = key(pred), key(gold)
    if gold['state'] in ('missing', 'uncertain'):
        aligned = same
    elif gold['state'] == 'conflicting':
        aligned = same and expected.issubset(actual)
    else:
        aligned = same and bool(expected.intersection(actual))
    return dict(zip(METRICS, (same, content, aligned, content and aligned,
                              content and actual == expected)))


def note_fields(draft):
    if not draft.get('note'):
        return [None] * 5
    evidence = {e['evidence_id']: e for e in draft['encounter']['evidence']}
    return [dict(field=c['field'], state=c['state'], value=c['value'], evidence=[
        {k: evidence[ref][k] for k in ('start', 'end', 'text', 'speaker')}
        for ref in c['evidence_ids']]) for c in draft['note']['claims']]


def totals(encounters, gold):
    counts = {path: Counter() for path in ('raw', 'checked', 'final_note')}
    for row in encounters:
        outputs = {'raw': row['raw_fields'] or [None] * 5,
                   'checked': (row['draft'].get('extraction') or {}).get('fields', [None] * 5),
                   'final_note': note_fields(row['draft'])}
        for path, predictions in outputs.items():
            for pred, expected in zip(predictions, gold[row['case_id']], strict=True):
                result = score(pred, expected)
                counts[path].update(result)
                counts[path]['fields'] += 1
                asserted = pred is not None and pred['state'] in ASSERTED
                counts[path]['emitted_asserted_fields'] += asserted
                counts[path]['asserted_state_or_content_mismatch'] += asserted and not result['normalized_content_correct']
                counts[path]['asserted_exact_field_mismatch'] += asserted and not result['grounded_exact_correct']
                counts[path]['correct_asserted_fields'] += expected['state'] in ASSERTED and result['grounded_exact_correct']
    return counts


def main():
    sources = {row['case_id']: row for row in read('sources.json')}
    annotations = {name: annotate(read(file), sources) for name, file in (
        ('adjudicated', 'gold.json'), ('original_a', 'annotations_a.json'), ('original_b', 'annotations_b.json'))}
    gold = annotations['adjudicated']
    locked = read('annotation_lock.json')['files']
    for name, sha in locked.items():
        assert digest(DATA / name) == sha, name
    code = read('implementation_lock.json')['files']
    for name, sha in code.items():
        assert digest(ROOT / name) == sha, name
    for mode in ('free', 'constrained', 'preflight'):
        provenance = read(mode + '/provenance.json')
        assert provenance['code_hashes'] == code
        for name, sha in provenance['code_hashes'].items():
            assert digest(DATA / mode / 'source_snapshot' / name) == sha, (mode, name)
        for name, sha in provenance['input_hashes'].items():
            assert digest(DATA / mode / 'input_snapshot' / name) == sha, (mode, name)
    state_counts = Counter(f['state'] for fs in gold.values() for f in fs)
    transport = []
    for cid, fields in gold.items():
        for field in fields:
            try:
                FieldOutput.from_dict(field)
            except ValueError as exc:
                transport.append(dict(case_id=cid, field=field['field'], error=str(exc)))
    assert transport == read('preflight/controls.json')['contract_transport_errors']
    sensitivity = read('annotation_sensitivity.json')['variants']
    local_saved = read('posthoc_diagnostics.json')['modes']
    summary = {'verified_source_files': len(code), 'verified_locked_inputs': len(locked),
               'cases': len(gold), 'gold_states': dict(state_counts),
               'gold_transport_errors': len(transport),
               'gold_transport_error_cases': len({e['case_id'] for e in transport}), 'modes': {}}
    for mode in ('free', 'constrained'):
        encounters = read(mode + '/encounters.json')
        strict_errors = Counter()
        for row in encounters:
            try:
                parsed = parse_state_span_summary(row['generation']['raw'], sources[row['case_id']]['transcript'])
                fields = [dict(field=p.field.value, state=p.state.value,
                               value=norm(p.spans[0].text) if p.state.value == 'supported' else None,
                               evidence=[s.to_dict() for s in p.spans]) for p in parsed]
                assert fields == row['raw_fields'] and row['raw_parse_error'] is None
            except ValueError as exc:
                assert row['raw_fields'] is None and str(exc) == row['raw_parse_error']
                strict_errors[str(exc)] += 1
        for variant, labels in annotations.items():
            values = totals(encounters, labels)
            expected = sensitivity[variant][mode]['comparison']
            for path in ('raw', 'checked'):
                for metric, value in values[path].items():
                    assert value == expected['metrics'][path][metric], (variant, mode, path, metric)
            for metric in ('emitted_asserted_fields', 'asserted_state_or_content_mismatch', 'asserted_exact_field_mismatch'):
                assert values['final_note'][metric] == expected['final_note'][metric]
            assert values['final_note']['correct_asserted_fields'] == expected['final_note']['correct_asserted_fields_emitted']
        primary = totals(encounters, gold)
        comparison = read(mode + '/comparison.json')
        assert comparison == sensitivity['adjudicated'][mode]['comparison']
        local = Counter()
        encounter_by_id = {r['case_id']: r for r in encounters}
        for row in local_saved[mode]['fields']:
            encounter = encounter_by_id[row['case_id']]
            traces = {f['field']: f for f in (encounter['draft'].get('inference_diagnostics') or {}).get('fields', [])}
            trace = traces.get(row['field'])
            recovered = None
            if trace and trace['parse_error'] is None and trace['raw_state'] is not None:
                recovered = dict(field=row['field'], state=trace['raw_state'], evidence=trace['raw_spans'],
                                 value=norm(trace['raw_spans'][0]['text']) if trace['raw_state'] == 'supported' else None)
            assert recovered == row['local_raw_prediction']
            checked = {f['field']: f for f in (encounter['draft'].get('extraction') or {}).get('fields', [])}
            assert score(checked.get(row['field']), row['gold']) == row['checked_score']
            result = score(row['local_raw_prediction'], row['gold'])
            assert result == row['local_raw_score']
            local['admitted_fields'] += row['local_raw_prediction'] is not None
            local.update(result)
            if row['gold']['state'] in ASSERTED:
                for suffix, metric in (('exact', 'grounded_exact_correct'), ('normalized', 'normalized_content_correct')):
                    local['correct_assertions_' + suffix] += result[metric]
                    local['correct_assertions_' + suffix + '_lost'] += result[metric] and not row['checked_score'][metric]
        for suffix in ('exact', 'normalized'):
            assert local['correct_assertions_' + suffix] == local_saved[mode]['counts']['local_raw_correct_assertions_' + suffix]
            assert local['correct_assertions_' + suffix + '_lost'] == local_saved[mode]['counts']['correct_assertions_' + suffix + '_lost_by_checking']
        assert local['admitted_fields'] == local_saved[mode]['counts']['locally_parse_admitted_raw_fields']
        summary['modes'][mode] = dict(primary=primary, strict_errors=strict_errors, local=local)
    menus = read('coverage_controls.json')['menus']
    concrete = [m for m in menus['details'] if any(c.startswith((':S[', ':A[', ':C[')) for c in m['choices'])]
    assert len(concrete) == menus['with_concrete_choices'] == 4
    summary['concrete_menus'] = concrete
    menu_matches = Counter()
    for menu in concrete:
        label = menu['field']
        field = LABELS[label]
        expected = next(f for f in gold[menu['case_id']] if f['field'] == field)
        outcomes = []
        for choice in menu['choices']:
            if not choice.startswith((':S[', ':A[')):
                continue
            response = '|'.join(k + (choice if k == label else ':M') for k in LABELS)
            proposals = parse_state_span_summary(response, sources[menu['case_id']]['transcript'])
            proposal = next(p for p in proposals if p.field.value == field)
            prediction = dict(field=field, state=proposal.state.value,
                              value=norm(proposal.spans[0].text) if proposal.state.value == 'supported' else None,
                              evidence=[s.to_dict() for s in proposal.spans])
            outcomes.append(score(prediction, expected))
        menu_matches['slots_with_normalized_correct_assertion_choice'] += any(s['normalized_content_correct'] for s in outcomes)
        menu_matches['slots_with_exact_correct_assertion_choice'] += any(s['grounded_exact_correct'] for s in outcomes)
    summary['concrete_menu_correct_assertion_availability'] = menu_matches
    for control_name, state, expected in (
            ('primary_oracle', None, read('preflight/controls.json')['primary_oracle']),
            ('primary_all_missing', 'missing', read('preflight/controls.json')['primary_all_missing']),
            ('all_u_control', 'uncertain', read('coverage_controls.json')['all_u_control'])):
        scores = Counter()
        for fields in gold.values():
            for f in fields:
                pred = f if state is None else dict(field=f['field'], state=state, value=None, evidence=[])
                scores.update(score(pred, f))
        assert all(v == expected['metrics']['raw'][k] for k, v in scores.items())
        summary[control_name] = scores
    a, b = annotations['original_a'], annotations['original_b']
    summary['annotation_agreement'] = dict(
        state=sum(x['state'] == y['state'] for cid in a for x, y in zip(a[cid], b[cid])),
        evidence=sum(x['evidence'] == y['evidence'] for cid in a for x, y in zip(a[cid], b[cid])))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
