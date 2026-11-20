"""Supplemental diagnostics from saved traces and locked code; no new inference.

Locally parse-admitted raw slots are separate from the prespecified strict
whole-record raw baseline. They are not corrected or filtered by gold labels.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.run_p1_independent_dev import admit, field_score, ASSERTED
from nano_ai.contract import normalize_value
from nanoscribe.check_note import inspect_context, transcript_source
from nanoscribe.context import question_field

DATA = Path(__file__).resolve().parent


def main():
    _, cases, hashes = admit(DATA)
    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'analysis_type': 'Prespecified supplemental local-slot diagnostics, separate from primary strict raw metrics.',
              'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'input_hashes': hashes, 'modes': {}, 'source_policy': {}}
    questions, unsafe = [], []
    for case in cases:
        source = transcript_source(case.request)
        if inspect_context(source)[1]:
            unsafe.append(case.case_id)
        for turn in source.turns:
            if turn.speaker.value == 'clinician':
                field = question_field(turn.text)
                questions.append({'case_id': case.case_id, 'text': turn.text,
                                  'recognized_field': field.value if field else None})
    result['source_policy'] = {'global_unsafe_case_ids': unsafe,
        'clinician_turns': len(questions),
        'recognized_singleton_requests': sum(q['recognized_field'] is not None for q in questions),
        'caution': 'Recognition is implementation coverage, not independently scored question-attribution accuracy; some clinician turns are not singleton requests.',
        'questions': questions}
    for mode in ('free', 'constrained'):
        run = DATA / mode
        encounters = {r['case_id']: r for r in json.loads((run / 'encounters.json').read_text())}
        count, reasons, decisions, traces = Counter(), Counter(), Counter(), []
        for case in cases:
            row = encounters[case.case_id]
            diag = row['draft'].get('inference_diagnostics') or {}
            fields = {f['field']: f for f in diag.get('fields', [])}
            checked = {f['field']: f for f in (row['draft'].get('extraction') or {}).get('fields', [])}
            count['strict_raw_failed_checked_recovered_cases'] += row['raw_parse_error'] is not None and row['draft'].get('extraction') is not None
            count['parse_admitted_recovery_slots'] += diag.get('recovered_field_count', 0)
            count['token_limit_reached_cases'] += bool(row['generation'].get('token_limit_reached'))
            for gold in case.fields:
                trace = fields.get(gold['field'])
                prediction = None
                if trace:
                    reasons[trace['reason']] += 1
                    decisions[trace['decision']] += 1
                    if trace['parse_error'] is None and trace['raw_state'] is not None:
                        evidence = trace['raw_spans']
                        prediction = {'field': gold['field'], 'state': trace['raw_state'],
                            'evidence': evidence, 'value': normalize_value(evidence[0]['text'])
                            if trace['raw_state'] == 'supported' else None}
                raw_score = field_score(prediction, gold)
                checked_score = field_score(checked.get(gold['field']), gold)
                count['fields'] += 1
                count['locally_parse_admitted_raw_fields'] += prediction is not None
                for key, value in raw_score.items():
                    count['local_raw_' + key] += value
                for kind, key in (('exact', 'grounded_exact_correct'), ('normalized', 'normalized_content_correct')):
                    if gold['state'] in ASSERTED:
                        count[f'local_raw_correct_assertions_{kind}'] += raw_score[key]
                        count[f'correct_assertions_{kind}_lost_by_checking'] += raw_score[key] and not checked_score[key]
                traces.append({'case_id': case.case_id, 'field': gold['field'], 'gold': gold,
                    'local_raw_prediction': prediction, 'local_raw_score': raw_score,
                    'checked_score': checked_score, 'check_reason': trace['reason'] if trace else None})
        result['modes'][mode] = {'counts': dict(count), 'checker_reasons': dict(reasons),
            'checker_decisions': dict(decisions), 'fields': traces}
    output = DATA / 'posthoc_diagnostics.json'
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print('source_policy', {k: v for k, v in result['source_policy'].items() if k != 'questions'})
    for mode, values in result['modes'].items():
        print(mode, values['counts'], values['checker_reasons'])


if __name__ == '__main__':
    main()
