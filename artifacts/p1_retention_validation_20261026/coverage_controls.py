"""Locked-policy coverage and all-U control; no model loading or tuning."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.run_p1_independent_dev import admit, compare, field_score
from nanoscribe.compact import LocalCompactPredictor
from nanoscribe.context import extract_context_fields

DATA = Path(__file__).resolve().parent


def main():
    _, cases, hashes = admit(DATA)
    lock = json.loads((DATA / 'implementation_lock.json').read_text())
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha
               for name, sha in lock['files'].items())
    all_u, menus, binding = [], [], []
    for case in cases:
        fields = [dict(field=f['field'], state='uncertain', value=None, evidence=[])
                  for f in case.fields]
        all_u.append(dict(case_id=case.case_id, raw_fields=fields, raw_parse_error=None,
                          draft=dict(extraction=dict(fields=fields), disposition='abstain')))
        source_fields = extract_context_fields(case.request)
        for predicted, gold in zip(source_fields, case.fields):
            binding.append(dict(case_id=case.case_id, field=gold['field'],
                                prediction=predicted.to_dict(), gold=gold,
                                **field_score(predicted.to_dict(), gold)))
        predictor = LocalCompactPredictor(str(DATA), decoding='constrained')
        def choose(messages, choices):
            menus.append(dict(case_id=case.case_id, field=messages[-1]['content'].split('|')[-1],
                              choices=choices,
                              concrete_choices=sum(c.startswith((':S[', ':A[', ':C[')) for c in choices)))
            return ':M', {}
        predictor._choose = choose
        predictor(case.request.transcript)
    control, _ = compare(cases, all_u)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), input_hashes=hashes,
                  explanation='Menu coverage uses M selections for all earlier slots. Actual model choices may consume spans and reduce later menus. This is policy availability, not a model score.',
                  all_u_control=control,
                  menus=dict(slots=len(menus), only_missing_or_empty_uncertain=sum(m['choices'] == [':M', ':U[]'] for m in menus),
                             with_concrete_choices=sum(m['concrete_choices'] > 0 for m in menus),
                             by_field={label: dict(total=sum(m['field']==label for m in menus),
                                 concrete=sum(m['field']==label and m['concrete_choices']>0 for m in menus))
                                 for label in ('CC','DUR','SEV','MED','ALG')}, details=menus),
                  binder=dict(state_counts=dict(Counter(b['prediction']['state'] for b in binding)),
                              grounded_exact_fields=sum(b['grounded_exact_correct'] for b in binding),
                              exact_assertions=sum(b['gold']['state'] in ('supported','absent') and b['grounded_exact_correct'] for b in binding),
                              details=binding))
    output = DATA / 'coverage_controls.json'
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: {k:v for k,v in result[key].items() if k!='details'} for key in ('menus','binder')}, indent=2))


if __name__ == '__main__':
    main()
