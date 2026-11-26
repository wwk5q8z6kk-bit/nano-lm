"""Recompute paired scores and transparent constant-state controls, no inference."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.run_p1_independent_dev import admit,compare,field_score

root=Path(__file__).resolve().parent
_,cases,_=admit(root)
controls={state:sum(field_score(dict(field=g['field'],state=state,value=None,evidence=[]),g)['grounded_exact_correct'] for c in cases for g in c.fields) for state in ['missing','uncertain']}
summary={'cases':len(cases),'fields':5*len(cases),'constant_state_grounded_exact_fields':controls,'runs':{}}
for name in ['constrained-02','free-01']:
 path=root/name
 if not (path/'comparison.json').exists():continue
 encounters=json.loads((path/'encounters.json').read_text())
 recomputed,_=compare(cases,encounters)
 stored=json.loads((path/'comparison.json').read_text())
 assert recomputed==stored,name
 summary['runs'][name]={'comparison':stored,'raw_contract_transport_errors':sum(bool(e['raw_contract_transport_errors']) for e in encounters),'checked_fields':{state:sum(f['state']==state for e in encounters for f in (e['draft'].get('extraction') or {}).get('fields',[])) for state in ['supported','absent','missing','uncertain','conflicting']},'dispositions':{state:sum(e['draft'].get('disposition')==state for e in encounters) for state in ['review','abstain']},'failures':sum(e['draft']['status']=='failed' for e in encounters)}
(root/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
