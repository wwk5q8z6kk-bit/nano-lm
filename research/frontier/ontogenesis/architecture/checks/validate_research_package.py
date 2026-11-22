"""Check Nano LM architecture research package integrity; no training or file mutation.
This checks structural/documentary consistency, not whether scientific claims are true.
"""
import json
import re
from pathlib import Path
from datetime import datetime
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
DOCS=("README.md","01_ARCHITECTURE_AND_THEOREMS.md","02_DECISION_LEDGER.md",
      "03_EXPERIMENTS_AND_EXECUTION.md","04_EVIDENCE_AND_REFERENCES.md",
      "05_GAP_CLOSURE_AND_FORMAL_REVIEW.md","06_VALIDATION_AND_UNRESOLVED_WORK.md")
REQUIRED=DOCS+("ledger_events.jsonl","ledger_event.schema.json","architecture_scheme.mmd")
LINK=re.compile(r'!?\[[^\]]*\]\(([^)]+)\)')
ALLOWED={"proposed","reviewed","mathematically_checked","preregistered",
         "exploratory","confirmed","rejected","superseded","integrated"}
ETYPE={"mathematical","repository","peer_reviewed","preprint","hypothesis","engineering"}

def validate():
    issues=[]
    for name in REQUIRED:
        if not (ROOT/name).is_file():issues.append("missing "+name)
    for name in DOCS:
        text=(ROOT/name).read_text(encoding="utf8")
        if text.count("```")%2:issues.append(name+" unbalanced code fences")
        for uri in LINK.findall(text):
            target=unquote(uri.split("#",1)[0].strip())
            if not target or target.startswith(("http://","https://","mailto:","data:","sandbox:")):continue
            if not (ROOT/target).exists():issues.append(name+" broken link "+target)
    ledger=(ROOT/"02_DECISION_LEDGER.md").read_text(encoding="utf8")
    a=ledger.split("## A. One-page decision register",1)[1].split("## B. Contradiction",1)[0]
    old_ids=re.findall(r'^\| (D-\d{3}) \|',a,flags=re.M)
    if old_ids!=[f"D-{i:03d}" for i in range(1,24)]:
        issues.append("Decision rows D-001 through D-023 not contiguous and unique")
    seen_dec=set();seen_events=set()
    schema=json.loads((ROOT/"ledger_event.schema.json").read_text())
    for n,line in enumerate((ROOT/"ledger_events.jsonl").read_text().splitlines(),1):
        if not line.strip():continue
        try:e=json.loads(line)
        except json.JSONDecodeError:issues.append(f"bad JSON event at line {n}");continue
        missing=set(schema["required"])-set(e)
        if missing:issues.append(f"event {n} missing {missing}")
        d=e.get("decision_id");eid=e.get("event_id")
        if d in seen_dec or eid in seen_events:issues.append(f"duplicate event or decision line {n}")
        seen_dec.add(d);seen_events.add(eid)
        if e.get("status") not in ALLOWED or e.get("evidence_type") not in ETYPE:
            issues.append(f"invalid event status/type line {n}")
        if not e.get("alternatives_considered") or not e.get("rationale") or not e.get("next_gate"):
            issues.append(f"event missing rationale/alternative/next gate line {n}")
        for file in e.get("source_refs",[]):
            if not file.startswith(("https://","http://")) and not (ROOT/file).exists():
                issues.append(f"event unresolved source {file}")
        try:datetime.fromisoformat(e["recorded_at_utc"].replace("Z","+00:00"))
        except (ValueError,KeyError):issues.append(f"bad timestamp line {n}")
    if seen_dec!=set(f"D-{i:03d}" for i in range(24,34)):
        issues.append("machine events must contain D-024 through D-033")
    for d in seen_dec:
        if not re.search(r'^\| '+re.escape(d)+r' \|',ledger,re.M):
            issues.append(f"missing human ledger {d}")
    return {"status":"PASS" if not issues else "FAIL","errors":issues,
            "events_checked":len(seen_events),"documents_checked":len(DOCS),
            "scope":"structural evidence contract, not independent scientific verification"}

if __name__=="__main__":
    report=validate()
    print(json.dumps(report,indent=2))
    if report["errors"]:raise SystemExit(1)
