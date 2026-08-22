#!/usr/bin/env python3
"""Verify docs/ACTIVE_NOW.md and ACTIVE_NOW.json agree exactly."""
from __future__ import annotations
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MD, JSON = ROOT/"docs/ACTIVE_NOW.md", ROOT/"docs/ACTIVE_NOW.json"
SYNCED = ("program_execution_status","capability_frontier","current_gate","evidence_core","local_zero_cost_exploratory_training","paid_training","frozen_confirmatory_execution","phi_on_cloud","phi_or_private_data","clinical_claims")
REQ = SYNCED + ("schema","updated","owner_gates","integration_base_sha")
BAD = frozenset({"9fe5b6b6","9fe5b6b6f8746005a0f0608b5440a6138bdd458b"})

def parse_table(md: str) -> dict[str,str]:
    rows, sec = {}, False
    for line in md.splitlines():
        if line.startswith("## Status"): sec=True; continue
        if sec and line.startswith("## "): break
        if sec and (m:=re.match(r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|", line)):
            k,v=m.group(1).strip(),m.group(2).strip()
            if k in rows: raise ValueError(f"duplicate {k}")
            rows[k]=v
    if not rows: raise ValueError("no table")
    return rows

def main() -> int:
    data=json.loads(JSON.read_text()); table=parse_table(MD.read_text())
    errs=[f"mismatch {k}" for k in SYNCED if str(data.get(k))!=table.get(k)]
    errs+=[f"json missing {k}" for k in SYNCED if k not in data]
    errs+=[f"md missing {k}" for k in SYNCED if k not in table]
    if set(table)-set(SYNCED): errs.append(f"extra md keys {sorted(set(table)-set(SYNCED))}")
    for k in REQ:
        if k not in data: errs.append(f"json missing required {k}")
    if not isinstance(data.get("owner_gates"),list) or not data["owner_gates"]: errs.append("owner_gates empty")
    base=str(data.get("integration_base_sha",""))
    if not base or base in BAD: errs.append(f"bad integration_base_sha {base!r}")
    if data.get("local_zero_cost_exploratory_training")!="ALLOWED": errs.append("local must be ALLOWED")
    if data.get("paid_training")!="OWNER_GATED": errs.append("paid must be OWNER_GATED")
    if data.get("evidence_core")!="FROZEN_UNTOUCHED_BY_DOC_PR": errs.append("bad evidence_core")
    if errs: print("\n".join(errs), file=sys.stderr); return 1
    print("ACTIVE_NOW_OK"); return 0
if __name__=="__main__": raise SystemExit(main())
