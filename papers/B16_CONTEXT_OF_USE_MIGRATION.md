# B16 — context_of_use migration (design proposal)

**MAY_AUTHORIZE_EXECUTION:** false  
**Needs:** `authorize commit` (or equivalent) before editing `EVIDENCE_LEDGER.json` rows.

## Schema (already noted in ledger)

```json
"context_of_use": {
  "task": "...",
  "utility": "...",
  "world": "...",
  "venue": "...",
  "does_not_imply": ["..."]
}
```

## Work when commit-authorized

1. Enumerate `GATE_VERDICT` rows missing `context_of_use`  
2. Fill from existing claim scope text (no epistemic upgrade)  
3. Add validator in `scripts/lint_claim_auth.py` or ledger validate script  
4. Split E1 decision vs cost claim IDs only under owner packaging auth (LAB.B18)

## Non-goals

- Do not upgrade PUBLIC_PARTIAL → FULL  
- Do not retarget freeze tags  
- Do not reopen E4
