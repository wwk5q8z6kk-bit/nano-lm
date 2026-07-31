# Clean-lineage freeze recipe (B17)

**Status:** M0/M1 design only. **Do not run** without `OWNER_TAG_OK` + tip policy `clean-lineage`.  
**Updated:** 2026-07-31T18:17Z  
**Companions:** `COUNCIL_HYBRID_CLOSEOUT.md`, `OWNER_SPEECH_ACTS.md`, `FIRST_PRINCIPLES_RISK_MITIGATION.md` §3.9

## Problem atom

`origin/master` ancestry includes E4 execute (`6af178d`). Naming that tip a **freeze** misrepresents stratified publication (P8).

## Admissible tip policies

| Policy | When |
|--------|------|
| `defer` | Default; log reason (current) |
| `clean-lineage` | This recipe |
| `non-freeze-snapshot` | Snapshot HEAD; name must not claim freeze |
| `verdict-annotation` | Additive `verdict/<claim>@<sha>` with ancestry disclosure |

## Recipe (clean-lineage)

```text
0. Preconditions
   - OWNER_TAG_OK: authorize_tag=true, tip_policy=clean-lineage
   - Protected tags unmoved:
       paper-alpha-v1
       post-alpha-evidence-freeze-2026-07-31
   - No E2/E4/fabric experiments in this session

1. branch freeze/reconciled-clean from post-alpha-evidence-freeze-2026-07-31

2. Cherry-pick ONLY freeze-hygiene commits (examples; verify each):
   - claim corrections / DIFF E ledger (e.g. 1fc8eea)
   - durable_raw packaging (e.g. ea001d4)
   - readiness / stratigraphy / EVIDENCE_CURRENT (when committed)
   - council hybrid closeout docs that do NOT depend on E4 RESULT as freeze-era
   REJECT any commit whose tree introduces E4 results as freeze evidence

3. Verify
   git merge-base --is-ancestor 6af178d HEAD  → must FAIL (exit 1)
   git rev-parse paper-alpha-v1^{commit} → unchanged
   git rev-parse post-alpha-evidence-freeze-2026-07-31^{commit} → unchanged
   shasum -a 256 -c artifacts/durable_raw/SHA256SUMS → 6/6

4. Annotated tag NEW distinct name at that tip
   (e.g. post-alpha-reconciled-evidence-freeze-YYYY-MM-DD)
   Message: residual-honest; disclose what was excluded (E4)

5. Push tag only if authorize_tag_push=true
```

## Explicit non-goals

- Do not retarget protected tags
- Do not fold E4 KILL into freeze brand
- Do not claim clinical readiness
