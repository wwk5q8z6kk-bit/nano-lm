# Active Frontier — Development Plan

**Branch:** `frontier/active-v1`  
**Mandate:** `frontier/ACTIVE_MANDATE.md`  
**Architecture map:** `frontier/ARCHITECTURE_EVOLUTION.md`  
**Proving ground:** `wedge_v1/` (Wedge A)  
**Rule:** Ship architectural value from failures; never silently rewrite Evidence Core.

---

## Phase map

| Phase | Goal | Status |
|-------|------|--------|
| **P0** Discovery + mandate | Choose wedge A; kill criteria | **DONE** |
| **P1** Verified Ask vertical slice | ask/find/scan/compare/report/ingest | **DONE** |
| **P1b** Architecture feedback loop | gallery → workstream → delta | **ACTIVE** |
| **P2** Owner-corpus loop | Private folder eval + failure gallery | **DONE** (harness; real OWNER_CORPUS pending) |
| **P3** Contact / usefulness / habit | Review labels + habit session + fine gallery | **ACTIVE** |
| **P4** Ingest SLA | OCR/PDF normalize measured recover_gap | **DONE** |
| **P5** Marginal model value | Classical vs +small vs +large under U | After irreducible abstain |
| **P6** Distribution | Minimal UI / install | After retention signal |

---

## Now (architecture, not polish)

Failure-driven registry live: `arch-registry` · traces on `ask` · `adversarial` suite.
`LM_PROBE = NOT_INDICATED`. Real `$OWNER_CORPUS` usefulness remains external-pending.


P3 contact loop: `review`, `habit` session, fine `gallery`, `owner-ready`.
Real `$OWNER_CORPUS` usefulness = **external pending dependency**.


1. **W1** BM25 retrieval-margin gating in `ask()`
2. **W2** Evidence-atom hard gate
3. `python -m wedge_v1 evolve` — recommend next workstream from galleries
4. Re-run dogfood + smoke; watch `low_margin_review` / empty-span rates

## Next

5. General multi-doc epistemic merge (typed fields, both spans)
6. Pluggable synonym/OCR/coref modules (no fixture doc-id control flow)
7. Real `$OWNER_CORPUS` gallery → feed W1–W4

## Later

8. Small-LM only on `over_abstain` buckets with ΔU > δ
9. Dense retrieve if BM25 margin saturates

## Never

- NanoScribe OS · OLD_TASK_U gen default · Evidence Core edits · unpaid→paid GPU without owner · governance-doc churn as substitute for architecture
