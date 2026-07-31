# Chain-of-Evidence Slice Report

**Date:** 2026-07-31  
**Branch:** `frontier/active-v1`  
**Invariant:** `EVIDENCE MUST BE CREATED WITH THE CLAIM, NOT RECONSTRUCTED AFTER THE CLAIM.`  
**Scope:** Nano Runtime local verified intelligence — **not** a Science One clone / paper agent.

---

## Schemas implemented

| Artifact | Path | Notes |
|----------|------|-------|
| `EvidenceAtom` | `wedge_v1/coe/schema.py` | doc_id, digest, offsets, text, relation typing |
| `TypedClaim` | same | claim_id, derivation, solver version, atoms, verification, reproducibility |
| `VerificationRecord` | same | independent flag + outcome |
| `DerivationKind` / `EvidenceRelation` | same | EXACT_QUOTE… / EXACTLY_STATED… |
| JSONL `EvidenceRecord` | `wedge_v1/coe/record.py` | append-safe run events (not claimed tamper-proof) |

Event types: `CORPUS_OPENED`, `QUERY_NORMALIZED`, `RETRIEVER_EXECUTED`, `SPAN_SELECTED`, `CLAIM_CONSTRUCTED`, `VERIFIER_EXECUTED`, `CONTRADICTION_FOUND`, `CLAIM_PRESENTED` / `CLAIM_REJECTED`, `RUN_*`.

---

## Invariants enforced

1. Presentable ask() claims receive `claim_id` + `coe` binding before return (render/report consumers see bound objects).
2. Completeness: PRESENT/CONFIRMED typed claims require evidence atoms (audit `claim_support`).
3. Correctness checks: offset validity against live corpus bytes; contradiction-not-ignored; classical-only spec.
4. Method–execution alignment: `solver_path` ↔ `trace.solvers`.
5. Citation faithfulness proxy: missing `claim_id` / invariant → `COE_POSTHOC_CITATION`.

---

## Trace example

```text
ask("How long before cache entries expire?")
→ coe.run_id = run_*
→ wedge_v1/.coe_runs/run_*.jsonl  (gitignored)
→ events include RUN_STARTED → CORPUS_OPENED → QUERY_NORMALIZED → …
→ CLAIM_CONSTRUCTED → SPAN_SELECTED → VERIFIER_EXECUTED → CLAIM_PRESENTED → RUN_FINALIZED
→ payload.claims[*].claim_id bound; payload.coe_claims typed
```

CLI:

```bash
python -m wedge_v1 coe-audit "How long before cache entries expire?"
python -m wedge_v1 coe-replay "How long before cache entries expire?"
```

---

## Audit checks

`evidence_existence` · `offset_validity` · `claim_support` · `trace_completeness` · `method_execution_alignment` · `spec_classical_only` · `citation_faithfulness_binding` · `contradiction_not_ignored`

---

## Adversarial tests (`wedge_v1/test_coe.py`)

| Case | Result |
|------|--------|
| Missing evidence PRESENT | audit fail / `COE_UNSUPPORTED_PREDICATE` |
| Invalid offsets | `offset_validity` fail |
| Post-hoc citation (no claim_id/invariant) | `COE_POSTHOC_CITATION` |
| Contradiction ignored (banner + SUPPORTED) | `COE_CONTRADICTION_IGNORED` |
| Compound query still CoE-bound | pass |
| Replay digest stable | pass |
| Overhead budget (<250ms avg ×5) | pass |

`COE_SLICE_OK` · `WEDGE_V1_SMOKE_OK`

---

## Overhead

- Latency: CoE bind+JSONL on synthetic ask typically **single-digit to low tens of ms** on this machine (budget assert <250ms avg).
- Storage: one JSONL run file under `wedge_v1/.coe_runs/` (gitignored); typically **few KB** per query.

---

## Failures found in existing code

1. **CLI import shadowing:** inner `from wedge_v1.runtime import DEFAULT_CORPUS` made `DEFAULT_CORPUS` local to `main()` → `UnboundLocalError` (fixed).
2. **Raw Claim path** can still construct empty-evidence PRESENT before verifier; CoE audit catches presentation without atoms.
3. **Composition gate domain coverage** is fixture-ish (ttl/dose/biblio only) — OOS conjuncts do not always yield `UNSUPPORTED_COMPOSITION` (gap → W3/W4 + `COE_INCOMPLETE_CONJUNCTION` enforcement).
4. Some atoms carry text without offsets — offset check relies on text∈doc when offsets absent.

---

## Fixes made

- `wedge_v1/coe/*` schema/record/bind/audit/replay
- `ask()` returns via `_finalize_with_coe`
- `COE_*` failure codes in `arch/failure_codes.py`
- CLI `coe-audit` / `coe-replay`
- Gallery surfaces `coe_run_id` / `coe_failure:*`
- Mandate invariant + W7 note

---

## Remaining CoE gaps

- Minimal-condition decomposition of compound propositions (partial support detection)
- Temporal source-version invalidation / claim-diff on corpus change
- Reversible user corrections as first-class events
- Counter-evidence search before PRESENT on high-risk claims
- Privacy redaction of raw text in persisted JSONL (currently local plaintext spans)
- Numeric recomputation audit independent of solver
- Full method–code alignment beyond solver_path string match

---

## Next highest-value architecture slice

**W3 + CoE minimal predicates:** typed multi-doc epistemic merge that emits one TypedClaim per atomic predicate, with `COE_INCOMPLETE_CONJUNCTION` when a user compound query is only partially evidenced — closing the composition-gate domain hole without LM.
