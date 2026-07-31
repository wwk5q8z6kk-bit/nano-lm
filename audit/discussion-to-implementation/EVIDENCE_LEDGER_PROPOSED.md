# Evidence Ledger — Proposed Strengthened Form (DIFF E)

*Proposal only. **Do not** replace `papers/EVIDENCE_LEDGER.md` until owner approves this DIFF E.*

## Disposition

```text
DIFF_E = OWNER_APPROVED_APPLIED_TO_LIVE_LEDGER
# after minor schema/status remediation 2026-07-31-02
```

Machine-readable twin: `EVIDENCE_LEDGER_PROPOSED.json` (schema `nano-lm.evidence_ledger_proposed.v2`).
Enum validation: **PASS** (literal vocabulary members only).

## Review snapshot

```json
{
  "review_snapshot_id": "review-2026-07-31-02",
  "head_commit": "6f3a82362027d5e1bef22417f49a12ad8db90c30",
  "paper_alpha_tag": "0e01d73205e9c35ea32925fd4d6c7e5fceb61137",
  "premature_post_alpha_tag": "post-alpha-evidence-freeze-2026-07-31",
  "premature_post_alpha_tag_target": "a9d12cb1c456f6c465284e1d469c6326cb14d329",
  "note": "worktree_manifest_sha256 filled after writing ledger files",
  "worktree_manifest_sha256": "8b4d9b2941d09e84e156314bc782ee5612871e780545961a40f340aa922d820b",
  "worktree_manifest_sha256_excluding_json": "5e74907f9e5946a565bed0038b8deb7c7091532fbd49f445f9f4fac4942449f3",
  "proposed_json_sha256": "eb1b77da761214a2a3710e2f42b0d84f87c2bba2d3880a9015252b4aaa7f30bf"
}
```

Each row carries `last_reviewed_snapshot_id = review-2026-07-31-02`.

## Vocabularies (orthogonal; compound strings forbidden)

| Field | Allowed values |
|-------|----------------|
| epistemic | FALSIFIED \| PLAUSIBLE \| PROVEN \| SPECULATION \| SUPPORTED \| UNRESOLVED \| VOID |
| claim_type | FUTURE_HYPOTHESIS \| GATE_VERDICT \| IMPLEMENTATION_STATE \| INTERPRETATION \| MEASUREMENT \| POLICY_BAN \| PRODUCT_THESIS |
| gate_verdict | BLOCKED \| GATED_STOP \| KILL \| N/A \| PASS \| REFUTED \| UNRESOLVED \| VOID |
| evidence_class | ASPIRATIONAL \| DECISION-GOVERNANCE \| LOCAL-DOCUMENTARY \| PUBLIC-ANCHORED \| RAW-UNINSPECTED \| STALE-CONTRADICTORY |
| claim_record_publication / supporting_evidence_publication | ABSENT_EXPECTED \| ABSENT_UNEXPLAINED \| COMMITTED_LOCAL \| IGNORED_LOCAL \| PUBLIC_TAGGED \| PUBLIC_UNTAGGED \| RELEASE_ASSET_PLANNED \| UNTRACKED_LOCAL |
| result_state | ABSENT_EXPECTED \| ABSENT_UNEXPLAINED \| NO_RESULT \| PRESENT |
| reproducibility | LOCAL_UNPUBLISHED \| LOCAL_VERIFIED \| NO_RESULT \| PUBLIC_PARTIAL \| PUBLIC_REPRODUCIBLE \| RAW_NOT_DURABLE |
| wording_policy | APPROVED \| FORBIDDEN \| HEDGE_REQUIRED \| N/A |

Optional free-text (not enums): `gate_label`, `gate_note`, `wording_note`, `repro_note`.

Publication semantics:

- `claim_record_publication` — where **this ledger row text** is published (proposal remains `PUBLIC_UNTAGGED` until approved+pushed)
- `supporting_evidence_publication` — where **supporting artifacts** live
- `result_state` — whether a measurement RESULT exists

Invalidation vs future updates are separate columns.

## Claims

| Claim ID | Exact wording | Type | Epistemic | Gate | Gate label | Evid class | Claim pub | Evidence pub | Result | Repro | Wording | Limitations | Invalidation | Future update |
|----------|---------------|------|-----------|------|------------|------------|-----------|--------------|--------|-------|---------|-------------|--------------|---------------|
| C_GAP_EXISTS | On the synthetic scribe instrument (exact field-value match; multi-instance Stage T / anchors), small LM pipelines exhibit held-out vs seen exact-copy failures | MEASUREMENT | SUPPORTED | N/A | — | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_REPRODUCIBLE | APPROVED | construct ≠ clinic | protocol VOID / construct collapse for this estimand | — |
| C_FIELD_LOC | Under m0–m4 fieldwise scoring, open-vocabulary fields (cc/med/alg) carry the held/seen gap; closed-value fields (dur/sev) ≈ 0 | MEASUREMENT | SUPPORTED | N/A | — | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_REPRODUCIBLE | APPROVED | template-vs-value control only | closed-field nonzero under same protocol | — |
| C_DIVERSITY | On the preregistered 10M allergy-slot D5→D80 sweep (+ position control), mean held-type recall rose +66.7 points | MEASUREMENT | SUPPORTED | PASS | H_SLOT_SUPPORTED | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_REPRODUCIBLE | HEDGE_REQUIRED | one family/seed/slot; type identity changes with D | protocol VOID | replication across pools/seeds/families |
| C_SCALE_OBSERVED | Across tested own-stack configs with unequal pretraining schedules, diluted gaps were 18.3±1.3 (3.15M/32.8M tok), 18.7±1.5 (10M/~200M), 16.9±1.7 (159M/~200M full-FT) | MEASUREMENT | SUPPORTED | N/A | — | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_REPRODUCIBLE | APPROVED | unequal tok/param; sparse training-run replication | artifact corruption / recompute mismatch | — |
| C_PARAMETER_ONLY_EFFECT | Parameter count alone is insufficient to reduce the held-out copying gap | INTERPRETATION | UNRESOLVED | N/A | — | DECISION-GOVERNANCE | PUBLIC_UNTAGGED | PUBLIC_TAGGED | ABSENT_EXPECTED | NO_RESULT | HEDGE_REQUIRED | parameters not isolated from token budgets | — | matched equal-token parameter intervention + RESULT |
| C_OWNSTACK_200M_FULLFT_GATE | Under the frozen own-stack rule, 159M/200M/full-FT diluted gap 16.9≥14 → historical label STACK-dominant vs Pythia-160M | GATE_VERDICT | SUPPORTED | PASS | STACK_DOMINANT | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_REPRODUCIBLE | HEDGE_REQUIRED | historical gate label for one cell | protocol VOID | — |
| C_ADAPT_DATA_CELLS | At 159M own-stack, measured diluted gaps: ~16.9 (200M/full-FT), ~7.1 (200M/LoRA), ~7.0 (3.2B/full-FT), ~4.2 (3.2B/LoRA) | MEASUREMENT | SUPPORTED | N/A | — | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_REPRODUCIBLE | APPROVED | few runs/cell; venue mix | recompute mismatch | — |
| C_ADAPT_DATA_INTERP | The observed 2×2 cell pattern is consistent with a weak-base × full-FT interaction. Whether additional pretraining and LoRA are mechanistic substitutes is unresolved. | INTERPRETATION | PLAUSIBLE | N/A | — | DECISION-GOVERNANCE | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | NO_RESULT | HEDGE_REQUIRED | E2 unidentified | — | valid E2 RESULT after owner re-scope |
| C_INTERFERENCE | The preregistered isolated-versus-contained lexical-interference contrast produces an effect ≥ the C1b support threshold | GATE_VERDICT | FALSIFIED | REFUTED | C1B_ISOLATED_VS_CONTAINED | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_PARTIAL | APPROVED | general lexical/morph effects remain open; primary JSONL TRACKED | protocol VOID | new registered lexical contrast |
| C_C3_TB | Transition availability and boundary type each produce effects ≥ the preregistered 40-pt support threshold in C3 | GATE_VERDICT | FALSIFIED | REFUTED | C3_TRANSITION_BOUNDARY | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_PARTIAL | APPROVED | replication JSONL IGNORED_LOCAL + local_raw_archive; wide intervals/imbalance | protocol VOID | equivalence-powered redesign |
| C_C3_L | Length factor drives residual (≥ support threshold) in C3 | GATE_VERDICT | UNRESOLVED | UNRESOLVED | C3_LENGTH | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_PARTIAL | APPROVED | underpowered | protocol VOID | decisive length arm under amended prereg |
| C_MORPH_DESC | Morphological re-inflection was the largest post-hoc descriptive error category in the C3 census (~44% of core-cell misses under recorded classification) | MEASUREMENT | SUPPORTED | N/A | — | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_PARTIAL | HEDGE_REQUIRED | post-hoc classification | re-label mismatch under same census definition | — |
| C_MORPH | Morphological re-inflection is the causal residual mechanism | INTERPRETATION | SPECULATION | N/A | — | ASPIRATIONAL | PUBLIC_UNTAGGED | PUBLIC_TAGGED | ABSENT_EXPECTED | NO_RESULT | FORBIDDEN | exploratory only | — | preregistered causal RESULT |
| C_POINTER_P1 | Explicit pointer/copy head closes OOD gap (P1) | GATE_VERDICT | VOID | VOID | POINTER_P1 | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_REPRODUCIBLE | APPROVED | manipulation check failed | — | — |
| C_POINTER_P2 | Copy-supervised pointer head closes OOD gap for this preregistered implementation | GATE_VERDICT | FALSIFIED | REFUTED | POINTER_P2 | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_REPRODUCIBLE | APPROVED | this impl only | protocol VOID | new implementation gets a new claim ID |
| C_E1_MEASUREMENT | Under frozen E1 U, reported utilities: M1≈0.999, official M0≈0.925, M2≈0.886 (margin M1−M0≈+0.074); ρ = review load | MEASUREMENT | SUPPORTED | N/A | — | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_PARTIAL | APPROVED | M1 generator-aligned; L/C device-normalization audit residual | recompute mismatch / protocol VOID | none; new U receives a new claim ID |
| C_E1_GATE | Frozen E1 rule returns KILL (M1 exceeds M0; M2 within δ=0.05 non-necessity margin) | GATE_VERDICT | SUPPORTED | KILL | E1_KILL | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_PARTIAL | APPROVED | decision-layer prior embedded in δ/weights | protocol VOID | — |
| C_E1_PRODUCT_THESIS | A generative LM is necessary/preferred proposer substrate for the old synthetic task under frozen E1 U | PRODUCT_THESIS | FALSIFIED | KILL | E1_KILL | DECISION-GOVERNANCE | PUBLIC_TAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_PARTIAL | HEDGE_REQUIRED | new U or R★ create new claim IDs; do not withdraw this scoped result | protocol/U VOID for this estimand | — |
| C_E3_NORMALIZE_RESULT | Frozen normalize-then-match rescued 0/486 M0 exact failures | MEASUREMENT | SUPPORTED | N/A | — | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_REPRODUCIBLE | APPROVED | thin normalize ≠ synonyms/paraphrase | recompute mismatch | — |
| C_E3_AGENT_AUDIT | agent-rubric-pass-1 assigned the frozen label faithful to 0/100 sampled M0 exact errors under the frozen rubric | MEASUREMENT | SUPPORTED | PASS | EXACT_SURVIVES | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_REPRODUCIBLE | HEDGE_REQUIRED | single agent pass; no IAA; historical filename says human | recompute / label-file corruption | — |
| C_E3_HUMAN_STATUS | Independent human or clinician validation of exact match has not been completed; IAA is absent. | IMPLEMENTATION_STATE | SUPPORTED | N/A | — | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | ABSENT_EXPECTED | NO_RESULT | APPROVED | agent audit ≠ human validation | repository inventory error (if arm actually exists) | completed independent human/clinician study with recorded agreement |
| C_FABRIC_SLICE | On closed synthetic inst0, propose→verify→abstain under rules-strong v2 drove presented-error → 0 | MEASUREMENT | SUPPORTED | PASS | FABRIC_SLICE_V1 | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_REPRODUCIBLE | HEDGE_REQUIRED | closed world; ledger = per-run rewritten JSONL | protocol VOID / recomputation mismatch | independently evaluated imperfect/open-world verifier (new claim ID) |
| C_NANOSCRIBE_STATE | Repo evidences fabric vertical slice; does not evidence NanoScribe control plane, durable memory, routing, tools, permissions, distribution, or UI | IMPLEMENTATION_STATE | SUPPORTED | N/A | — | PUBLIC-ANCHORED | PUBLIC_TAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_REPRODUCIBLE | APPROVED | absence in audited repo ≠ proof no private impl | repository inventory error | evidenced modules are implemented |
| C_LORA_GEOM | LoRA preserves copy-circuit geometry | INTERPRETATION | SPECULATION | GATED_STOP | E2 | ASPIRATIONAL | PUBLIC_UNTAGGED | PUBLIC_TAGGED | ABSENT_EXPECTED | NO_RESULT | FORBIDDEN | no results_e2_* | — | valid E2 RESULT after owner re-scope |
| C_E2_STATUS | E2 U1–U4 universe discrimination has not produced a RESULT; E2 is GATED/STOP. | IMPLEMENTATION_STATE | SUPPORTED | GATED_STOP | E2 | PUBLIC-ANCHORED | PUBLIC_TAGGED | PUBLIC_TAGGED | ABSENT_EXPECTED | NO_RESULT | APPROVED | terminated U3 residue is not a measurement | status prose invents a RESULT | owner-authorized re-scope followed by a valid RESULT |
| C_RSTAR_VALUE | Generative proposers add value in regime R★ under U_R★ | FUTURE_HYPOTHESIS | SPECULATION | BLOCKED | E4 | ASPIRATIONAL | PUBLIC_TAGGED | PUBLIC_TAGGED | ABSENT_EXPECTED | NO_RESULT | FORBIDDEN | no builder/world/scores | — | E4 RESULT under owner authorization |
| C_ZERO_HALLUC_OPEN | Zero hallucination in the open world | PRODUCT_THESIS | SPECULATION | N/A | — | ASPIRATIONAL | PUBLIC_TAGGED | ABSENT_EXPECTED | ABSENT_EXPECTED | NO_RESULT | FORBIDDEN | only zero accepted violations of R if ever proven for that R | — | soundness proof for a specified R |
| C_CLINICAL_DEPLOYMENT | The current nano-lm evidence supports clinical deployment readiness. | PRODUCT_THESIS | SPECULATION | N/A | — | ASPIRATIONAL | PUBLIC_UNTAGGED | ABSENT_EXPECTED | ABSENT_EXPECTED | NO_RESULT | FORBIDDEN | synthetic task; no clinical validation, workflow study, risk study, or deployment evidence | — | clinical validation program with recorded evidence |

### Notes on selected rows

- **C_E1_MEASUREMENT / GATE / PRODUCT_THESIS:** reproducibility = `PUBLIC_PARTIAL` because L/C clean-clone reconstruction remains pending (`repro_note` in JSON).
- **C_E2_STATUS:** supported status fact that measurement was **not** completed; not an inconclusive measurement.
- **C_E3_HUMAN_STATUS:** negative current-state row (replaces positive unresolved validation sentence).
- **C_ADAPT_DATA_INTERP:** no “substitutes” hardening.
- **C_CLINICAL_DEPLOYMENT:** restored policy-ban / forbidden public wording row.
- **Existing premature tag:** `post-alpha-evidence-freeze-2026-07-31` → `a9d12cb` (PUBLIC; preserve; do not recreate). See `TAG_AUDIT_POST_ALPHA.md`.

### Retired IDs

- `C_SCALE_FLAT` → split → C_SCALE_OBSERVED + C_PARAMETER_ONLY_EFFECT + C_OWNSTACK_200M_FULLFT_GATE
- `C_ADAPT_DATA` → split → C_ADAPT_DATA_CELLS + C_ADAPT_DATA_INTERP
- `C_E1_KILL` → split → C_E1_MEASUREMENT + C_E1_GATE + C_E1_PRODUCT_THESIS
- `C_E3_NORM` → prefer C_E3_NORMALIZE_RESULT
- `C_E3_RUBRIC` → prefer C_E3_AGENT_AUDIT + C_E3_HUMAN_STATUS
- `C_E3_HUMAN_VALIDITY` → replaced by C_E3_HUMAN_STATUS
- `C_NANOSCRIBE_IMPL` → prefer C_NANOSCRIBE_STATE

## Required public wording for E3 (H1)

> A bounded agent-applied rubric audit assigned the frozen label `faithful` to 0 of 100 sampled exact errors. This single agent pass does not establish independent human or clinician acceptability, inter-rater agreement, or synonym-equivalence validity.

> The agent-rubric audit is complete. Independent human or clinician validation, IAA, and broader semantic-equivalence validation remain unresolved.

## Owner approval

Approve this DIFF E revision to replace `papers/EVIDENCE_LEDGER.md` (and optionally commit `EVIDENCE_LEDGER_PROPOSED.json` as the machine twin). Until then: proposal-only.

