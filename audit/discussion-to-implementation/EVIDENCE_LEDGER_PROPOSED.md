# Evidence Ledger — Proposed Strengthened Form (DIFF E)

*Proposal only. **Do not** replace `papers/EVIDENCE_LEDGER.md` until owner approves this DIFF E.*

## Disposition

```text
DIFF_E = OWNER_APPROVED_REMEDIATIONS
# approved via OWNER_APPROVED_DIFF_E_REMEDIATION (20260731T180656Z)
# nine schema/status remediations + enum validation PASS
```

Machine-readable twin: `EVIDENCE_LEDGER_PROPOSED.json` (schema `nano-lm.evidence_ledger_proposed.v2`).
Enum validation: **PASS** (literal vocabulary members only).
Remediation checklist: `DIFF_E_REMEDIATION_ACCEPTANCE.md`.

## Review snapshot

```json
{
  "review_snapshot_id": "review-2026-07-31-03",
  "head_commit": "2e03e0df564008cf51c4309e9dbdf01a59c3c7b5",
  "paper_alpha_tag": "0e01d73205e9c35ea32925fd4d6c7e5fceb61137",
  "premature_post_alpha_tag": "post-alpha-evidence-freeze-2026-07-31",
  "premature_post_alpha_tag_target": "a9d12cb1c456f6c465284e1d469c6326cb14d329",
  "worktree_manifest_sha256": "bf7114f3864a4996697b4c13c378a917290ef1e8d2873dbc43765a9ebe4044aa",
  "worktree_manifest_sha256_excluding_json": "a1f537eafe76ba389db21246befeea9d9309d8bf6f67681f6b27617943d515ee",
  "proposed_json_sha256": "7b6e3e5e667a019e6b50dfccaefb3b7c630b5c4448497ffbe224e50289efbfb8",
  "note": "DIFF E schema/status remediation acceptance; premature tag preserved"
}
```

Each row carries `last_reviewed_snapshot_id = review-2026-07-31-03`.

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

```text
STATUS = OWNER_APPROVED_REMEDIATIONS
MARKER = OWNER_APPROVED_DIFF_E_REMEDIATION (20260731T180656Z)
CONSTRAINT = Do not recreate/move post-alpha-evidence-freeze-2026-07-31
```

Remediations approved. Live ledger status banners synced; claim table already matched remediations.

---

## PROPOSED (2026-08-25) — P1 span-port line. Staged, not a live ledger edit.

Owner approval required before these reach `papers/EVIDENCE_LEDGER.json`.
Constraint honoured: no frozen tag recreated or moved.

### Proposed new claims

| Claim ID | Exact wording | Type | Epistemic | Evid class | Limitations | Invalidation |
|---|---|---|---|---|---|---|
| `C_SPANPORT_DELIMIT` | On the canonical unified question form, with all three leak channels closed, Qwen2.5-1.5B selects the correct conversational turn for 97/120 gold-bearing slots (81%) but delimits the gold span within it for 2/120; located quotes are turn-scale (median 29 chars vs gold median 8, ratio 0.32, median quote/enclosing-turn 1.000, 1 of 95 exceeding its turn) | MEASUREMENT | SUPPORTED | PUBLIC-ANCHORED | one model, one suite, synthetic; containment is looser than exact offsets — always cite with the length bound | a length audit showing quotes are not turn-scale; or LOCATED moving with question form as exact-extent does |
| `C_SPANPORT_FORM_SENSITIVITY` | `asserted_grounded` for the identical condition `C1off_C2off_Qon_QSoff` is 16/192 under one question form and 2/192 under another (8×), while LOCATED is 95/120 vs 97/120 (invariant). Exact-extent is therefore demoted to secondary and may not be cited without this disclosure; LOCATED is primary | MEASUREMENT | SUPPORTED | PUBLIC-ANCHORED | two forms only; canonical = unified | LOCATED shown to move with form |
| `C_SPANPORT_C1_VERDICT` | The C1 leakage contrast is UNRESOLVED — coverage shift — on both available form-matched pairs: grounded and unbound move in the same direction, co-movement rule fires | GATE_VERDICT | UNRESOLVED | PUBLIC-ANCHORED | C1 buys assertions, not evidence-finding | a pair where grounded rises and unbound does not |

### Proposed methodology note — pilot-then-confirm

**This is the program's first clean instance of pilot-then-confirm, and it should
become the default for any design whose effect size is itself measurable.**

Before the eight-cell grid ran, a pilot on throwaway encounters **disjoint from
every measurement instance** measured $\hat\pi_{C2} = 0/40$ (95% Clopper–Pearson
upper bound 0.072) and registered the C1×C2 interaction as **unidentified rather
than underpowered** — the distinction being that no replicate count repairs a
manipulation the model gives nothing to act on. That was recorded in
`research/preregistrations/PREREG_P1_leakage_2x2.md` §5 before any cell ran.

The landed data at n=192 confirmed it exactly: `asserted_grounded` identical
across all four C2 pairs in all twelve instances, **Δ 0.000, sd 0.000**.

Why this is worth more than the measurement alone: a null discovered after the
fact is indistinguishable from an underpowered design, and the program has
already paid for that ambiguity (C_C3_L, "underpowered"; the E1/E2 seed
underpower caveats). A null **named in advance, with its effect size bounded and
its unidentifiability argued from mechanism**, is not ambiguous. It cost one
minute of local compute against ~52 minutes for the grid.

Registered as `docs/RUNBOOK_contrast_hygiene.md` R7.

### Proposed status change

`C_NANOSCRIBE_STATE` — no change proposed, but note that the span-port
measurement line it describes is now known to have carried a prompt-side gold
channel from `09745ec` forward. Post-freeze span-metric claims in that window
need the taint check in `artifacts/measurement-integrity-audit.md` §1.3. No
frozen claim is affected (`audit/ADDENDUM_2026-08-25_leakage_taint_audit.md`).

### Proposed methodology note — dissociation by instrument manipulation

The form change was not designed as an experiment; it was a repair. But it
produced the cleanest result in the P1 line: **one manipulation moved
exact-extent 8× (16 → 2) and moved LOCATED not at all (95 → 97).** That is a
single dissociation, and it separates *finding the evidence* from *delimiting
it* more convincingly than decomposing a single accuracy scalar ever could.

Operational consequence: when a metric proves highly sensitive to an instrument
choice, that sensitivity is a **measurement to report**, not only a defect to
fix. The fragile axis and the invariant axis are different capacities, and the
fragility is what tells you which is which. Reporting order should follow
invariance, not tradition — hence LOCATED primary, exact-extent secondary.
