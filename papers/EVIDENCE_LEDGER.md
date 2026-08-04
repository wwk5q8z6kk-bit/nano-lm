# Evidence ledger

This is the canonical boundary between measured evidence and project ambition. A
claim belongs here only when its wording, result, limitations, and supporting
artifact agree. Product plans are not evidence.

Evidence anchors:

- Paper α: tag `paper-alpha-v1` at `0e01d73205e9c35ea32925fd4d6c7e5fceb61137`
- Premature post-α tag, retained for history: `post-alpha-evidence-freeze-2026-07-31` at `a9d12cb1c456f6c465284e1d469c6326cb14d329`
- Reconciled post-α evidence: `post-alpha-reconciled-evidence-freeze-2026-07-31` at `67bf87b1f968a38e68c0225b2b556f7bba5ea1cc`
- Machine-readable claim-to-artifact map: `papers/EVIDENCE_MANIFEST.json`

Do not strengthen a claim beyond the table below. New experiments either add a
new claim ID or explicitly replace one with traceable evidence.

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
| C_ADAPT_DATA_INTERP | The observed 2×2 cell pattern is consistent with a weak-base × full-FT interaction. Whether additional pretraining and LoRA are mechanistic substitutes is unresolved. | INTERPRETATION | PLAUSIBLE | N/A | — | DECISION-GOVERNANCE | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | NO_RESULT | HEDGE_REQUIRED | E2 unidentified | — | valid E2 result under a new preregistered design |
| C_INTERFERENCE | The preregistered isolated-versus-contained lexical-interference contrast produces an effect ≥ the C1b support threshold | GATE_VERDICT | FALSIFIED | REFUTED | C1B_ISOLATED_VS_CONTAINED | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_PARTIAL | APPROVED | general lexical/morph effects remain open; primary JSONL TRACKED | protocol VOID | new registered lexical contrast |
| C_C3_TB | Transition availability and boundary type each produce effects ≥ the preregistered 40-pt support threshold in C3 | GATE_VERDICT | FALSIFIED | REFUTED | C3_TRANSITION_BOUNDARY | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_PARTIAL | APPROVED | replication JSONL IGNORED_LOCAL + local_raw_archive; wide intervals/imbalance | protocol VOID | equivalence-powered redesign |
| C_C3_L | Length factor drives residual (≥ support threshold) in C3 | GATE_VERDICT | UNRESOLVED | UNRESOLVED | C3_LENGTH | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_PARTIAL | APPROVED | underpowered | protocol VOID | decisive length arm under amended prereg |
| C_MORPH_DESC | Morphological re-inflection was the largest post-hoc descriptive error category in the C3 census (~44% of core-cell misses under recorded classification) | MEASUREMENT | SUPPORTED | N/A | — | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_PARTIAL | HEDGE_REQUIRED | post-hoc classification | re-label mismatch under same census definition | — |
| C_MORPH | Morphological re-inflection is the causal residual mechanism | INTERPRETATION | SPECULATION | N/A | — | ASPIRATIONAL | PUBLIC_UNTAGGED | PUBLIC_TAGGED | ABSENT_EXPECTED | NO_RESULT | FORBIDDEN | exploratory only | — | preregistered causal RESULT |
| C_POINTER_P1 | Explicit pointer/copy head closes OOD gap (P1) | GATE_VERDICT | VOID | VOID | POINTER_P1 | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_REPRODUCIBLE | APPROVED | manipulation check failed | — | — |
| C_POINTER_P2 | Copy-supervised pointer head closes OOD gap for this preregistered implementation | GATE_VERDICT | FALSIFIED | REFUTED | POINTER_P2 | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_REPRODUCIBLE | APPROVED | this impl only | protocol VOID | new implementation gets a new claim ID |
| C_NANO_H2_POINTER_DEV | On sealed synthetic development data, the best of six H2 direct state/pointer checkpoints scored raw 61.82% overall, 64.75% held-value, and 88.40% missing-target accuracy, but 0% absence, 24.80% conflict, 21.60% uncertain, and 1,707 wrong-presented fields; post-verification it scored 65.74% overall and zero false-presented fields, so H2 failed its frozen raw-plus-final quality gate and was rejected | GATE_VERDICT | SUPPORTED | REJECT | H2_NATIVE_POINTER_SPAN | CONTENT-ADDRESSED | PUBLIC_UNTAGGED | TRACKED_SUMMARY_LOCAL_LARGE | PRESENT | LOCAL_CONTENT_ADDRESSED | APPROVED | sealed development used for model selection; two seeds/six checkpoints; exact architecture/recipe only; no latency; `fresh-v1` sealed; large row/checkpoint artifacts local | artifact/hash mismatch or protocol VOID | bounded H3 receives a new claim ID and requires a frozen contract |
| C_NANO_H3_EVIDENCE_QUERY_DEV | On known synthetic development data, the training-only-selected H3 evidence-query checkpoint scored raw 56.42% overall, 72.55% held-value, 85.20% missing-target, 45.52% absence, 37.60% conflict, and 38.80% uncertain accuracy, with zero decode failures and 1,264 wrong-presented fields; it failed frozen uncalibrated admission and was rejected before threshold, verifier, latency, or `fresh-v1` | GATE_VERDICT | SUPPORTED | REJECT | H3_EVIDENCE_QUERY_POINTER | CONTENT-ADDRESSED | PUBLIC_UNTAGGED | TRACKED_SUMMARY_LOCAL_LARGE | PRESENT | LOCAL_CONTENT_ADDRESSED | APPROVED | known synthetic development; two seeds/six checkpoints; perfect training-only calibration; exact architecture-plus-training-family intervention only; downstream stages intentionally null; `fresh-v1` sealed; large row/checkpoint artifacts local | artifact/hash mismatch or protocol VOID | completed H4 has its own claim ID; subsequent interventions require a new claim ID and frozen contract |
| C_NANO_H4_SURFACE_TRANSFER_DEV | On known synthetic development data, the training-only-selected H4 surface-transfer candidate scored raw 44.96% overall, 38.03% held-value, 53.60% missing-target, 7.99% absence, 22.80% conflict, and 33.60% uncertain accuracy, with zero decode failures and 2,141 wrong-presented fields; it failed frozen uncalibrated admission and was rejected before threshold, verifier, latency, or sealed confirmation | GATE_VERDICT | SUPPORTED | REJECT | H4_SURFACE_TRANSFER_DATA_ONLY | CONTENT-ADDRESSED | PUBLIC_UNTAGGED | TRACKED_SUMMARY_LOCAL_LARGE | PRESENT | LOCAL_CONTENT_ADDRESSED | APPROVED | known adaptive synthetic development; two seeds/six checkpoints; exact data-only recipe; evidence order, distractors, and long-context distance were not changed; downstream stages intentionally null; large row/checkpoint artifacts local | artifact/hash mismatch or protocol VOID | completed H5 has its own claim ID; subsequent interventions require a new claim ID and frozen contract |
| C_NANO_H5_BALANCED_REPLAY_DEV | On known synthetic development data, the training-only-selected H5 fixed 50:50 replay candidate scored raw 78.18% overall, 74.32% held-value, 100% missing-target, 67.80% absence, 59.60% conflict, and 64.80% uncertain accuracy, with zero decode failures and 1,013 wrong-presented fields; it failed frozen absence, conflict, and uncertainty admission gates and was rejected before threshold, verifier, latency, or sealed confirmation | GATE_VERDICT | SUPPORTED | REJECT | H5_BALANCED_REPLAY_DATA_ONLY | CONTENT-ADDRESSED | PUBLIC_UNTAGGED | TRACKED_SUMMARY_LOCAL_LARGE | PRESENT | LOCAL_CONTENT_ADDRESSED | APPROVED | known adaptive synthetic development; two seeds/six checkpoints; exact 50:50 replay recipe only; state/span localization descriptive not causal; downstream stages intentionally null; large row/checkpoint artifacts local | artifact/hash mismatch or protocol VOID | H6 representation-only coupling test requires its own claim ID and frozen contract |
| C_E1_MEASUREMENT | Under frozen E1 U, reported utilities: M1≈0.999, official M0≈0.925, M2≈0.886 (margin M1−M0≈+0.074); ρ = review load | MEASUREMENT | SUPPORTED | N/A | — | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_PARTIAL | APPROVED | M1 generator-aligned; L/C device-normalization audit residual | recompute mismatch / protocol VOID | none; new U receives a new claim ID |
| C_E1_GATE | Frozen E1 rule returns KILL (M1 exceeds M0; M2 within δ=0.05 non-necessity margin) | GATE_VERDICT | SUPPORTED | KILL | E1_KILL | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_PARTIAL | APPROVED | decision-layer prior embedded in δ/weights | protocol VOID | — |
| C_E1_PRODUCT_THESIS | A generative LM is necessary/preferred proposer substrate for the old synthetic task under frozen E1 U | PRODUCT_THESIS | FALSIFIED | KILL | E1_KILL | DECISION-GOVERNANCE | PUBLIC_TAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_PARTIAL | HEDGE_REQUIRED | new U or R★ create new claim IDs; do not withdraw this scoped result | protocol/U VOID for this estimand | — |
| C_E3_NORMALIZE_RESULT | Frozen normalize-then-match rescued 0/486 M0 exact failures | MEASUREMENT | SUPPORTED | N/A | — | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_REPRODUCIBLE | APPROVED | thin normalize ≠ synonyms/paraphrase | recompute mismatch | — |
| C_E3_AGENT_AUDIT | agent-rubric-pass-1 assigned the frozen label faithful to 0/100 sampled M0 exact errors under the frozen rubric | MEASUREMENT | SUPPORTED | PASS | EXACT_SURVIVES | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_REPRODUCIBLE | HEDGE_REQUIRED | single agent pass; no IAA; historical filename says human | recompute / label-file corruption | — |
| C_E3_HUMAN_STATUS | Independent human or clinician validation of exact match has not been completed; IAA is absent. | IMPLEMENTATION_STATE | SUPPORTED | N/A | — | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | ABSENT_EXPECTED | NO_RESULT | APPROVED | agent audit ≠ human validation | repository inventory error (if arm actually exists) | completed independent human/clinician study with recorded agreement |
| C_FABRIC_SLICE | On closed synthetic inst0, propose→verify→abstain under rules-strong v2 drove presented-error → 0 | MEASUREMENT | SUPPORTED | PASS | FABRIC_SLICE_V1 | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_REPRODUCIBLE | HEDGE_REQUIRED | closed world; ledger = per-run rewritten JSONL | protocol VOID / recomputation mismatch | independently evaluated imperfect/open-world verifier (new claim ID) |
| C_NANOSCRIBE_STATE | Repo evidences fabric vertical slice; does not evidence NanoScribe control plane, durable memory, routing, tools, permissions, distribution, or UI | IMPLEMENTATION_STATE | SUPPORTED | N/A | — | PUBLIC-ANCHORED | PUBLIC_TAGGED | PUBLIC_TAGGED | PRESENT | PUBLIC_REPRODUCIBLE | APPROVED | absence in audited repo ≠ proof no private impl | repository inventory error | evidenced modules are implemented |
| C_LORA_GEOM | LoRA preserves copy-circuit geometry | INTERPRETATION | SPECULATION | GATED_STOP | E2 | ASPIRATIONAL | PUBLIC_UNTAGGED | PUBLIC_TAGGED | ABSENT_EXPECTED | NO_RESULT | FORBIDDEN | no results_e2_* | — | valid E2 result under a new preregistered design |
| C_E2_STATUS | E2 U1–U4 universe discrimination has not produced a RESULT; E2 is GATED/STOP. | IMPLEMENTATION_STATE | SUPPORTED | GATED_STOP | E2 | PUBLIC-ANCHORED | PUBLIC_TAGGED | PUBLIC_TAGGED | ABSENT_EXPECTED | NO_RESULT | APPROVED | terminated U3 residue is not a measurement | status prose invents a RESULT | completed preregistered E2 measurement |
| C_E4_RESULT | Under frozen U_R★ on locked R★ v1, best classical U≈0.638 (C-M2) and best generative+verify U≈−1.623 (G-ref verify-on); sensitivity flip false. | MEASUREMENT | SUPPORTED | N/A | — | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_UNTAGGED | PRESENT | PUBLIC_PARTIAL | APPROVED | regime-scoped; after premature freeze tag | recompute mismatch / protocol VOID | — |
| C_E4_GATE | E4 Gate 4 decision under frozen δ=0.05 is KILL for generative substrate on tested R★ v1. | GATE_VERDICT | SUPPORTED | KILL | E4 | PUBLIC-ANCHORED | PUBLIC_UNTAGGED | PUBLIC_UNTAGGED | PRESENT | PUBLIC_PARTIAL | APPROVED | does not support NanoScribe/fabric expansion | protocol VOID | — |
| C_RSTAR_VALUE | Whether some preregistered R★ revision can yield generative+verify utility above classical remains open; revision budget ≤1 requires a new preregistered design. | FUTURE_HYPOTHESIS | UNRESOLVED | N/A | RSTAR_REVISION | DECISION-GOVERNANCE | PUBLIC_UNTAGGED | PUBLIC_UNTAGGED | ABSENT_EXPECTED | NO_RESULT | HEDGE_REQUIRED | tested R★ v1 already KILL; no silent redesign | — | new preregistration and result |
| C_ZERO_HALLUC_OPEN | Zero hallucination in the open world | PRODUCT_THESIS | SPECULATION | N/A | — | ASPIRATIONAL | PUBLIC_TAGGED | ABSENT_EXPECTED | ABSENT_EXPECTED | NO_RESULT | FORBIDDEN | only zero accepted violations of R if ever proven for that R | — | soundness proof for a specified R |
| C_CLINICAL_DEPLOYMENT | The current nano-lm evidence supports clinical deployment readiness. | PRODUCT_THESIS | SPECULATION | N/A | — | ASPIRATIONAL | PUBLIC_UNTAGGED | ABSENT_EXPECTED | ABSENT_EXPECTED | NO_RESULT | FORBIDDEN | synthetic task; no clinical validation, workflow study, risk study, or deployment evidence | — | clinical validation program with recorded evidence |

### Notes on selected rows

- **C_E1_MEASUREMENT / GATE / PRODUCT_THESIS:** reproducibility = `PUBLIC_PARTIAL` because L/C clean-clone reconstruction remains pending (`repro_note` in JSON).
- **C_E2_STATUS:** supported status fact that measurement was **not** completed; not an inconclusive measurement.
- **C_E3_HUMAN_STATUS:** negative current-state row (replaces positive unresolved validation sentence).
- **C_ADAPT_DATA_INTERP:** no “substitutes” hardening.
- **C_NANO_H2_POINTER_DEV:** raw metrics describe the learned model proposal
  before transcript verification. The post-verification zero false-presented
  result came from verifier acceptance/rejection behavior and does not satisfy
  the failed raw-model gate. The tracked summary is SHA-256
  `f61bb7f0f3401dfbaff8a5ab7e987d313a4811442f790b1d03f0883b403806cc`;
  the 34,000,758-byte row-level evaluation remains local at SHA-256
  `641c08f826a5669220cd7fda8c52fbd2a682a352c8be36a97f93a099ecfe3833`.
  H2 stopped at quality: latency was intentionally not measured and
  `fresh-v1` was not read.
- **C_NANO_H3_EVIDENCE_QUERY_DEV:** all six H3 checkpoints scored 4,000/4,000
  on training-only calibration; the frozen tie-break selected
  `seed-20260805-epoch-1`. The tracked summary is SHA-256
  `cbd23c6a5799179b487119e9bee6e181dd328e691ee5388b12d03689f538ec82`;
  the 1,790,093-byte row-level evaluation remains local at SHA-256
  `df6896855980172aabd03149affffca8352c2ef9732fb860a79b3f50d854c831`.
  Failed uncalibrated admission correctly left calibrated raw, verifier-final,
  latency, and `fresh-v1` unassessed. Perfect calibration versus poor
  development is evidence for a data-transfer repair, not proof that data is
  the only possible cause.
- **C_NANO_H4_SURFACE_TRANSFER_DEV:** training-owned calibration selected
  `seed-20260806-epoch-2`, checkpoint SHA-256
  `6408524c43b6ada8249aeb83e440b6aa0f64512006219663be4105f6d586e13f`.
  The 1,855,164-byte development evaluation remains local at SHA-256
  `b8f8b350f4ad772b06a292c5e156d097e549f62e15be09c3c55c608574c0ed82`.
  Allergy emitted `absent` on all 1,000 rows; medication had 832/1,000 correct
  states but only 194/1,000 exact spans; all 413 absent fields had the correct
  state but only 33 exact evidence spans. These post-hoc observations motivate
  replay and failure localization; they do not prove a causal mechanism.
- **C_NANO_H5_BALANCED_REPLAY_DEV:** training-owned calibration selected
  `seed-20260805-epoch-3`, checkpoint SHA-256
  `04ba7b4d0dc876ca3d8de7fe7d809ca16796e5bf55249ad93ba1dd3557c394fe`.
  The 1,836,514-byte development evaluation remains local at SHA-256
  `c67393962299470fc6b5026031b61617bbae85a2883105b8b8abfcdb30820c47`;
  the complete result archive is SHA-256
  `bab5327e900597a083cb04631e645f2e0f500f14f01ec7db195e754b34620749`.
  H5 passed overall, held-value, missing, failure, and state-balance gates but
  failed absence, conflict, and uncertainty. The frozen stop left calibrated,
  verifier-final, latency, and sealed-confirmation stages null. Its 727
  state-correct/span-wrong and 270 span-correct/state-wrong fields motivate a
  coupling test but do not prove the cause.
- **C_CLINICAL_DEPLOYMENT:** restored policy-ban / forbidden public wording row.
- **Existing premature tag:** `post-alpha-evidence-freeze-2026-07-31` → `a9d12cb` is historical and must not be moved or recreated.

## Required public wording for E3 (H1)

> A bounded agent-applied rubric audit assigned the frozen label `faithful` to 0 of 100 sampled exact errors. This single agent pass does not establish independent human or clinician acceptability, inter-rater agreement, or synonym-equivalence validity.

> The agent-rubric audit is complete. Independent human or clinician validation, IAA, and broader semantic-equivalence validation remain unresolved.

## Use

For a compact narrative of the same evidence, see
`papers/EMPIRICAL_FOUNDATION.md`. For exact files and hashes, use
`papers/EVIDENCE_MANIFEST.json` and `artifacts/MANIFEST.json`.
