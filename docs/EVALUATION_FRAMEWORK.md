# Evaluation Framework

## Principles

1. **Pre-register** what would change the roadmap before running.
2. **Same instrument, same rules** when comparing arms.
3. **Honest failure** — a FAIL that locates the next lever beats an unregistered PASS.
4. **Scope claims** — verifier relation, distribution, and utility function must accompany every metric.
5. **Human review** for consequential domains — automatic metrics necessary, not sufficient.

## Utility pattern (when comparing solvers)

\[
U = Q - w_E E - w_R R - w_L L - w_C C
\]

Symbols vary by program; always freeze weights before scoring. Kill/keep: arm must beat best cheaper solver by \(\delta\) (default 0.05 unless preregistered).

## By capability program

| Program | Primary measures |
|---------|-------------------|
| P1 Scribing | factual precision, critical omission, exact-value transport, span correctness, negation/uncertainty/temporality, attribution, section completeness, coherence, unsupported claims, review burden, clinician edit effort |
| P2 Summarization | precision, critical omission, salience, compression, provenance retention, contradiction/uncertainty retention |
| P3 Charting | entity resolution, timeline correctness, supersession, contradiction handling, traceability to source events |
| P4+ | program-specific — see [CAPABILITY_LADDER.md](CAPABILITY_LADDER.md) |

## P1 exit gate (not yet satisfied)

Stage-1 mastery requires satisfactory performance across the scribing metric set **plus** external medical-dialogue evaluation and **blinded human evaluation**.

Mock/synthetic benchmark success ≠ clinical validation.

## World-class P1 exit gate (draft rubric)

Operational definition for "best evidence-grounded medical scribe" — targets are **preregistration hypotheses**, not current claims. See [ACCELERATED_RESEARCH_CAMPAIGN_V2.md](ACCELERATED_RESEARCH_CAMPAIGN_V2.md) for campaign sequencing.

### Automatic metrics (same evaluator, all arms)

| Metric | Measurement | Interim gate (v2) | World-class target (confirmatory) |
|--------|-------------|-------------------|-----------------------------------|
| Critical error rate | Unsupported/contradicted claims with clinical harm potential | Track; zero tolerance class for allergy/dose | ≤ 0.5% external held-out |
| Exact-value transport | Normalized numeric/unit/dose match | Report separately from span | ≥ 95% eligible |
| Evidence span correctness | Gold offset agreement (transport layer) | ≥ 25% C2 (`exact_gold_span_rate`) | ≥ 85% eligible |
| Assertion / negation / temporality | STATED/DENIED/NOT_MENTIONED + axes | ≥ 90% C2 | ≥ 95% eligible |
| Omission severity | Criticality-weighted recall | Critical omission ≤ 5% interim | ≤ 2% critical; ≥ 90% any-tier |
| Unsupported claims in note | Post-render verifier | Zero critical unsupported | Zero critical; ≤ 1% non-critical |
| Malformed output | Parser/selector failures | ≤ 5% C2 | ≤ 1% |

**Campaign v1 baseline (measured):** managed reference C2 `exact_gold_span_rate` = 0.110, `assertion_state_correct_rate` = 1.00 (`artifacts/campaign/student_gap_v1.json`).

### Human metrics (required for exit)

| Metric | Protocol |
|--------|----------|
| Clinician edit distance | Blinded review; median edits / note length |
| Time to final note | Transcript → clinician-accepted note |
| Critical-error adjudication | ≥2 reviewers; severity rubric |
| Usefulness / safety | Owner sign-off on P1 exit record |

### Benchmarks (planned)

| Dataset | Role |
|---------|------|
| Internal C1/C2 (`p1_screening_eval_v1`) | Frozen regression — never train |
| MTS-Dialog, ACI-Bench, PriMock57 | External OOD dialogue faithfulness |
| Classical + verification stack | Comparison arm per E1/E4 — must beat on utility U |

### Kill / keep (utility)

Arm must beat **classical extraction + constrained span selection + independent verification** on frozen U at equal or lower review burden before claiming P1 superiority. Raw LLM generation alone is not the comparison baseline.

## Evidence vs product evaluation

| Layer | Location | Use |
|-------|----------|-----|
| Scientific | `papers/`, `trajectory/` | Immutable tagged results, preregistrations |
| Product / wedge | `wedge_v1/` harnesses | Local usefulness, abstention economics |
| P1 scribe (future) | external + human protocols | [domains/medical/EVALUATION_PROTOCOL.md](domains/medical/EVALUATION_PROTOCOL.md) |

## Failure loop

Every important experiment enters [FAILURE_TO_ARCHITECTURE.md](FAILURE_TO_ARCHITECTURE.md):

```text
failure → classify → competing explanations → cheapest discriminating test
→ result → belief update → architectural response → regression test
```
