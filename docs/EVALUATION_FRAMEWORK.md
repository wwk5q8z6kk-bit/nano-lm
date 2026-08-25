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
| **SLW** — Synthetic Longitudinal World | **First** Core benchmark for Program K1. Ground truth constructible. **Not built.** |
| **LCRB** — Longitudinal Clinical Reconstruction Benchmark | **Final** benchmark, not the first. Medicine is the stress test, not the debugging environment. **Not built.** |

#### Ordering correction (2026-08-25)

An earlier revision listed LCRB as the flagship to build first. That was wrong
and is retracted here rather than edited away. Medicine is an excellent stress
test and a poor debugging environment: when a run fails you cannot tell whether
the architecture, the extraction, the ontology, or the ground truth is at fault.

**Build the Core against a synthetic world where the answer is known, then move
to medicine.** This is not a detour around the capability ladder — it is the
existing Core/DomainPack split applied honestly. See `SYSTEM_ARCHITECTURE.md`.

#### SLW — synthetic longitudinal world (design sketch)

A generated world of ~100 entities with relationships, events, state changes,
and deliberately imperfect observation: conflicting reports, delayed reports,
missing observations, temporal dependencies. Ground truth is the generator's
own state, so every question has a checkable answer.

Questions it must answer, each independently scorable:
what changed · what caused the change · what happened first · what evidence
supports the conclusion · what information is missing · predict the next state.

The last one is the discriminator: **only a system with an actual state model
can predict the next state.** Retrieval and summarisation cannot fake it.

#### Complexity ladder to LCRB

```text
L1 single document          L5 multimodal records
L2 multiple documents       L6 continuous data stream
L3 longitudinal documents   L7 tool use
L4 contradictory documents  L8 open-world unfamiliar information
```

Each rung is a measurable step, and the rung where performance breaks is the
result — not a failure to be engineered around.

#### LCRB (design sketch — construction not authorized)

Each case: 5-20 years, 100-1,000+ pages, multiple institutions and clinicians,
deliberate contradictions, deliberate missing data, lab series, medication
history, imaging reports, procedures, patient-reported material. Ground truth is
expert-constructed, so cases must be synthetic or de-identified.

Scored outputs: timeline · problem list · medication trajectory · diagnostic
trajectory · treatment trajectory · lab trajectories · imaging evolution ·
functional trajectory · contradictions · missing information · current state ·
evidence citations · concise and full presentations.

**Leakage is the dominant construction risk, not label noise.** A benchmark with
gold answers over generated records reproduces exactly the failure this program
already hit three times (D5.1-D5.3 in `artifacts/DEFECT_INDEX.md`): the answer
template handed over the gold string, the parser substituted gold on empty
output, and the *question* interpolated the gold value in 14/16 slots — voiding a
whole 2x2 ablation. LCRB must ship with a parrot floor (what a question-only
model scores with no record access) before any model number is quoted.

### Evidence-weighted recall (proposed primary metric)

Summarization quality is the wrong objective. A fluent 2-page summary that drops
a critical adverse drug reaction is a failure; a 4-page one carrying every
critical fact with provenance is better.

```text
score = Σ_facts  importance(fact | task) · recalled(fact) · traceable(fact)
```

- **Importance is task-conditional**, not intrinsic: a remote cardiac event is
  background for dermatology and critical for anaesthesia. Any fixed importance
  weighting is therefore wrong by construction.
- **`traceable` is a gate, not a bonus** — an unciteable correct fact scores 0.
  This is the same discipline as `absence-never-from-silence` in `fabric/`.
- **Do not optimise for brevity.** The objective is maximum clinically relevant
  information at minimum cognitive burden, which is not minimum word count.

Unresolved before this can be a gate: who assigns `importance`, and the
inter-rater agreement on it. Without that it is a proposal, not a metric.

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
