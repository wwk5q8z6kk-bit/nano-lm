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
| **LCRB** — Longitudinal Clinical Reconstruction Benchmark | The benchmark that decides the program. Not the **first** one run. **Not built.** |

#### Ordering correction (2026-08-25)

An earlier revision listed LCRB as the flagship to build first. That was wrong
about **ordering** and is retracted here rather than edited away.

**Medicine is the important benchmark. It is not the first benchmark.** Those are
separate claims and both hold:

- *Important* — LCRB is what the program is ultimately judged on. Medicine is
  DomainPack #1 precisely because it is high-consequence, structured and
  measurable (`PROJECT_CHARTER.md` § Why medicine first). Nothing here demotes
  it, and success on a synthetic world is **not** a result about medicine.
- *Not first* — a benchmark that decides the program is a bad instrument for
  debugging the Core. When a medical run fails, the fault could be the
  architecture, the extraction, the ontology, or the ground truth, and the run
  itself cannot tell you which. That is a property of the instrument, not a
  criticism of the domain.

**No conflict with "why medicine first."** The charter fixes *domain* priority:
medicine is DomainPack #1 and stays there. This fixes *benchmark* ordering: the
Core is debugged against a pack whose ground truth is constructible, then
measured on the pack that matters. See `SYSTEM_ARCHITECTURE.md` § DomainPack-0.

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

#### Benchmark ladder — B0 to B5

Each rung adds one class of difficulty. **The rung where performance breaks is
the result**, not a failure to be engineered around.

| # | Environment | Adds | Isolates |
|---|---|---|---|
| **B0** | Synthetic world (SLW) | nothing — state only | can it maintain state *without language*? |
| **B1** | Longitudinal text | language, many documents, contradiction, irrelevance | can it build state *from* language? |
| **B2** | Multimodal world | image, table, audio describing the same event | cross-modal identity |
| **B3** | Open-world research | unbounded sources, unknown terminology | autonomous information acquisition |
| **B4** | Scientific reasoning | hypotheses, experiments, conflicting results | hypothesis formation, correlation vs causation |
| **B5** | **Medicine (LCRB)** | consequence, specialised knowledge, all of the above | the program's decisive benchmark |

**B0 deliberately precedes language.** Streaming entity/event/state observations
with no text isolates the state machinery from linguistic competence — if the
architecture cannot answer "what is currently true" over 100 entities without
language, no language benchmark will diagnose why.

**B4 is the bridge, and it is not optional.** Medicine requires hypothesis
formation, evidence weighing, and correlation-versus-causation discipline. Meeting
those first in a domain where ground truth is checkable is what makes a medical
failure interpretable.

Within-rung document complexity (single → multiple → longitudinal →
contradictory) applies inside B1 onward.

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
