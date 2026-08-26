# Nano vNext — master specification

**Status:** canonical consolidation, 2026-08-25
**Evidence companion:** [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)
**Authority:** [`PROJECT_AUTHORITY.md`](PROJECT_AUTHORITY.md) §3 — which is where
this document's standing, and its supersession of `papers/NANOSCRIBE_VNEXT.md`,
are actually established. A document does not become canonical by saying so
about itself.

---

## 0. How to read this, and who owns what

This document consolidates. It does **not** re-derive structure that already
exists in executable form, because a second taxonomy is the fragmentation this
consolidation exists to end.

**Normative source for the architecture taxonomy is code, not this document.**
`nano/architecture.py` defines `Layer`, `ProvingStage`, `LearningLevel`,
`NEURAL_CANDIDATES`, `CROSS_CUTTING`, `CANONICAL_CHAIN`, `NANO_INVARIANT`,
`EXTENSIBLE_SETS` and `INVARIANT_SETS`, with tests. Where this document and that
module disagree, **the module wins** and this document is stale.

### Branch ownership map

| Concern | Owning branch | State when this was written |
|---|---|---|
| Architecture taxonomy, `nano/*` modules, capability spec | `frontier/accelerated-research-campaign-v2` | **Live and moving** — tip `09621fe`, 2026-08-25 12:05 |
| Evidence ledger, E-DELIMIT, leakage grid, authorization record | `work/e-delimit-result`, `work/architecture-program-review` | This document's branch |
| Span-port delimitation analysis, contrast-hygiene runbook | `work/leakage-power-analysis` @ `23ede77` | Checked out elsewhere; carries the **superseded** yes/no question form — see §23 |

`nano/` was read at `09621fe`. It is under active development by another
session; treat §§4–19 citations to it as a snapshot, and re-read before relying
on a specific symbol.

**Where cited paths resolve.** Of the 21 paths this document and
`RESEARCH_STATUS.md` cite, only 4 exist on `work/e-delimit-result`, the branch
these documents were authored on. Audited 2026-08-25:

| Path prefix | Resolves on |
|---|---|
| `nano/*`, `docs/NANO_CAPABILITY_SPEC.md`, `artifacts/campaign/*`, `papers/*`, `trajectory/*` | `frontier/accelerated-research-campaign-v2` (12 paths) |
| `artifacts/span_extent_L000_unified.json` | `work/leakage-power-analysis` (1 path) |
| `research/decision_records/2026-08-25-*`, `research/reviews/*` | `work/architecture-program-review` |
| `nanoscribe/*`, `docs/*`, `research/preregistrations/PREREG_E_DELIMIT.md`, `research/negative_results/*` | this branch (4 paths) |

Read a cited file with `git show <branch>:<path>` rather than switching
branches — two of the three branches above are checked out by other sessions.

### The eleven planes are the fourteen layers

Verified by explicit diff, not assumed. The taxonomies are content-identical;
only the numbering and one promotion differ.

| Plane (owner formulation) | `nano/architecture.py::Layer` |
|---|---|
| Identity & Authority | `III_identity_authority` |
| Observation / Perception | `IV_observation_perception` |
| Evidence & Provenance | `V_evidence_provenance` |
| World Representation | `VI_world_representation` |
| Temporal / Causal State | `VII_temporal_causal` |
| Memory & Knowledge | `VIII_memory_knowledge` |
| Cognitive Control | `IX_cognitive_kernel` |
| Capability / Tool Fabric | `X_capability_tool_fabric` |
| Artifact & Interaction Compilation | `XI_artifact_compilation` |
| Verification / Epistemic Control | `XII_verification_epistemic_control` |
| Learning / Adaptation | `XIV_learning_adaptation` |

Two layers the code adds, both of which the owner specification separately
requires as sections here: `I_ontology` (§4) and `II_epistemic_contract` (§5).

One promotion: **Dependency / Invalidation moves from cross-cutting to
`XIII_dependency_invalidation`**, a layer. This is the right call — invalidation
has its own state machine and its own failure modes, and burying it in
cross-cutting concerns is how stale derived conclusions survive a correction.

Cross-cutting is then: observability, security, resource control — plus
**evaluation, reproducibility, and science/evidence tracking**, which the code
adds and which this program's history justifies.

---

## 1. Mission

Build a general-purpose personal and medical intelligence system that runs
primarily **locally and cheaply**, and that earns every architectural component
experimentally.

The mission is not "a novel small language model." A novel model is at most an
implementation detail of the mission, and on current evidence not obviously the
right one (§23, breadth-before-specialization).

The system must **get the individual job done**. It must not assume that one
monolithic model performs every operation itself.

## 2. Product targets

Primary surfaces, roughly in order of near-term value:

- personal assistant use
- medical scribing
- medical summarization
- longitudinal personal health records
- physician-facing patient-record review
- evidence-grounded clinical documentation
- patient-facing explanation

Later: broader reasoning, mathematics, graphics, multimodality, tool use.

**Optimization target.**

> maximum **verified** capability per unit of compute, latency, memory, and cost.

"Verified" is load-bearing. An unverified answer has negative value in the
medical surfaces, so raw capability per FLOP is the wrong objective function.

**Size posture.** A locally runnable model in the **1–3B** range would be
excellent; **4–7B** is acceptable where the capability/efficiency tradeoff
justifies it. **Do not optimize around parameter count alone** — this program has
already measured that parameter count is not the explanatory variable it looks
like (§23: 159M own-stack reads 16.9 where 160M Pythia reads 3.5, and the
capability-floor result is confounded with tokenizer context fit, not size).

## 3. Scientific principles

1. **Name the instrument and its measured bottleneck before proposing a
   mechanism.** Different lines have different bottlenecks; a mechanism aimed at
   the wrong one cannot inform.
2. **Every architectural claim survives a pre-registered experiment or it is not
   a claim.** Decision rules fixed before the run; honest-FAIL reported.
3. **A manipulation must be checked, not assumed.** State the invariance the
   manipulation must preserve, and check it *before* reading the primary
   endpoint. E-DELIMIT arm B is the standing example of why (§23).
4. **A VOID is not a negative result.** An instrument that failed carries no
   information about the hypothesis.
5. **Contrast hygiene.** A contrast varies exactly one thing; "distinct" is not
   "equivalent". See `docs/RUNBOOK_contrast_hygiene.md` (R1–R5), every rule of
   which was earned by a defect that actually occurred here.
6. **Physics-inspired mechanisms receive no special status.** Neither do
   biological metaphors. A mechanism earns its place by a measured effect.
7. **Smallest sufficient tool.** Do not use a learned component where a
   deterministic one is correct, and do not use a large model where a small one
   plus a verifier is sufficient.
8. **Do not rewrite history.** Exploratory results never become confirmatory
   evidence retroactively. Gaps are recorded, not silently corrected.

## 4. Ontology

Normative: `nano/ontology.py`. The universal primitives:

```
Observation · Source · Evidence · Entity · Event · Relation · Claim · State
Change · Time · Uncertainty · Provenance · Belief · Hypothesis · Goal
Constraint · Action · Tool · Decision · Memory · Knowledge · Artifact
Verification · Feedback
```

The set is **open by construction**. Adding a primitive is ordinary work; a
primitive carrying two meanings is a defect.

**The invariant chain** (`CANONICAL_CHAIN`, designated INVARIANT):

```
WORLD → OBSERVATION → EVIDENCE → REPRESENTATION → STATE/BELIEF → REASONING → ARTIFACT
```

The direction never reverses. And (`NANO_INVARIANT`):

> No generated artifact is the canonical representation of reality. A note is not
> the state; a summary is not memory; a timeline is not the ledger; a chart is
> not the data; an answer is not the world model; a latent state is not evidence;
> a prediction is not a historical fact.

## 5. Epistemic contract

Two axes, kept orthogonal. Neither may be collapsed.

**How a proposition came to be held:**

```
OBSERVED ≠ DERIVED ≠ INFERRED ≠ HYPOTHESIZED ≠ PREDICTED ≠ SIMULATED
```

**How it currently stands against evidence:**

```
SUPPORTED ≠ CONFLICTING ≠ MISSING ≠ STALE ≠ SUPERSEDED ≠ UNVERIFIABLE
```

Hard rule, inherited from the Fabric slice and still binding:

> **¬Found(x) ⇏ ¬x.** Absence of evidence is never encoded as evidence of
> absence. MISSING and CONFLICTING are distinct states, and both are distinct
> from "not mentioned."

`nano/contracts.py::EpistemicStatus` implements the first axis; the second is
carried by verification state on the claim.

## 6. World model

**Hybrid, not reducible to an LLM and not reducible to a knowledge graph.** Six
representations, each because a query class needs it:

| Representation | Answers |
|---|---|
| **Event ledger** | append-only truth of what was observed, and when |
| **Entity / relation graph** | who and what, and how they connect |
| **Latent state** | learned representation for retrieval and generation |
| **Temporal state projections** | what was true at time *t* |
| **Belief / evidence graph** | what we hold, on what support, at what confidence |
| **Dependency lineage** | what a conclusion rests on, so it can be invalidated |

State is a **projection over the ledger**, never a mutable primary record
(`nano/test_nano_clin_001.py::test_state_is_a_rebuildable_projection`).

## 7. Time / state model

Bitemporal at minimum, and the distinction is a capability, not bookkeeping:

- **event time** — when it happened
- **documentation time** — when it was written down
- **discovery time** — when we learned it
- **system time** — when we recorded it

The system must answer both:

- *"What did we know then?"* — reconstruct the belief state at a past instant
- *"What do we now believe happened then?"* — current beliefs about a past instant

Time precision is preserved and never invented
(`nano/contracts.py::TemporalExtent.__post_init__` raises rather than fabricate
an exact date from an approximate one). New evidence appends; history is not
mutated (`STA-VERSION`).

## 8. Memory

Tiers: working · encounter · episodic · longitudinal · semantic · procedural.

Currently **ABSENT** in implementation (`MEM-HIER`, `MEM-SELECT` in
`docs/NANO_CAPABILITY_SPEC.md`) — this section is specification, not description.

Standing rule, inherited and unchanged:

> Generated statements never become persistent memory without
> classify → provenance → verify → contradiction-check → dedupe → scope →
> expiry → commit.

Retrieval into memory scores on semantic + lexical + graph + temporal +
authority + verification − contamination. Deciding *what is worth remembering*
is itself a capability, not a side effect of generation.

## 9. Cognitive kernel

Normative: `nano/kernel.py`.

**The kernel is deterministic.** It owns identity, persistence, state,
WorkSlices, dependencies, budgets, permissions, provenance, tool invocation,
schemas, events, checkpoints, retries, rollback, memory addressing, and
verification states.

The kernel does not think. It decides *what is allowed*, *what is owed*, and
*what has been verified*.

## 10. WorkSlice

The fundamental unit of cognition. One goal, one bounded piece of work, one
auditable trail.

```
Goal
 → reconstruct relevant state
 → identify gaps / uncertainty
 → generate hypotheses / approaches
 → estimate action value
 → compile work graph
 → execute
 → observe
 → verify
 → update world / beliefs
 → replan or stop
```

A WorkSlice carries: objective · state · assumptions · unknowns · hypotheses ·
constraints · evidence requirements · candidate actions · capabilities · tools ·
budget · risk · execution graph · artifacts · evidence · decisions · stop
conditions.

Two properties make it worth the machinery: it is **resumable** (state is
explicit, not in a context window) and it is **auditable** (every decision names
its evidence). Both are prerequisites for the medical surfaces.

## 11. Capability fabric

Three ownership classes, and the boundaries are the design.

**Deterministic substrate** — see §9.

**Learned components** own interpretation, representation, hypothesis
generation, decomposition, retrieval planning, strategy selection, ambiguous
judgment, tool-selection *proposals*, synthesis, and communication.

**Specialized systems** own arithmetic, statistics, SQL and database work, graph
algorithms, symbolic mathematics, SAT/SMT, Lean, rendering, OCR, image and
signal processing, and other deterministic computation.

> A learned component that proposes a tool call is doing its job. A learned
> component that *performs* arithmetic instead of delegating it is a design
> failure, not a capability gap.

**Selection rule.** The fabric chooses the **smallest sufficient combination** of
model + memory + retrieval + tool + verifier for the task. Routing starts
deterministic (rules plus classifier on expected utility); learned routing is a
hypothesis (§20), not a starting point.

## 12. Perception

Text first; audio, tables, images and signals plug in **without redefining
evidence** — the `Modality` set is extensible by construction, and an evidence
span must locate a claim in its source modality-independently
(`nano/contracts.py::EvidenceSpanV2`, `EVD-LOCATE`).

Perception records source identity, authorship, source type and bitemporal times
at ingest (`SRC-PROV`), and distinguishes measured / observed / reported /
inferred at ingest (`SRC-EPIST`). A patient report is not promoted to a
measurement.

## 13. Retrieval

Retrieval is a **routed** capability, not a single index: semantic · exact ·
temporal · graph · episode. Currently ABSENT (`RET-ROUTER`).

Two measured constraints from this program's own record:

1. Retrieval and delimitation are **separable failures** and must be measured
   separately — the span-port line locates 97/120 and delimits 2 (§23).
2. Any manipulation that claims to hold retrieval fixed must **prove** it with a
   LOCATED-invariance check before its primary endpoint is read.

## 14. Reasoning

Modes: temporal (before/after/during/overlap/recurrence), causal (documented
rationale vs temporal association vs inferred cause — never conflated), and
metacognitive (what is known / unknown / conflicting / needed, as typed machine
state, not prose).

Reasoning consumes state and evidence; it does not consume its own prior
artifacts as if they were evidence (§4 invariant).

## 15. Verification

Verification is not a post-hoc filter; it is the epistemic control plane.

> **trustworthiness = generation + verification + abstention + review routing**

- Every factual claim is verified against evidence (`VRF-CLAIM`, IMPLEMENTED).
- **Abstention is a first-class outcome** (`UNC-ABSTAIN`, IMPLEMENTED), not a
  failure mode.
- Selective prediction reports **both** presented risk and review load. A gate on
  presented risk alone is gameable by abstaining on everything.
- Verification operates on **both** the semantic artifact and the **rendered**
  artifact (§16). A correct chart specification rendered into a misleading image
  is an unverified output.
- Adversarial self-attack before emission (`VRF-ADVERSARIAL`) is ABSENT and
  wanted.

## 16. Artifact compilation

```
WorldState + Goal + Audience → ArtifactIR → representation compiler
```

Outputs: prose · summaries · medical notes · timelines · tables · charts ·
diagrams · JSON/FHIR · speech · presentations · proofs · tool calls.

One state, many artifacts. Same facts, different audience — **without changing
the underlying facts** (`GEN-AUDIENCE`, `HUM-AUDIENCE`, both ABSENT). Charts and
timelines compile from state, never from prose (`VIS-FROMSTATE`).

## 17. Identity and authority

First-class, and a **capability** rather than a security wrapper.

The system distinguishes at minimum: patient identity · source identity ·
clinician/actor identity · authorization · provenance · observation vs inference
· temporal validity · conflicting records · stale/superseded information ·
derived conclusions and their evidence.

> **A correct fact attached to the wrong patient is an intelligence failure, not
> merely a security failure.**

Patient isolation is enforced in the **type system**
(`nano/test_nano_clin_001.py::test_no_cross_patient_contamination`), not by
convention.

Two modes, and their authority boundaries do not blur:

| Mode | Question | Authority |
|---|---|---|
| **Patient** | "Help me understand my complete health history" | own record; explanation; no documentation authorship |
| **Clinician** | "Help me understand this patient's record and produce verified documentation" | delegated record access; documentation authorship under review |

Human review is the action boundary (`SAF-ACTION`).

## 18. Dependency and invalidation

Layer `XIII`, normative in `nano/dependency.py` (`DependencyGraph.invalidate`,
`recompute_order`).

Every derived conclusion records what it rests on. When a source is corrected,
superseded or withdrawn, everything downstream is invalidated and scheduled for
recompute — **not silently left standing**.

Known gap: the graph, invalidation and recompute ordering exist and are tested;
the recompute step is **not yet wired to a producer** (`LRN-CORRECTION`,
PARTIAL). Until it is, invalidation marks staleness without repairing it.

## 19. Learning hierarchy

Normative: `nano/architecture.py::LearningLevel`.

| Level | Timescale |
|---|---|
| **L0** | inference-time adaptation |
| **L1** | working-state updates |
| **L2** | persistent memory |
| **L3** | procedural / skill learning |
| **L4** | adapters / post-training |
| **L5** | base-model retraining |
| **L6** | architecture evolution |

> **Corrections do not automatically become training data.** Information crosses
> upward only through validation gates. A human correction updates L1/L2
> immediately; reaching L4 or L5 requires the correction to have survived
> verification, deduplication, and contradiction-checking, in volume.

Fast loop touches state; slow loop touches weights, offline (`LRN-LOOPS`).

## 20. Neural architecture research program

**No architecture is committed.** DMLA is a hypothesis. So is the transformer.
`NEURAL_CANDIDATES` in `nano/architecture.py` enumerates mechanisms so an
experiment can *name* one; membership implies nothing measured.

Candidate mechanisms: multiscale representations · persistent latent state ·
recurrent/iterative refinement · retrieval · structured memory · adaptive
computation · sparse specialists · energy/constraint dynamics · early exits ·
modality experts · retrieval-conditioned computation.

**Every proposed mechanism supplies all six before it is an experiment:**

```
hypothesis → baseline → expected benefit → cost → falsifier → experiment
```

**Explicitly not approved:**

- The **A0→A7 ladder as written**. It is a sequence of mechanisms with no
  instrument named per rung and no bottleneck each rung addresses.
- The **5–10M mechanism-testing tier as automatic**. The 30M instrument already
  failed its capability floor; shrinking it may reduce discrimination further,
  and a mechanism test that cannot discriminate is not cheap, it is worthless.
  A smaller tier becomes appropriate only once an instrument at that scale is
  shown to discriminate.
- **"Build DMLA" / "build an MoE" / "use physics"** as instructions. These are
  not experiments.

## 21. Capability lattice

Normative: `docs/NANO_CAPABILITY_SPEC.md` (50 capabilities, 26 domains) and
`nano/capabilities.py`. Status at `09621fe`: **IMPLEMENTED 12 · PARTIAL 14 ·
PROPOSED 6 · ABSENT 18**.

Proving stages (`ProvingStage`) — the order in which capability is *proven*,
never a layer ordering:

```
A evidence → B state → C memory → D retrieval → E reasoning
→ F artifacts → G multimodal → H general cognition → I continual
```

The lattice is **not** the architecture. The architecture is the invariant
substrate (§§4–19); the lattice describes what has been earned.

## 22. Evaluation strategy

- **Instrument-first.** An evaluation names the instrument, the bottleneck it
  probes, and what a null looks like.
- **Coverage before accuracy.** An instrument with zero coverage yields
  `INVALID_NO_SIGNAL`, not a null — the distinction that the D2.3 guard enforces
  and that wave-1 native30 would otherwise have laundered into six false nulls.
- **Guard against tautological metrics.** Constrained candidate selection makes
  span metrics unfailable by construction
  (`span_metrics_are_tautological: True`); such cells are reported with the flag
  attached.
- **Baselines are mandatory and adversarial.** A parrot / majority-class /
  constant baseline on the *same instrument*. E-DELIMIT's index-0 parrot at
  LOCATED 19.2% is what made arm B's 25.0% legible.
- **Multi-seed or it did not happen.** An effect smaller than the seed spread is
  not an architecture effect.
- **Report presented risk and review load together.**
- **Contrast hygiene R1–R5** (`docs/RUNBOOK_contrast_hygiene.md`) applies to
  every comparison.

## 23. Current experimental record

Full ledger: [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md). The load-bearing facts:

**Span-port line — the bottleneck is delimitation, not retrieval.** With all leak
channels closed the model selects the correct turn for **97/120** gold-bearing
slots and delimits the gold span in **2**. All 95 non-exact located quotes are
over-extended; zero under-extended. Replicated (`e04b3016` → `38b12909`).

**E-DELIMIT is complete and did not refute H5.** Arm B is **VOID**: its
LOCATED-invariance precondition failed, retrieval collapsing **97/120 → 30/120**
against a constant-baseline 23/120. `asserted_grounded` 0/192 would have fired
the kill condition; it must not be read that way. **H5 is untested, not
weakened.** Arm C is secondary by pre-registration and is not evidence about H5.

**Scribe line — a different bottleneck.** The OOD copying gap survived curriculum
(Stage C), scale (Stage S) and architecture (Stage P). Suspects narrow to
retrieval/induction-circuit capacity, much larger scale, or the objective. **Do
not use a delimitation experiment to answer this, or a memory/reasoning
experiment to answer delimitation.**

**native30 line.** Arms do not separate (`NOT_SEPARATED`; effect 0.0133 below
seed spread 0.04). The capability floor fired **in its permissible form only**:
*30M at 1800 steps with a tokenizer that cannot fit 83% of eval prompts in its
context* is below the floor; 30M with a fitting tokenizer is **untested**. Wave 1
was a **false null** (non-causal decoder) and must not be banked. "Ran clean
under the integrity gate" is **PENDING** the `reval30_*_fixed_*` re-run.

**Breadth before specialization.** At matched parameter count a generally
pretrained model reads 3.5 ± 0.7 where the domain-native model reads 16.9 ± 1.7.
This is the strongest single argument against "train a novel small model from
scratch" as the default path — and it is SUPPORTED, not ESTABLISHED.

**Procedural.** The `ddb5ce6` authorization gap is recorded, not corrected.

## 24. Open hypotheses

See RESEARCH_STATUS § HYPOTHESES. In priority order by information value:

1. **Is the native30 capability floor a tokenizer-context artifact?** The single
   largest ambiguity in the record, and an existing asset resolves it.
2. **H5 — is delimitation a representational limit?** Untested; needs the
   two-stage design.
3. **What accounts for the scribe OOD gap** after curriculum, scale and
   architecture all failed?
4. **Monolith vs modular Nano-System** — open in both directions.
5. **Does any `NEURAL_CANDIDATES` mechanism buy verified capability per FLOP?**
   No mechanism has an instrument yet.

## 25. Approved next experiments

**None.** No experiment is authorized. Three candidates are ranked in
RESEARCH_STATUS § NEXT CANDIDATE EXPERIMENTS; each requires its own
pre-registration **and** experiment-scoped authorization before launch.

Every future experiment states, before it runs:

```
instrument · bottleneck · hypothesis · manipulation · invariance requirement
preregistered decision rule · authorization · compute/cost · artifact/provenance
```

The **invariance requirement** is the newest of these and is not optional. It
exists because E-DELIMIT arm B satisfied every other field and still produced
nothing.

Its concrete form for output-format manipulations is **R8** in
`docs/RUNBOOK_contrast_hygiene.md` — registered after the arm-B VOID and stated
in `RESEARCH_STATUS.md`. The lesson generalizes past the span-port line: arm B's
*format-feasibility* gate would have passed at ~96%, because the model used the
menu format perfectly well. **Format compliance and task preservation are
different properties.** Name the capacity the manipulation is supposed to leave
alone, bound it numerically, and make breaching it a VOID condition.

## 26. Future roadmap

Sequenced by what each stage would make falsifiable, not by ambition.

**Near** — resolve the tokenizer confound; establish an instrument that
discriminates; close the PENDING revalidation; wire dependency recompute to a
producer.

**Middle** — memory tiers (currently ABSENT) with validated-write gating;
retrieval routing; conditional delimitation measured properly; adapters at L4
once corrections accumulate through the validation gates.

**Far** — modality expansion through the existing evidence interfaces;
adversarial self-verification; audience-aware artifact compilation; the
Nano-System decomposition (Core / Reason / Memory / Verify / Route / Vision /
Audio) *if* experiments show modularity beats a monolith at equal verified
capability per unit cost — and the converse if they do not.

**Development posture — Mac-first.** Experiments are developed and run primarily
on local Apple Silicon: reproducible locally, cheap enough to iterate rapidly,
MPS/Metal compatible where appropriate, instrumented for CPU/GPU/memory,
independently reproducible, and able to scale to larger compute later. The four
E-DELIMIT runs cost \$0 and took under six minutes in total. **Do not introduce
infrastructure because it is fashionable.**

---

## Standing rule

> Do not code the future architecture now. Consolidate the specification and the
> evidence, then identify the **smallest next experiment that discriminates
> between competing explanations**. Every experiment must make the architecture
> more falsifiable, not merely more elaborate.

The goal is a highly capable 1–7B-class local Nano system. Every architectural
component is earned experimentally or it is not in the system.
