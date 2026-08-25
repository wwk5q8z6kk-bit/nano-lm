# Nano vNext — Research Constitution

**Frozen 2026-08-25** as *architecture intent*, per §L step 1.

> We are freezing Nano's research constitution, not freezing a final theory of
> intelligence.

This document is strong enough to guide implementation now and deliberately
open to new primitives, representations, algorithms, neural mechanisms, learning
methods, modalities, tools, and forms of cognition discovered later.

---

## Authority — read this before treating anything below as binding

This document is **intent**, and intent ranks below measurement.

| Document | Owns |
|---|---|
| `docs/PROJECT_AUTHORITY.md` | how conflicts between sources are resolved |
| **this file** | architecture intent + research constitution |
| `docs/SYSTEM_ARCHITECTURE.md` | the *current realisation* of that intent |
| `docs/NANO_CANONICAL_BRIEFING.md` | operational onboarding |
| `nano/*.py` + tests | what is actually built |

Per `PROJECT_AUTHORITY.md`, **no aspirational document may overrule measured
evidence** — including this one. Per §XLIII below, architecture plans must never
silently become scientific claims. If this file and a measured result disagree,
the result wins and this file gets edited.

---

## 0. Canonical definition

Nano is a **medical-first, general-core research program for compact, reliable,
stateful grounded intelligence.**

It develops systems that actively observe partially observed changing worlds;
construct, maintain, retrieve, transform and reason over evidence-grounded
internal models of those worlds; recognise uncertainty and missing information;
acquire information or computation when useful; plan and act through learned and
deterministic capabilities; verify beliefs, actions and outputs; communicate
through task-appropriate artifacts; and improve from validated experience.

Medicine is Nano's first major real-world domain and one of its hardest proving
grounds. **Medicine is not the boundary of the architecture.**

Nano is **not** fundamentally an LLM, a chatbot, a scribe, a knowledge graph, a
RAG pipeline, an agent loop, a workflow engine, a collection of tools, a fixed
ontology, a Transformer, an MoE, or a parameter count. Any of those may become
*components*.

> The intelligence is the evolving internal model of the world and the machinery
> that operates on it. The generated text is only one possible artifact.

---

## I. Five specifications that must never be conflated

Most previous conceptual confusion came from mixing these.

1. **Ontology** — what Nano can represent.
2. **Cognitive architecture** — how it turns what it knows into work.
3. **Neural architecture** — which learned mechanisms implement which function.
4. **Learning architecture** — how capabilities are acquired and improved.
5. **Capability progression** — what integrated abilities we prove, in order.

The ladder (P1–P9) belongs to (5). **A ladder is not an architecture.**

### Ontology expansion law

Current primitives are a starting set, not a ceiling:

> Expand when representation demands it; do not collapse different jobs merely
> to preserve a small ontology.

Candidate future primitives — Process, Mechanism, Distribution, Program,
SimulationState, Policy, Intent, Counterfactual, Strategy, Skill, LatentObject,
CausalMechanism, Concept, Schema, SpatialField, DynamicalSystem — are added when
evidence shows that forcing them into existing objects loses structure.

### Learning timescales

| Loop | Changes | Evidence required |
|---|---|---|
| Fast | working state, retrieval, episodic memory, beliefs, plans | observation |
| Intermediate | skills, indexes, routing, semantic memory, tool policy | validated feedback |
| Slow | weights, architectures, tokenizers, training distributions | controlled experiment |

**Slow learning requires stronger evidence than fast state updates.** Direct
observation → foundation-weight update is explicitly *not* the architecture
(§XXXI).

---

## II. The loop

```
                         REAL WORLD
                             │
                             ▼
                 UNIVERSAL OBSERVATION          passive + active
                             │
                             ▼
                    EVIDENCE / GROUNDING
                             │
                             ▼
                 REPRESENTATION SYSTEM          explicit + latent + learned
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
              MEMORY                  KNOWLEDGE
                 └───────────┬───────────┘
                             ▼
                    EVOLVING WORLD MODEL        temporal / causal / epistemic
                             │
                             ▼
                         WORKSLICE               goal + constraints + state
                             │
                             ▼
                UNCERTAINTY / INFORMATION NEED
                             │
                ┌────────────┼────────────┐
                ▼            ▼            ▼
             RETRIEVE      REASON        PLAN
                └────────────┼────────────┘
                             ▼
                   CAPABILITY ROUTING
                             │
             ┌───────────────┼───────────────┐
             ▼               ▼               ▼
           MODEL         ALGORITHM          TOOL
             └───────────────┼───────────────┘
                             ▼
                         EXECUTION
                             ▼
                      NEW OBSERVATION
                             ▼
                  STATE / BELIEF UPDATE
                             ▼
                 DEPENDENCY INVALIDATION
                             ▼
                    ARTIFACT / ACTION IR
                             ▼
                        VERIFICATION
               ┌─────────────┼─────────────┐
               ▼             ▼             ▼
            PRESENT        ABSTAIN        REVIEW
               ▼
           COMMUNICATE → FEEDBACK → VALIDATED ADAPTATION ──↺
```

Cross-cutting everything: **identity · authority · provenance · uncertainty ·
dependency · observability · security · resource control · reproducibility ·
measurement science · human oversight.**

This is a functional decomposition, **not** an assertion that intelligence
permanently consists of these boxes. If experiments reveal a missing function,
Nano expands (§XXXIII).

---

## III. Epistemic architecture — three orthogonal axes

Never collapse these into one enum or a single confidence scalar.

| Axis | Values |
|---|---|
| **Derivation** — how Nano produced it | OBSERVED · DERIVED · INFERRED · RECONSTRUCTED · HYPOTHESIZED · PREDICTED · SIMULATED |
| **Status** — what standing it has | SUPPORTED · CONFLICTING · MISSING · UNKNOWN · STALE · SUPERSEDED · UNVERIFIABLE |
| **Authority** — who vouches for it | PATIENT_REPORTED · CLINICIAN_ASSERTED · DIRECT_MEASUREMENT · DOCUMENTED_EXTERNAL · MODEL_GENERATED · TOOL_GENERATED · HUMAN_CORRECTION |

> **Nano must never silently convert generated output into world truth.**

---

## IV. Output decision

At every meaningful output boundary:

```
candidate result → VERIFY → PRESENT | ABSTAIN | REVIEW
```

This is more fundamental than "hallucination reduction". **Reliability includes
knowing when not to speak.**

---

## V. Medicine and generality

```
NANO CORE  (observation · evidence · state · time · memory · knowledge ·
            cognition · tools · verification · learning)
     │
     ▼
DOMAIN PACKS — Nano Clinical · Mathematics · Software · Science · Graphics · …
```

Nano Clinical *specialises* general objects: Entity → Patient/Medication;
Event → Admission/Procedure; Observation → lab/note/imaging; Artifact →
note/timeline/handoff.

**The Core must not contain the concept "scribe."** That is a load-bearing
falsifier of generality — if it appears there, generality was lost.

NanoScribe is not a discarded predecessor. It is the first demanding *interface*
over Nano Clinical, and its research (evidence transport, selective prediction,
held-out copying, tokenizer studies, measurement integrity) is retained.

### Debugging order

```
GENERAL MECHANISM → SYNTHETIC CONTROLLED WORLD → DOMAIN BENCHMARK
                  → MEDICAL STRESS TEST → REAL-WORLD UTILITY
```

Medicine is important throughout **without being the first debugging
environment for every subsystem** — a synthetic world has exact ground truth.

---

## VI. Measurement science is science, not overhead

Every major experiment carries three layers:

1. **Scientific hypothesis** — what mechanism is being tested?
2. **Measurement hypothesis** — what result distinguishes the explanations?
3. **Instrument-validity hypothesis** — what shows the pipeline can *detect
   failure* rather than launder it?

```
VALIDATE INSTRUMENT → VALIDATE CONTROL → RUN → TEST PRE-REGISTERED CRITERIA
                    → INTERPRET
```

Never: `RUN → SEE INTERESTING NUMBER → CONSTRUCT STORY`.

The native30 wave earned this the hard way — truncated targets, unmasked prompt
positions, bidirectional attention under next-token training, arms that were
scalar rescalings of one loss, zero-coverage runs reported as NOT_SEPARATED,
tests writing into real checkpoints, char-level tokenization pushing most eval
prompts over context. Any one corrupts interpretation. Together they establish:

> **Measurement infrastructure is itself an experimental object. A high metric
> cannot validate the instrument that produced it.**

Precommitment is proven useful, not ceremonial: on the causal-fix wave, seed 0
showed control 0/6 vs bottleneck 6/6 — a compelling mechanistic story that seeds
1 and 2 erased. Use it where the result space admits post-hoc storytelling; do
not apply it bureaucratically to every exploratory probe.

---

## VII. Three levels of intelligence, reported separately

- **Parametric** — what the weights do unaided.
- **System** — weights + memory + state + retrieval + algorithms + tools +
  specialists + verification.
- **Autonomous** — what completes reliably without rescue.

A 100M controller with excellent retrieval, deterministic algorithms, memory,
tools and verification may have far lower parametric capability and greater
*system* capability on a useful WorkSlice than a larger standalone LM. **That is
not cheating. It is one of Nano's central hypotheses.**

### Objective

```
        Verified Useful Capability
max  ─────────────────────────────────────────────────────────────
     Active Compute + Memory + Latency + Energy + Money + Human Review
```

subject to reliability · grounding · safety · generalization · privacy ·
reproducibility.

**Parameter count is one variable in the denominator, not the definition of
compactness.**

---

## VIII. Expansion protocol

```
NEW REQUIREMENT
   → can existing representation express it without loss?
        YES → use it
        NO  → propose primitive / representation / relation / module / mechanism
              → define interface → define invariants → build falsifier
              → controlled benchmark → compare alternatives
              → KEEP | REVISE | REJECT
```

This is how Nano evolves without another wholesale conceptual reset every time a
new insight appears.

---

## IX. What is frozen, and what is not

**Frozen (the constitution):**

separation of concerns · evidence before belief · state without history
destruction · explicit epistemic status · provenance · dependency/invalidation ·
measurement before claims · mechanisms chosen by experiment · capabilities
measured independently · minimum sufficient computation · human-verifiable
outputs · architecture extensibility.

**Explicitly not frozen:**

number of primitives, planes, modules or models · model size · Transformer vs
SSM vs recurrence · dense vs MoE · tokenizer · memory implementation · graph
representation · retrieval algorithm · reasoning architecture · learning method ·
routing mechanism · modality architecture · DomainPack structure.

Even the frozen principles may be challenged by strong evidence. They constitute
the *current* research constitution.

---

## X. The laws

> Evidence is not belief.
> Knowledge is not observation.
> Generated output is not world truth.
> Current state is not history.
> Absence of evidence is not evidence of absence.
> Confidence is not provenance.
> Retrieval is not memory.
> Context is not persistent state.
> The language model is a capability, not the system.
> A benchmark is not a capability.
> A ladder is not an architecture.
> A mechanism model is not a capability model.
> No data is not a null.
> A high metric does not validate its own instrument.
> A single seed is not a mechanism.
> Parameter count is not system intelligence.
> Domain specialization must not define the general Core.
> The architecture may expand when reality demands a new abstraction.
>
> **Nano should preserve distinctions until evidence demonstrates that
> collapsing them loses nothing important.**

---

## XI. The research question

> What is the smallest and most efficient artificial intelligence system that
> can actively observe a changing, partially observed world; construct and
> continuously revise an evidence-grounded internal model of it; preserve useful
> information across time; identify uncertainty and acquire missing information;
> retrieve, reason, plan, compute and act using the smallest sufficient mixture
> of learned and deterministic capabilities; verify its beliefs, actions and
> outputs; communicate appropriately; and improve from validated experience?

And beneath it:

> Can such a persistent, grounded, modular cognitive architecture provide
> greater verified useful capability per unit of compute, memory, latency,
> energy, money and human review than increasingly relying on a monolithic
> context-centered language model?

---

## XII. Resumption ledger — §L, with live status

Status is measured, not asserted. Recheck: `pytest nano` and
`scripts/run_nano_slw_001.py --sweep`.

| # | Step | Status | Evidence |
|---|---|---|---|
| 1 | Freeze this constitution as architecture intent | **done** | this file |
| 2 | Preserve the scientific ledger and defect index | **done** | `artifacts/DEFECT_INDEX.md`, `papers/DEFECT_NATIVE_CAUSAL_MASK.md`, `papers/METHODS_ADVERSARIAL_INSTRUMENTATION.md`, `artifacts/campaign/reval_results/FALSE_NULL_DIAGNOSIS.md`, `TOKENIZER_CONTEXT_CONFOUND.md`, `docs/FAILURE_TO_ARCHITECTURE.md` — all tracked |
| 3 | Keep the current NanoState contracts | **done** | `nano/contracts.py`, unchanged in substance |
| 4 | Complete dependency → recomputation | **done** | `LRN-CORRECTION` IMPLEMENTED; work list taken *from* `recompute_order()`; 10/10 obligations discharged |
| 5 | First executable WorkSlice / InformationNeed / Capability / Verification contracts | **done** | `nano/runtime.py`; `nano/needs.py`; `MTA-EPISTEMIC` IMPLEMENTED |
| 6 | Build DomainPack-0 synthetic dynamical world | **done** | NANO-SLW-001 — 176 entities, 3 typed relation families, six corruption modes, hidden state, corrections, supersession |
| 7 | State-maintenance and invalidation baselines | **done** | recompute ratio 0.279; invalidation P/R 1.000; branch isolation 1.000 |
| 8 | Test retrieval versus persistent state | **next** | needs the executor, which now exists — the two arms are the runtime answering from maintained state vs from retrieved raw observations |
| 9 | Test BPE/representation before another scale claim | **recorded, not run** | `trajectory/PREREG_tokenizer_bpe_vs_char.md` |
| 10 | Mechanism-model research separate from capability-model | standing rule | §XXV of the source directive |
| 11–17 | learned components behind stable interfaces → Nano Clinical → longitudinal reconstruction → multimodality → 1,000-page benchmark → system-vs-model comparison → expand on evidence | **not started** | — |

Interventions in DomainPack-0 (§XXXVII) are **not** built. The world has
corrections, supersession and hidden state; it does not yet have actions that
*cause* world change. Recorded here rather than quietly folded into step 6.

Each iteration follows:

```
UNDERSTAND → HYPOTHESIZE → BUILD → VALIDATE THE INSTRUMENT → TEST
           → ATTACK THE RESULT → LEARN → INTEGRATE OR REJECT → REPEAT
```
