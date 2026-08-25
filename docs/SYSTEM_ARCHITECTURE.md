# System Architecture

## Nano Core (domain-general)

```text
                    NANO CORE
              domain-general intelligence
                        │
       ┌────────────────┼────────────────┐
       │                │                │
   Representation     Memory          Evidence
       │                │                │
       └──────────┬─────┴──────┬────────┘
                  │            │
                State       Retrieval
                  │            │
             Temporality    Salience
                  │            │
                  └──────┬─────┘
                         │
                     Synthesis
                         │
                       Plan
                         │
                    Generation
                         │
                   Verification
                         │
           ┌─────────────┼─────────────┐
           │             │             │
        Present        Abstain        Review
```

### Core primitives

```text
source representation · segmentation · entity/event representation
evidence spans · state · temporality · uncertainty · contradiction
memory · retrieval · salience · hierarchical compression
planning · generation · verification · selective prediction · human review
```

### Core primitives — pending extensions (additive; the list above is unchanged)

The primitives above are the committed Core. These are the extensions the
longitudinal clinical-intelligence target requires and the Core does **not**
currently name. They are listed separately so nothing above is silently
reinterpreted, and each carries the level it serves (see
`CAPABILITY_LADDER.md`).

```text
multimodal source representation  (L0)  image · signal · audio · scanned doc
typed time                        (L2)  event / documentation / discovery / onset
                                        / relative / uncertain time as distinct axes
evidence graph                    (L3)  nodes + relations over spans, not spans alone
longitudinal state                (L4)  cross-encounter patient state with carry-forward
trajectory representation         (L7)  state-over-time as a first-class object
```

**Why these five and not the whole level list.** `evidence spans`, `state`,
`temporality`, `uncertainty`, `contradiction`, `memory`, `retrieval`,
`salience`, `hierarchical compression`, `verification`, `selective prediction`
and `human review` are *already* Core primitives. The clinical target does not
need them added — it needs them **built**. Only the five above are genuinely
missing from the design.

**Two are refinements, not additions.** `typed time` sharpens the existing
`temporality` primitive; `evidence graph` builds over the existing
`evidence spans`. They are listed here because the distinction is load-bearing:
a span answers *where did this come from*, a graph answers *how do these
relate* — and conflating them is how a provenance layer degrades into a
citation decoration.

**Do not add these to the Core diagram until a program owns one.** An unowned
primitive is a claim about the future, and this file is Layer 2.

### DomainPack boundary

**Nano Core** holds domain-general interfaces. **DomainPack** holds:

```text
domain schema · ontology mappings · normalization rules
domain negation/temporality semantics · evaluation sets · transformations
```

Do not hardwire medicine into every Core interface. Do not generalize away medically necessary semantics.

**DomainPack-0 — synthetic world (Core development harness).** Medicine is
DomainPack #1 for the *product*. It is the wrong *first* harness for Core
development: when a run fails against real records you cannot separate an
architecture fault from an extraction fault, an ontology fault, or bad ground
truth. A generated world whose ground truth is the generator's own state
isolates the Core.

This is the existing Core/DomainPack boundary used as intended, not a detour
around the capability ladder. The Core interfaces are identical under both
packs — which is itself the test: **if the Core needs different interfaces for
the synthetic pack and the medical pack, the boundary is drawn wrong.**

The machinery already exists — `nanoscribe/native/corpus/` generates synthetic
cases with gold, dedup, axis-coverage floors and leakage gates. DomainPack-0 is
an extension of it, not new infrastructure.

**Isolate capabilities, then compose.** Build and test perception,
representation, memory, world model, retrieval, reasoning, tools, verification
and generation as separable experiments with their own gates. One
all-components-at-once experiment cannot attribute a gain to a component — and
this program has already paid for that lesson: the causalfix arm split produced
a mechanistic story attached to a single-seed artifact
(`trajectory/PREREG_causalfix_wave_arm_split.md`). Component isolation is what
makes an attribution survivable.

## Storage and memory separation (design; nothing here is built)

**Polyglot by intent.** One store cannot serve these access patterns, and
forcing it is a known failure. Each row is a *role*, not a product choice.

```text
relational      exact structured clinical data
object store    original documents, images, audio — never destroyed
vector index    semantic retrieval
graph           relationships between facts
time-series     measurements and signals over time
ontology store  standardised concepts (SNOMED / LOINC / RxNorm / ICD / UCUM)
latent memory   learned representations
```

**The graph holds relationships, not content.** A node points at the source; it
does not contain the 500-page note. `graph node → source pointer → document
store → exact passage`. Violating this turns the graph into a second, lossy copy
of the record.

**Four memories that must never merge.** This is an invariant, not a layout:

```text
PATIENT      what happened to this patient
MEDICAL      what medicine knows (literature, guidelines, pharmacology)
INSTITUTIONAL what this organisation does (protocols, formulary, pathways)
MODEL        what the system learned in training
```

Merging PATIENT with MEDICAL is how "the literature says X" becomes "the patient
has X". The reasoning core *connects* them and reports which world each claim
came from.

## Hierarchical zoom (the context-window answer)

The response to a 1,000-page record is not a longer context. It is addressable
compression with a descent path back to source:

```text
L0 patient overview
L1 problem overview
L2 disease trajectory
L3 episode          (grouped events: onset → ED → ECG → troponin → dx → tx → f/u)
L4 event
L5 source document
L6 exact passage / measurement
```

Compression is only legitimate if every level can descend to L6. **Compression
without information destruction** is the requirement; a summary that cannot be
descended is a lossy rewrite.

## Architectural anti-patterns (explicit)

Recorded because each is locally reasonable and globally wrong:

```text
one giant context window as the memory
one graph containing everything, content included
one vector database standing in for memory
one model doing OCR + arithmetic + retrieval + reasoning + verification
a chatbot before a reconstruction engine
autonomous clinical decisions before verified documentation
live weight updates from unvalidated patient interactions
any generated claim that cannot be traced to evidence
```

The last is already enforced in code for the built slice (`fabric/` provenance
gates); the rest are design commitments, not measured properties.

## Deployment shape

```text
Nano Core + Medical DomainPack + P1  =  NanoScribe

Nano Core + Medical DomainPack + P1–P3
  =  longitudinal medical documentation intelligence (earned sequentially)

P4–P9  =  broader synthesis, reasoning, planning, action, adaptation
```

## Compute (training ≠ deployment)

```text
Local Apple Silicon / CPU  →  development, smoke, analysis, preprocess, eval, small experiments
RunPod (primary GPU)       →  training, adaptation, CUDA validation, scaling research
Deployment target          →  chosen independently (compact/local/private remains an axis)
```

RunPod is active established infrastructure; costly/confirmatory runs stay experiment-scoped. No PHI on cloud. See [ACTIVE_NOW.md](ACTIVE_NOW.md) and [infrastructure/RUNPOD.md](infrastructure/RUNPOD.md).

## Subsystems in this repository

| Component | Path | Role |
|-----------|------|------|
| **P1 NanoScribe** | `nanoscribe/` | Encounter v0, CandidateAtom adapters, harness, campaign ([subsystems/NANOSCRIBE.md](subsystems/NANOSCRIBE.md)) |
| Mechanism / compact models | `pretrain/`, `sft/`, `scribe/` | Integrated training experiments (historical + mechanism) |
| Model core (future) | `nano_ai/` | **Cross-branch — not yet integrated** — see [research/MODEL_RESEARCH_PROGRAM.md](research/MODEL_RESEARCH_PROGRAM.md) |
| Verification harness | `fabric/` | Typed claims, verifiers, abstention regression |
| Verified document intelligence | `wedge_v1/` | Local corpus Q&A with span evidence ([subsystems/WEDGE.md](subsystems/WEDGE.md)) |
| Campaign artifacts | `artifacts/campaign/` | Manifests, spend ledger, round summaries ([research/ACCELERATED_CAMPAIGN.md](research/ACCELERATED_CAMPAIGN.md)) |
| Experimental record | `trajectory/` | Result JSONs, prereg companions |

## Routing principle (from E1/E4 evidence)

```text
User task → classify → cheapest sufficient solver
  ├── deterministic parser / rules
  ├── retrieval / search
  ├── symbolic tool
  ├── compact specialized model
  └── larger teacher (ceilings, synth, judge — not default deploy)
→ verification → present | abstain | review
```

## P1 scribe pipeline (target)

```text
audio / transcript → immutable source → turns & speakers
→ clinical events → entities & values → state
→ temporality · uncertainty · contradictions → evidence spans
→ verified encounter record → note plan → coherent note
→ claim decomposition → independent verification → present | abstain | review
```

**Canonical truth object:** structured, evidence-grounded encounter representation.  
**Free-form note:** a rendering — not the primary truth object.

## P2/P3 future contract (design now, build later)

Encounter records should support references:

```text
entity_id · event_id · encounter_id · source_id
timestamp / interval · state transition
supersedes · contradicts · supports · derived_from
confidence · evidence spans
```

## Co-design checklist

For every subsystem ask:

1. What should software/retrieval/schemas/verifiers own?
2. What should a learned representation own?
3. What is the smallest sufficient solver today?

See [research/SYSTEM_RESEARCH_PROGRAM.md](research/SYSTEM_RESEARCH_PROGRAM.md).
