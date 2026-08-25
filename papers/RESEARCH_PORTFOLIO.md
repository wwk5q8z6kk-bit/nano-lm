# Research Portfolio

**Layer 3 — Research agenda.** Expansive. Non-evidential.  
**Adopted:** 2026-07-31  
**Constitution:** `LABORATORY_CONSTITUTION.md`


> **NONCLAIM** — Vision / portfolio only. Not evidence. Not authorization to build, ship, or deploy.
> Negative results falsify scoped hypotheses only. Cite `EVIDENCE_LEDGER.md` for what is measured.

> Dream extremely big. Then destroy hypotheses one by one.  
> This file lists **questions worth answering**, not results and not a build queue.

Status tags for each program (portfolio hygiene, not evidence):

| Tag | Meaning |
|-----|---------|
| `OPEN` | Worth pursuing; little/no decisive evidence yet |
| `ACTIVE_DESIGN` | Protocol/design work authorized (not execution) |
| `PARTIALLY_TOUCHED` | Some related measurements exist in the ledger |
| `FALSIFIED_HYPOTHESIS` | A *specific* hypothesis inside the program died; program remains |
| `PARKED` | Deprioritized, not refuted |

---

## Program A — Mechanistic understanding of held-value copying
**Status:** `PARTIALLY_TOUCHED` (behavior measured; mechanisms unidentified; E2 STOP)  
**Questions:** What internal computations support seen-value emission vs held-value failure? Geometry vs optimization vs early-stop vs module site?  
**Does not require:** NanoScribe, E1 revival.

## Program B — Generalization under distribution shift
**Status:** `PARTIALLY_TOUCHED` (held/seen, diversity, Pythia ladder)  
**Questions:** When do lexical, morphological, template, and domain shifts break exact emission? What transfers across fields/domains?

## Program C — Verification theory
**Status:** `PARTIALLY_TOUCHED` (Fabric slice under decidable R)  
**Questions:** Soundness/completeness tradeoffs; abstention calculus; when is a verifier an extractor in disguise; open-world limits.

## Program D — Calibration
**Status:** `OPEN`  
**Questions:** Can models or routers know when they will fail held-value emission? Selective prediction vs verify-and-abstain.

## Program E — Evidence theory / provenance
**Status:** `PARTIALLY_TOUCHED` (typed claims, spans in Fabric)  
**Questions:** Content-addressed evidence, contradiction states, epistemic status lattices, audit trails at scale.

## Program F — Semantic equivalence
**Status:** `PARTIALLY_TOUCHED` (E3: normalize fails; agent-rubric; clinician open)  
**Questions:** Exact vs normalized vs synonym vs clinical acceptability; IAA; ontology design without circular rescue.

## Program G — Retrieval versus generation
**Status:** `OPEN`  
**Questions:** When is copy-from-context / retrieval sufficient? When must generation invent structure rather than values?

## Program H — Classical vs LM hybrids
**Status:** `PARTIALLY_TOUCHED` (E1 classical win on closed world)  
**Questions:** Routing; when classical fails intrinsically; information-parity bakeoffs; maintenance-aware utilities.  
**Linked design:** R★ / E4 (`ACTIVE_DESIGN` protocol only).

## Program I — Structured reasoning
**Status:** `OPEN`  
**Questions:** Schemas, constraints, typed intermediate states, program-like reasoning traces for extraction and beyond.

## Program J — Tool use
**Status:** `OPEN`  
**Questions:** Search, calculators, DB lookup, validators as first-class actions with logged permissions and cost in \(U\).

## Program K — Long-context and persistent memory
**Status:** `ACTIVE_DESIGN` (raised 2026-08-25 — a decidable hypothesis now exists; execution NOT authorized)  
**Questions:** Episodic vs semantic vs user vs graph vs causal memory; write authorization; contamination; forgetting.

**K1 — the central architectural hypothesis.** Stated so it can die:

> A persistent, multiscale, evidence-grounded representation of a changing
> real-world system supports longitudinal reconstruction more accurately and at
> lower compute than a long-context autoregressive model reading raw sequences.

Baseline is **not** "raw LLM generation" (per the Kill/keep rule): it is a
long-context transformer over the raw record, at matched compute. Candidate is
multiscale encoding + persistent state + temporal graph + retrieval memory +
iterative inference with adaptive compute.

Comparison axes, all already measurable in this repo's idiom: extraction recall ·
temporal placement accuracy · entity-resolution accuracy · contradiction
detection · evidence-weighted recall (see `docs/EVALUATION_FRAMEWORK.md`) ·
provenance traceability · compute · latency · robustness to incremental new data.

**Why this program and not a new one.** K already owns persistent memory. The
hypothesis sharpens K rather than forking it; L (agents), J (tools) and G
(retrieval vs generation) supply components and are not absorbed.

**Falsifier.** If the long-context baseline matches the candidate on
evidence-weighted recall and temporal accuracy at equal compute, K1 is dead
regardless of architectural elegance.

**K1 is a compound hypothesis and must be decomposed before it is tested.** It
bundles persistent state, multiscale representation, temporal structure,
external memory, learned routing, adaptive compute, tool use and iterative
inference. A single experiment carrying all eight cannot attribute a gain to any
one of them — and an unattributable gain is how a seed artifact acquires a
mechanistic story (see `trajectory/PREREG_causalfix_wave_arm_split.md`, where
0/6-vs-6/6 on one seed looked like an architecture effect and was not).

Each component gets its own gate before composition:

| Component | Isolated question | Discriminator |
|---|---|---|
| persistent state | does carrying state beat re-reading? | next-state prediction — retrieval cannot fake it |
| multiscale representation | do learned abstractions beat one token scale? | performance at fixed compute |
| temporal structure | are before/after/during/overlap native? | temporal placement accuracy |
| external memory | does selective write beat write-everything? | recall at fixed memory budget |
| learned routing | does the model choose the right retrieval mode? | routing accuracy vs oracle |
| adaptive compute | does it spend more on harder questions? | compute vs question difficulty correlation |
| tool use | does it delegate rather than estimate? | arithmetic/lookup error rate |
| iterative inference | does another pass help? | accuracy vs iteration count, with a halting rule |

**Do not hard-code the hierarchy.** Levels are a structural bias, not a rule
table. A hand-written `sentence → event` mapping tests the rules, not the
architecture.

**First harness is DomainPack-0, not medicine** (`docs/SYSTEM_ARCHITECTURE.md`).
Medicine cannot separate an architecture fault from an extraction, ontology or
ground-truth fault.

**Standing caution.** The claim "do not optimize primarily for next-token
prediction" is a *hypothesis inside K1*, not a finding. The measured native track
is a next-token model, and it currently sits below the capability floor for
reasons that are tokenizer/context-bound (D3.3), not objective-bound. Do not cite
the causalfix null as support for K1 — it does not bear on it.

## Program L — Agent collaboration
**Status:** `OPEN`  
**Questions:** Multi-proposer protocols; debate; division of labor; consensus under verification; failure isolation.

## Program M — Program synthesis
**Status:** `OPEN`  
**Questions:** Synthesizing extractors, verifiers, and routers from schemas and traces; compiler-assisted pipelines.

## Program N — Learning under scarce supervision
**Status:** `PARTIALLY_TOUCHED` (diversity effects; undertrained 160M/200M cell)  
**Questions:** Data curricula, active learning for hard types, tokens-per-parameter interactions without overclaiming scale laws.


## Program P — Benchmark infrastructure (inside nano-lm)
**Status:** `ACTIVE_DESIGN` (Program 0 only; Program 1 not authorized)  
**Questions:** Can nano-lm run one pinned, reproducible, per-item-logged benchmark path (Gate 0)? Later: where do existing checkpoints lose under fair eval?  
**Does not require:** a separate research institution; NanoScribe; training.  
**Non-claim:** Program 0 smoke ≠ leadership on any public suite.


## Program O — Compiler-assisted reasoning / runtime
**Status:** `OPEN`  
**Questions:** Lowering typed plans to deterministic runtimes; replay; observability; deterministic debugging of cognitive stacks.

---

## Program Q - Domain acquisition
**Status:** `OPEN` (added 2026-08-25)  
**Question:** Can the system learn an unfamiliar domain **without rebuilding the
system**?

Given quantum physics, oncology, or an unknown field, the target behaviour is:
recognise unfamiliar terminology, observe, learn concepts, connect them,
construct a domain model, retrieve, self-test, integrate. Knowledge is acquired;
capability is not modified.

**Why this is the load-bearing program for the Core/DomainPack boundary.** The
boundary claims capability is general and knowledge is concentrated
(`docs/SYSTEM_ARCHITECTURE.md`). O is the experiment that falsifies it: if
acquiring a new domain requires changing Core interfaces, capability was never
general and the split is decorative. Medicine cannot test this - it is the pack
the Core was designed alongside, so a fit proves nothing.

**Falsifier.** A new domain that cannot be acquired without a Core change.

**Distinct from N.** N is learning a task from few labels. O is acquiring a
*conceptual* domain - vocabulary, entities, relations, semantics - where the
capability set is unchanged and only knowledge grows.

**Convergent external evidence.**
`papers/FINDING_BREADTH_BEFORE_SPECIALIZATION.md` - compute-matched studies find
reusing a broadly pretrained model dominates training a domain-native one from
scratch, which supports O's premise (general capability plus added knowledge)
over domain-native construction.

## Cross-cutting domains (also portfolio, not queue)

Learning · Memory · Verification · Retrieval · Planning · Execution · Collaboration · Interfaces · Hardware · Human factors · Evaluation · Benchmarks · Tool ecosystems · Security · Alignment · Compilers · Runtime · Distributed systems · Theory

Each domain may host multiple programs. Domains do **not** inherit evidence from NanoScribe branding.

## Relation to killed product thesis

| Killed (Layer 1) | Still open (Layer 3) |
|------------------|----------------------|
| Generative substrate preferred on old closed task under frozen \(U\) | Whether ∃ any regime where gen+verify wins under matched \(U\) |
| “Scale alone buys copying” as parameter-only law | How data, adaptation, and architecture jointly affect copying |
| LoRA “geometry preservation” as established | Mechanism discrimination (E2) as optional science |
| Fabric as cognitive OS | Verification theory and provenance as research programs |

## Anti-timidity note

An empty **execution queue** is compatible with a full **research portfolio**.  
Do not delete programs A–P because the laboratory is idle on experiments.
