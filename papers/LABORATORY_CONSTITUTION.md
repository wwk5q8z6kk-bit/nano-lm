# Laboratory Constitution

**Adopted:** 2026-07-31  
**Purpose:** Separate *what is true*, *what we may dream*, *what we might build someday*, and *what we do next* so ambition and evidence never compete.

```text
PROGRAM_EXECUTION_STATUS: IDLE_AFTER_FREEZE
AUTHORIZED_NONEXECUTION_WORK: E4_DESIGN_ONLY  # protocol docs only; not experiments
EVIDENCE_STANDARDS: CONSERVATIVE  # unchanged by this constitution
VISION_STANDARDS: EXPANSIVE       # allowed here without becoming a claim
```

## The three layers (plus two operating docs)

| Layer | Question | Document | Standard |
|-------|----------|----------|----------|
| **1. Scientific evidence** | What have we demonstrated? | `EVIDENCE_LEDGER.md` (+ Paper α, preregs, artifacts) | **Conservative.** No ambition. |
| **2. Engineering / technology roadmap** | If unlimited evidence eventually supported it, what *could* we build? | `TECHNOLOGY_ROADMAP.md` | **Ambitious.** Explicitly non-evidential. |
| **3. Research agenda / portfolio** | What is actually true? What questions are worth 5–10 years? | `RESEARCH_PORTFOLIO.md` | **Expansive.** Hypotheses to destroy one by one. |
| Operating: **Execution queue** | What are we authorized to build/run *now*? | `EXECUTION_QUEUE.md` | Tiny. Gate-bound. |
| Operating: **Decision gates** | What evidence promotes an idea into implementation? | `DECISION_GATES.md` | Strict promotion rules. |

**Constitutional rule:** Negative evidence kills **hypotheses**, not **curiosity**.  
E1 KILL falsifies “generative extraction is preferred for *this* closed task under *this* \(U\)”.  
It does **not** falsify memory, verification theory, planning, collaboration, compilers, or world models.

## Long-term mission (aspirational — not a claim)

Build a factorized, evidence-first, verification-gated cognitive laboratory and (eventually, if evidence warrants) systems stack in which:

- learning, memory, retrieval, planning, execution, and collaboration are first-class;
- every claim carries provenance and uncertainty;
- verification and abstention are engineered, not hoped;
- classical, generative, hybrid, and tool-using methods compete under explicit utilities;
- failure modes remain permanent regression instruments.

This mission is a **north star**. It is not licensed by current results.

## Research map (expansive)

```
Long-term Mission
        │
        ▼
Research Programs  (see RESEARCH_PORTFOLIO.md)
        │
        ├── Learning
        ├── Memory
        ├── Verification
        ├── Retrieval
        ├── Planning
        ├── Execution
        ├── Collaboration
        ├── Interfaces
        ├── Hardware
        ├── Human factors
        ├── Evaluation / Benchmarks
        ├── Tool ecosystems
        ├── Security / Alignment
        ├── Compilers / Runtime
        ├── Distributed systems
        └── Theory
```

Experiments only **move evidence** within this map. They do not redefine the map by shrinking it after a kill gate.

## Document authority

| Doc | May claim “is true / measured”? | May describe future systems? | Authorizes build/run? |
|-----|----------------------------------|------------------------------|------------------------|
| `EVIDENCE_LEDGER.md` | **Yes** (scoped) | No | No |
| `RESEARCH_PORTFOLIO.md` | No | Questions only | No |
| `TECHNOLOGY_ROADMAP.md` | No | **Yes** (conditional) | No |
| `EXECUTION_QUEUE.md` | Only by citing ledger | No | **Yes** (current auth only) |
| `DECISION_GATES.md` | Process | No | Defines promotion |
| `MASTER_PLAN.md` / `NANOSCRIBE_VNEXT.md` | Historical + pointers | Legacy architecture prose | **No** — not the queue |
| `AMBITION.md` | Bridge note | Narrow R★ framing | Design-only carve-out |

## Anti-contamination rules

1. Do not put roadmap modules into the Evidence Ledger until measured.  
2. Do not delete research programs because a product thesis died.  
3. Do not treat `EXECUTION_QUEUE` emptiness as “stop dreaming.”  
4. Do not treat Technology Roadmap items as “next sprint.”  
5. Promote Layer 2→queue only through `DECISION_GATES.md`.

## Companions

- Evidence: `EVIDENCE_LEDGER.md`, `EMPIRICAL_FOUNDATION.md`, `CLAIM_GLOSSARY.md`  
- Freeze: `post-alpha-evidence-freeze-2026-07-31` (immutable)  
- Current R★ design: `trajectory/REGIME_P1_…`, `trajectory/PREREG_E4_…`  
- Status: `audit/discussion-to-implementation/CANONICAL_STATUS_TABLE.md`
