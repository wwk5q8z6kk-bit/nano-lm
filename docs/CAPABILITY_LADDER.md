# Capability Ladder

## Macro-phases

```text
FOUNDATION I
P1 — Master Scribing
faithful capture

        ↓

FOUNDATION II
P2 — Summarization
P3 — Charting
faithful compression + longitudinal state

        ↓

INTELLIGENCE EXPANSION
P4–P9
synthesis → questioning → reasoning → planning → action → adaptation
```

## Programs P1–P9

| Program | Question | Near-term? |
|---------|----------|------------|
| **P1 Scribing** | What happened? | **Yes — current frontier** |
| **P2 Summarization** | What matters? | Spec / interface only |
| **P3 Charting** | What is the state? | Spec / interface only |
| **P4 Synthesis** | What does combined evidence show? | Architectural requirement |
| **P5 Questioning** | What is unknown? | Architectural requirement |
| **P6 Reasoning** | What follows? | Architectural requirement |
| **P7 Planning** | What should happen next? | Architectural requirement |
| **P8 Tools/Action** | What should we do? | Architectural requirement |
| **P9 Adaptation** | What did we learn? | Architectural requirement |

Do not skip ordering because a frontier model can superficially perform a later stage.

## Capability levels — a decomposition *within* P1–P9, not a parallel scheme

The clinical-intelligence level stack (L0–L12) refines the existing programs; it
does **not** replace them. Each level is a gate inside its program, so the
ordering discipline above still governs. Status is per level, against the
ledger — `BUILT` means a supporting claim exists, not that the level is finished.

| Level | Capability | Program | Status |
|-------|-----------|---------|--------|
| L0 | Perceive — documents, speech, images, labs, signals | P1 | **PARTIAL** — text + structured only; image/signal/audio absent |
| L1 | Understand — entities, events, diagnoses, medications | P1 | **BUILT** — encounter v0, atom specs |
| L2 | Temporalize — when, duration, sequence, recurrence | P1 | **PARTIAL** — `temporality` is a Core primitive; event/doc/onset/discovery time not separated |
| L3 | Structure — patient state, evidence graph, timeline | P1→P3 | **PARTIAL** — evidence *spans* built (`fabric/schemas.py`); evidence *graph* absent |
| L4 | Remember — longitudinal patient memory | P3 | **ABSENT** — state is per-encounter |
| L5 | Retrieve — find exactly what matters | P2 | **DESIGNED** — `retrieval`/`salience` are Core primitives |
| L6 | Synthesize — presentations, summaries, histories | P2 | **DESIGNED** |
| L7 | Visualize — timelines, trajectories, diagrams | P3 | **ABSENT** |
| L8 | Reason — relationships, differentials, inconsistencies | P4–P6 | **PARTIAL** — `contradiction` is a Core primitive, unbuilt |
| L9 | Monitor — change, risk, missing follow-up | P5 | **ABSENT** |
| L10 | Assist — documentation, planning, communication | P7 | **ABSENT** |
| L11 | Decision support — evidence + guidelines + state | P8 | **ABSENT — regulated**, see below |
| L12 | Learn — outcomes, longitudinal feedback | P9 | **ABSENT** |

**What is already built is the part usually retrofitted.** Provenance is not a
later feature here: `fabric/schemas.py` carries typed `EvidenceSpan` / `Claim` /
`VerificationResult` with content-addressed ids and an
absence-never-from-silence gate, and `C_FABRIC_SLICE` measures propose→verify→
abstain driving presented error to 0 on closed synthetic inst0. The three-way
terminal decision (present / abstain / review) is already the Core's exit.

**The genuine architectural gaps** are therefore narrower than the level list
suggests: multimodal perception (L0), the evidence **graph** as a first-class
structure distinct from spans (L3), longitudinal cross-encounter state (L4), and
visualization (L7). Everything else is either built, or a named Core primitive
awaiting a program.

**L11 changes the regulatory posture.** Documentation and summarization sit
outside device classification; diagnosis and treatment recommendation do not.
`docs/ACTIVE_NOW.md` already forbids clinical claims without external human
validation. Crossing L10→L11 is an owner decision with regulatory consequences,
not a capability increment.

## Product naming (medical DomainPack)

| Stage earned | Name |
|--------------|------|
| P1 | **NanoScribe** — verified encounter record + note |
| P1–P3 | Longitudinal medical documentation intelligence |
| P4–P9 | Broader medical intelligence capabilities (sequential) |

Details: [domains/medical/SCRIBING.md](domains/medical/SCRIBING.md)
