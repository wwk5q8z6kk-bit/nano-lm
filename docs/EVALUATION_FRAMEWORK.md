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
