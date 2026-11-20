# Evaluation Framework

Measure the general language model on held-out text, reasoning and instruction following with explicit model identity and a matched baseline. Report any additional retrieval, tools, graders or external model assistance. The historical P1 exit criteria below applied to the former medical product.

## Principles

1. **Pre-register** what would change the roadmap before running.
2. **Same instrument, same rules** when comparing arms.
3. **Honest failure** — a FAIL that locates the next lever beats an unregistered PASS.
4. **Scope claims** — verifier relation, distribution, and utility function must accompany every metric.
5. **Human review** for consequential domains — automatic metrics necessary, not sufficient.

## Three evaluation layers (frozen)

Frozen by owner directive, 2026-08-27, quoted verbatim:

> Public benchmarks measure capabilities. Native evaluations measure the engineering process.
> System evaluations measure whether those capabilities survive contact with tools, environments,
> budgets, uncertainty, and verification.

The research question this frames:

> Can the system reduce uncertainty, take evidence-justified actions, revise its beliefs when
> evidence changes, produce correct engineering outcomes, and establish that those outcomes are
> sufficiently verified?

In the agent-system study, native process evaluations were the research contribution. For the current base model, use public held-out language and task evaluations directly. When evaluating an agentic system, record the additional components and execution conditions using [`../benchmarks/AGENT_RUN_IDENTITY.schema.json`](../benchmarks/AGENT_RUN_IDENTITY.schema.json)
and, for the program design, [`../papers/SWE_CAPABILITY_EVALUATION_PROPOSAL.md`](../papers/SWE_CAPABILITY_EVALUATION_PROPOSAL.md)
(design proposal — not authorization).

## Utility pattern (when comparing solvers)

\[
U = Q - w_E E - w_R R - w_L L - w_C C
\]

Symbols vary by program; always freeze weights before scoring. Kill/keep: arm must beat best cheaper solver by \(\delta\) (default 0.05 unless preregistered).

## Historical application measures

| Program | Primary measures |
|---------|-------------------|
| P1 Scribing | factual precision, critical omission, exact-value transport, span correctness, negation/uncertainty/temporality, attribution, section completeness, coherence, unsupported claims, review burden, clinician edit effort |
| P2 Summarization | precision, critical omission, salience, compression, provenance retention, contradiction/uncertainty retention |
| P3 Charting | entity resolution, timeline correctness, supersession, contradiction handling, traceability to source events |
| P4+ | program-specific — see [CAPABILITY_LADDER.md](CAPABILITY_LADDER.md) |

## Historical P1 exit gate

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
