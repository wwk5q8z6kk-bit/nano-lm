# Nano LM — Research-to-architecture package (2026-11-21)

**One native model as the scientific destination; candidate architecture and narrow mathematical results only; research ledger maintained by explicit evidence updates, not optimistic prose.**

## Quick conclusions

| Question | Current decision | Evidence status |
|---|---|---|
| Which architecture wins? | **Not known**; dense is mandatory reference, hybrids and APS are competing hypotheses | Observed M21 code loss, otherwise incomplete |
| Which finding changed the plan? | Original fixed-address acquisition 3.4% late; hashing 99.9% and Adam-state reset 86.1% | Repository, one-seed pilot |
| What has been proved? | Invertible transport preserves decodes; nullspace protects sampled old logits to first order; exact facts require bits; versioned records answer history; compute routing has threshold | Conditional mathematics, with 5/5 numerical checks |
| What was *not* proved? | No superior language-model architecture, general self-improvement, or true continual learning | Unproven |
| Next decisive research | Address×optimizer and correction/replay factorial, then learned state migration and cross-episode procedural transfer | Planned; not executed here |
| Compute policy | No paid experiment launched or authorized by this package | Verified repository rules |

## Contents

1. [Architecture candidate and mathematical propositions](01_ARCHITECTURE_AND_THEOREMS.md) — diagram, governing equations, five scoped proofs, counterexamples, modern baselines.
2. [Decision and research ledger](02_DECISION_LEDGER.md) — 22 decisions, corrections/reversals, evidence inventory, open hypotheses, maintenance rule.
3. [Experimental program and engineering plan](03_EXPERIMENTS_AND_EXECUTION.md) — 18 experiment gates, controls, native training and scale budgets.
4. [Evidence and primary research register](04_EVIDENCE_AND_REFERENCES.md) — repository citations, contemporary literature and audit limitations.
5. [Executable mathematical sanity checks](checks/formal_sanity.py) and [results](results/formal_sanity_20261121.json) — deterministic CPU checks, not neural-model experiments.
6. [Memory-addressing diagnostic chart](figures/memory_acquisition_controls.png) — prior repository data, one seed.

![Prior repository memory-acquisition result, one-seed pilot](figures/memory_acquisition_controls.png)

## Research-architecture lifecycle

```mermaid
flowchart LR
    E[Repository experiments and independent literature] --> H[Competing hypotheses]
    H --> M[Equations, proofs, counterexamples]
    M --> X[Cheap preregistered experiments]
    X --> L[Decision / evidence ledger]
    L --> I{Survives matched controls?}
    I -->|no| H
    I -->|yes| N[Native language and code evaluation]
    N --> U{Quality / cost frontier improves?}
    U -->|yes| A[One integrated Nano LM architecture candidate]
    U -->|no| L
```

### Use and restrictions

This package augments—not overwrites—the existing `research/frontier/ontogenesis/LEDGER.md` on the repository research branch. All past results remain attached to their original commit/protocol. It does not rename Nano LM or assume any one model configuration is chosen. Private owner data and PHI are not authorized. No large training or dataset download is implied.