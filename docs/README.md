# Nano documentation index

**Current project truth lives here.** Scientific history remains under `papers/` and `trajectory/`.

## Authority hierarchy

```text
PROJECT_CHARTER.md      What and why
        ↓
CAPABILITY_LADDER.md    Capability ordering (P1–P9)
        ↓
SYSTEM_ARCHITECTURE.md  Nano Core + DomainPacks
        ↓
ROADMAP.md              Major phases and evolution
        ↓
ACTIVE_NOW.md           Current gate and work (human)
        ↓
EXECUTION_PLAN.md       Executable tasks
```

Machine-readable mirror: [`ACTIVE_NOW.json`](ACTIVE_NOW.json)

## Core

| Document | Purpose |
|----------|---------|
| [PROJECT_CHARTER.md](PROJECT_CHARTER.md) | Mission, optimization target, medical-first rationale |
| [PROJECT_AUTHORITY.md](PROJECT_AUTHORITY.md) | What document wins when sources conflict |
| [CAPABILITY_LADDER.md](CAPABILITY_LADDER.md) | P1 Scribing → P9 Adaptation |
| [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) | Nano Core, DomainPacks, model/software co-design |
| [ROADMAP.md](ROADMAP.md) | Historical arc → current program |
| [ACTIVE_NOW.md](ACTIVE_NOW.md) | Current state and gate |
| [EXECUTION_PLAN.md](EXECUTION_PLAN.md) | Bounded executable work |
| [EVALUATION_FRAMEWORK.md](EVALUATION_FRAMEWORK.md) | How capabilities are measured |
| [FAILURE_TO_ARCHITECTURE.md](FAILURE_TO_ARCHITECTURE.md) | Major failures → architectural lessons |

## Research

| Document | Purpose |
|----------|---------|
| [research/MODEL_RESEARCH_PROGRAM.md](research/MODEL_RESEARCH_PROGRAM.md) | Mechanism / compact / teacher model layers |
| [research/SYSTEM_RESEARCH_PROGRAM.md](research/SYSTEM_RESEARCH_PROGRAM.md) | Retrieval, memory, verifiers, routing |
| [research/EXPERIMENT_STRATEGY.md](research/EXPERIMENT_STRATEGY.md) | Prereg, gates, failure-to-architecture loop |

## Domains

| Document | Purpose |
|----------|---------|
| [domains/medical/README.md](domains/medical/README.md) | Medical DomainPack overview |
| [domains/medical/SCRIBING.md](domains/medical/SCRIBING.md) | P1 scribing target and exit gate |
| [domains/medical/SUMMARIZATION_AND_CHARTING.md](domains/medical/SUMMARIZATION_AND_CHARTING.md) | P2/P3 future contract |
| [domains/medical/DATA_REGISTRY.md](domains/medical/DATA_REGISTRY.md) | Data boundaries (no PHI in git) |
| [domains/medical/EVALUATION_PROTOCOL.md](domains/medical/EVALUATION_PROTOCOL.md) | External + human evaluation requirements |

## Subsystems

| Document | Purpose |
|----------|---------|
| [subsystems/WEDGE.md](subsystems/WEDGE.md) | Verified local document intelligence (`wedge_v1/`) |
| [subsystems/VERIFICATION.md](subsystems/VERIFICATION.md) | Fabric + constructive faithfulness |

## Infrastructure

| Document | Purpose |
|----------|---------|
| [infrastructure/RUNPOD.md](infrastructure/RUNPOD.md) | Paid compute backend (not scientific authority) |
| [infrastructure/REPRODUCIBILITY.md](infrastructure/REPRODUCIBILITY.md) | Tags, manifests, recompute |

## Archive

| Document | Purpose |
|----------|---------|
| [archive/README.md](archive/README.md) | Historical strategy index |
| [archive/LEGACY_STRATEGY_INDEX.md](archive/LEGACY_STRATEGY_INDEX.md) | Old planning doc map |

## Not moved (evidence-protected)

- `papers/EMPIRICAL_FOUNDATION.md`, `papers/EVIDENCE_LEDGER.md`
- `papers/PREREG_*`, `papers/RESULT_*`, manuscripts
- `trajectory/` primary result JSONs, freeze tags, SHA manifests
