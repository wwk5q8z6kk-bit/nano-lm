# Project Charter

## Mission

> **Build the smallest useful, reliable intelligence system that can construct a faithful representation of messy evidence, compress it without corrupting it, maintain that representation across time, and ultimately reason, plan, and act over it while preserving provenance and uncertainty.**

Nano is **not** defined by parameter count, a single benchmark, or one application.

## Macro-phases (strategic intent)

```text
FOUNDATION I — P1 Master Scribing
faithful capture: "What happened?"

        ↓

FOUNDATION II — P2 Summarization + P3 Charting
faithful compression + longitudinal state:
"What matters?" · "What is the state, and how has it changed?"

        ↓

INTELLIGENCE EXPANSION — P4–P9
synthesis → questioning → reasoning → planning → action → adaptation
```

The nine-program ladder ([CAPABILITY_LADDER.md](CAPABILITY_LADDER.md)) details P1–P9. Only **Foundation I** is the near-term product frontier.

## Capability-aware product naming

```text
Nano Core + Medical DomainPack + P1  =  NanoScribe (faithful encounter record)

Nano Core + Medical DomainPack + P1–P3
  =  longitudinal medical documentation intelligence (not yet earned)

P4–P9  =  progressively broader synthesis, reasoning, planning, action, adaptation
```

A scribe is **not** the full medical-intelligence system.

## Optimization target

Optimize the Pareto frontier across:

```text
capability × factual reliability × evidence grounding × controllability
× privacy × memory × latency × compute × energy × monetary cost × human review burden
```

## Why medicine first

Medical scribing is **DomainPack #1** — high-consequence, structured, measurable. Nano Core stays domain-general.

## Model / software co-design

> **Do not ask the model to learn what software can solve more reliably, and do not maintain brittle software for what learned representations can solve more generally.**

## Compute

**RunPod is Nano’s primary GPU training and experimental-compute backend** (`training_status=ACTIVE`). Local Apple Silicon/CPU is for development, smoke tests, analysis, preprocessing, evaluation, and small/cheap experiments. Expensive or confirmatory runs may still be experiment-scoped; PHI/private clinical data must not go to RunPod. Training venue ≠ deployment venue — compact/local/private deployment remains a long-term axis. Canonical: [ACTIVE_NOW.md](ACTIVE_NOW.md), [infrastructure/RUNPOD.md](infrastructure/RUNPOD.md).

## Three research layers

| Layer | Role |
|-------|------|
| Mechanism models | Cheap controlled experiments (3M–100M from-scratch) |
| Compact production models | Pretrained/adapted/distilled deployable models |
| Teacher/reference models | Ceilings, synth data, judging — not default deploy |

## Historical foundation

The **3.15M from-scratch LM** remains the experimental foundation. Measured kills (E1, E4) apply to **tested regimes** — see typed authority in [PROJECT_AUTHORITY.md](PROJECT_AUTHORITY.md).
