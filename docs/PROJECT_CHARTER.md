# Project Charter

## Mission

> **Build the smallest useful, reliable intelligence system that can construct a faithful representation of messy evidence, compress it without corrupting it, maintain that representation across time, and ultimately reason, plan, and act over it while preserving provenance and uncertainty.**

Nano is **not** defined by parameter count, a single benchmark, or one application. It is a research and engineering program for compact, verification-gated intelligence.

## Optimization target

Optimize the Pareto frontier across:

```text
capability
× factual reliability
× evidence grounding
× controllability
× privacy
× memory
× latency
× compute
× energy
× monetary cost
× human review burden
```

## Why medicine first

Medical scribing is the **first proving ground**, not the permanent boundary.

It forces the system to solve, in one consequential domain:

- exact information transport and values
- speaker and experiencer attribution
- negation and uncertainty
- temporality and contradiction
- structured state and evidence binding
- salience, compression, and coherent generation
- verification, abstention, and review under error cost

Medicine is **DomainPack #1**. Nano Core must remain domain-general.

## Architectural shape

```text
Nano Core  +  DomainPack  =  domain system

Nano Core  +  Medical DomainPack  =  NanoScribe / Medical Intelligence
Nano Core  +  future Legal/Scientific/Technical pack  =  future system
```

## Model / software co-design (governing rule)

> **Do not ask the model to learn what software can solve more reliably, and do not maintain brittle software for what learned representations can solve more generally.**

Use the **smallest sufficient solver**: deterministic software, retrieval, schemas, constrained decoding, compact models, teachers, or hybrids.

## Three research layers (all programs)

| Layer | Role |
|-------|------|
| **Mechanism models** | 3M–100M from-scratch experiments — cheap microscopes |
| **Compact production models** | Pretrained/adapted/distilled — capability per byte/watt/dollar |
| **Teacher/reference models** | Ceilings, synthetic data, judging, distillation, adversarial eval |

Teachers are **not** the deployed system.

## Historical foundation (not the definition)

The original **3.15M from-scratch LM** on Apple Silicon remains the project's experimental foundation and evaluation testbed. It is not the final architecture or size target.

Measured kills (E1, E4 on tested regimes) apply to **those tested substrates and utilities** — not to the full capability-ladder program above.

## Evidence discipline

- **Science / evidence:** `papers/`, `trajectory/`, freeze tags — conservative, immutable where tagged
- **Current program truth:** `docs/` (this tree)
- **Branch notes:** `frontier/` when present — never canonical authority

See [PROJECT_AUTHORITY.md](PROJECT_AUTHORITY.md).
