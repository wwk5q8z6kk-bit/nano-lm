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

### DomainPack boundary

**Nano Core** holds domain-general interfaces. **DomainPack** holds:

```text
domain schema · ontology mappings · normalization rules
domain negation/temporality semantics · evaluation sets · transformations
```

Do not hardwire medicine into every Core interface. Do not generalize away medically necessary semantics.

## Deployment shape

```text
Nano Core + Medical DomainPack = NanoScribe / Medical Intelligence
Nano Core + future pack         = future domain system
```

## Subsystems in this repository

| Component | Path | Role |
|-----------|------|------|
| Mechanism / compact models | `pretrain/`, `sft/`, `scribe/`, `nano_ai/` | Training and model experiments |
| Verification harness | `fabric/` | Typed claims, verifiers, abstention regression |
| Verified document intelligence | `wedge_v1/` | Local corpus Q&A with span evidence ([subsystems/WEDGE.md](subsystems/WEDGE.md)) |
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
