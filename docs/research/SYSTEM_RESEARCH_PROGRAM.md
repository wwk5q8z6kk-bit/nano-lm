# System Research Program

Software-system track alongside model layers.

## Components

```text
retrieval · memory · schemas · constrained decoding
structured representations · verifiers · deterministic transforms
tools · routing · human review
```

## In-repo implementations

| Component | Path | Status |
|-----------|------|--------|
| Verification harness | `fabric/` | Regression slice — typed claims, literal/rules verifiers |
| Local document intelligence | `wedge_v1/` | Classical-first Q&A with span evidence ([subsystems/WEDGE.md](../subsystems/WEDGE.md)) |
| Benchmark infra | `benchmarks/` | Program 0 sentinel |
| Auth / speech-act lint | `scripts/lint_claim_auth.py` | Governance automation |

## Co-design rule

For each subsystem:

1. Can deterministic software + retrieval + schema solve it more reliably?
2. Is brittle rule logic hiding a learnable representation?
3. What is the smallest sufficient solver **today**?

## Memory and state (forward)

P3 charting requires persistent identity. Today's work must expose:

```text
entity_id · event_id · encounter_id · provenance links
timestamps · supersession · contradiction edges
```

without building the full longitudinal engine prematurely.

## Failure loop

See [FAILURE_TO_ARCHITECTURE.md](../FAILURE_TO_ARCHITECTURE.md) and [EXPERIMENT_STRATEGY.md](EXPERIMENT_STRATEGY.md).
