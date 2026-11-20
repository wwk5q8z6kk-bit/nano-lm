# Verification Subsystem

## Fabric (`fabric/`)

**Role:** Verification vertical slice / regression harness — **not** NanoScribe architecture.

```text
propose → verify → present | abstain | review
```

- Typed claims with literal/rules verifiers
- Scoped to defined verifier relations and distributions
- Stage G/A historical results: presented precision gains on synthetic scribe distribution — **not** open-world hallucination elimination

**CI:** `pytest fabric/test_fabric.py`

## Constructive faithfulness (Wedge / LM probe)

When generation is used:

1. Retrieve candidate passages
2. Model quotes **verbatim** substring only
3. Relocate uniquely in corpus
4. `verify_claim` + ablation must pass — else ABSTAIN

See `wedge_v1/lm/mlx_backend.py` (local MLX path, gated).

## E1/E4 as routing evidence

| Gate | Scoped lesson |
|------|----------------|
| E1 KILL | Classical beats generative on **old closed task** under frozen U |
| E4 KILL | Classical beats generative+verify on **tested R★** |

Use for **solver routing**, not global "no AI."

## Human review

Final validation for P1 scribing requires clinician review — agent rubrics (E3) are not human IAA.

## Related docs

- [FAILURE_TO_ARCHITECTURE.md](../FAILURE_TO_ARCHITECTURE.md)
- [subsystems/WEDGE.md](WEDGE.md)
- `papers/FIRST_PRINCIPLES_RISK_MITIGATION.md` (B-atom register)
