# Medical DomainPack

This directory records NanoScribe, the former medical application. Its protocol and naming are preserved for reproduction. General-model research follows the [project charter](../../PROJECT_CHARTER.md).

## Why the earlier program used medicine

High-consequence, structured, measurable errors force mastery of:

- exact values and medications
- negation, uncertainty, temporality
- speaker/experiencer attribution
- evidence binding and verification
- coherent clinical documentation

## Documents

| File | Scope |
|------|-------|
| [SCRIBING.md](SCRIBING.md) | P1 — encounter → verified record → note |
| [SUMMARIZATION_AND_CHARTING.md](SUMMARIZATION_AND_CHARTING.md) | P2/P3 — future contract |
| [DATA_REGISTRY.md](DATA_REGISTRY.md) | Data boundaries |
| [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) | External + human validation |

## Deployment name

```text
P1 (NanoScribe) = Nano Core + Medical DomainPack + Master Scribing
```

Not the full medical-intelligence system — see [../../CAPABILITY_LADDER.md](../../CAPABILITY_LADDER.md).

## Non-goals (current)

- Clinical deployment product claims
- PHI in public git
- Skipping P1 exit gate for P2/P3 buildout
