# Medical Data Registry

## Current program policy (strict)

| Class | Policy |
|-------|--------|
| **Public / openly licensed / appropriately de-identified** | Allowed subject to dataset registry entry |
| **Private owner material** | **NOT AUTHORIZED** for current Nano experiments |
| **PHI** | **NOT AUTHORIZED** |
| **Cloud PHI** | **NOT AUTHORIZED** |

“Not in Git” is **not** sufficient protection. A future sensitive-data program may supersede this explicitly.

## External inference egress

Hosted API and serverless adapters fail closed unless their `ModelInput` carries
an `ExternalEgressAuthorization`: an explicit `NON_PHI_AUTHORIZED`
classification, source and run provenance IDs, a source ID/exact outbound
transcript-digest binding, and a specific authorized cloud target (including
the RunPod endpoint ID). Its claims require an HMAC signature from the
out-of-process approval authority; the verification key is required at runtime
and is never stored in this repository. The gate runs before a client is
created and emits no transcript telemetry. Local-weight and fixture paths do
not egress data; fixture execution must be labeled as such in eval output.

## Dataset registry (in-repo)

| Path | Content | License / notes |
|------|---------|-----------------|
| `scribe/` | Synthetic dialogue / scribe fixtures | In-repo synthetic |
| `data/` | External lexicons (e.g. clinical termsets) | Per-file license |
| `wedge_v1/data/` | Demo/fixture corpora | No PHI |

## Cross-branch data

H6 and span-port datasets live on **not-yet-integrated** branches — not assumed in this tree.

## Evaluation data for P1 exit

External medical dialogue eval sets must be licensed/authorized and referenced by manifest — see [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md).
