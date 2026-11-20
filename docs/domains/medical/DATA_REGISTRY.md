# Medical Data Registry

## Current program policy (strict)

| Class | Policy |
|-------|--------|
| **Public / openly licensed / appropriately de-identified** | Allowed subject to dataset registry entry |
| **Private owner material** | **NOT AUTHORIZED** for current Nano experiments |
| **PHI** | **NOT AUTHORIZED** |
| **Cloud PHI** | **NOT AUTHORIZED** |

“Not in Git” is **not** sufficient protection. A future sensitive-data program may supersede this explicitly.

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
