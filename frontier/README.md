# Active Frontier

## Current development state

Active development is underway on `frontier/active-v1` under
`DEVELOPMENT_PLAN.md`.

The current product frontier is Nano Runtime’s local research-document
intelligence wedge: a local-first, verification-gated system that uses the
smallest sufficient solver for each task.

P0 Discovery and P1 Verified Ask CLI are complete. P2 is green on the
fixture corpus but still requires owner-corpus validation. P3 has started
with contact evaluation and the initial habit surface.

Small-language-model work is not the current default. It may be considered
only after real P2–P3 use identifies repeated, evidence-backed
over-abstention cases that cheaper solvers cannot resolve.

**Authority:** [`../papers/PROGRAM_AUTHORITY.md`](../papers/PROGRAM_AUTHORITY.md) · Paper α landing: [`../papers/README.md`](../papers/README.md) (publication only)

**Mandate:** `BUILD_SMALL_POWERFUL_USEFUL_SYSTEM_V1` (see `ACTIVE_MANDATE.md`)

| Path | Role |
|------|------|
| `PRODUCT_DISCOVERY_SPRINT.md` | **Canonical** Phase-1 discovery deliverable |
| `PRODUCT_DISCOVERY_SPRINT_V1.md` | Earlier draft (kept for lineage) |
| `verified_ask_report.py` | Frontier claim-report helper used by `wedge_v1 report verified` |
| Product runtime | `../wedge_v1/` |

Evidence Core (Paper α, ledger, protected tags) is **out of scope** for edits here.

## Owner-corpus dogfood

Prove the harness on the public fixture (no private docs needed):

```bash
PYTHONPATH=. python3 -m wedge_v1 owner-dogfood --demo
```

Run against a private folder **outside git** (never commit PHI):

```bash
PYTHONPATH=. python3 -m wedge_v1 owner-dogfood --corpus ~/path/to/private/docs
# or: export OWNER_CORPUS=~/path/to/private/docs && python3 -m wedge_v1 owner-dogfood
```

Writes gitignored `wedge_v1/results_owner_dogfood.json` + failure gallery. Classical-only; not Layer-1 evidence.

## Fixture ΔU arms (classical vs hybrid stub)

```bash
PYTHONPATH=. python3 -m wedge_v1 eval-arms --demo
```

Scores draft U on both arms and emits `KEEP_CLASSICAL` or `ADMIT_HYBRID_STUB` when ΔU > δ (default 0.05). No training; stub escalates only on classical ABSTAIN with verified spans.

