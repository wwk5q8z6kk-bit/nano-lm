# Active Frontier

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
