# Reproducibility

Canonical companion to `trajectory/REPRODUCIBILITY.md` (detailed env + tag instructions).

## Layers

| Layer | Location |
|-------|----------|
| Tagged evidence freezes | git tags e.g. `paper-alpha-v1`, `post-alpha-evidence-freeze-2026-07-31` |
| Result JSONs | `trajectory/results_*.json` |
| Evidence manifest | `papers/EVIDENCE_MANIFEST.json` |
| Packaging map | `audit/discussion-to-implementation/CANONICAL_STATUS_TABLE.md` |

## Immutable means

Tagged archival state at a commit — not "every working-tree edit is frozen."

## Verify

```bash
pytest fabric/test_fabric.py trajectory/test_recompute_c3.py -q
```

Full ML stack: `requirements-ml.txt` — see `trajectory/REPRODUCIBILITY.md`.

## Protected

Do not move or rewrite freeze manifests, SHA pins, or primary result JSONs as part of doc migration.
