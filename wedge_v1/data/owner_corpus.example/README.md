# Owner corpus layout (example)

Copy this folder to `wedge_v1/data/owner_corpus/` (gitignored) or set:

```bash
export WEDGE_OWNER_CORPUS=/path/to/your/private/notes
python -m wedge_v1 owner-smoke
```

## Layout

```text
owner_corpus/
  *.md | *.txt | *.pdf   # recursive OK
  .wedge_manifest.json   # written by `wedge_v1 ingest` (gitignored)
```

## Rules

- Do **not** commit private notes, PHI, or secrets.
- Results land in `results_owner_smoke.json` (gitignored).
- Classical-first only — no LM probe in this slice.
- Not Layer-1 Evidence Ledger material until owner promotes.

## Seed

The `*.md` files here are **synthetic stand-ins** for CI. Replace with real docs in the gitignored `owner_corpus/` path.
