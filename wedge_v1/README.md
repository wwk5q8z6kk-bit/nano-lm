# wedge_v1 — Local research document intelligence

```bash
python3 -m wedge_v1 smoke
python3 -m wedge_v1 ingest --corpus DIR          # md/txt (+ PDF if pypdf installed)
python3 -m wedge_v1 ask --corpus DIR "…"
python3 -m wedge_v1 find --corpus DIR "exact phrase"
python3 -m wedge_v1 scan --corpus DIR
python3 -m wedge_v1 compare --corpus DIR metformin
python3 -m wedge_v1 report ask "How long before cached entries expire?"
python3 -m wedge_v1 report compare metformin
python3 -m wedge_v1 dogfood                      # papers/ classical pack
python3 -m wedge_v1 owner-dogfood --demo         # synthetic owner-path rehearsal
# export OWNER_CORPUS=~/notes && python3 -m wedge_v1 owner-dogfood --smoke
```

**Latest:** papers dogfood + owner-corpus harness (`OWNER_CORPUS.md`). Classical-first; no LM.

Optional: `pip install pypdf` for PDF text-layer ingest (`wedge_v1/requirements-optional.txt`).

Not a Layer-1 Evidence Ledger claim until owner promotion.

## Owner-corpus dogfood (Active Frontier)

```bash
# Smoke on synthetic (CI-safe):
PYTHONPATH=. python -m wedge_v1 owner-dogfood --corpus wedge_v1/data/corpus

# Real private folder (gitignored results; no PHI in git):
export OWNER_CORPUS=/path/to/private/docs
PYTHONPATH=. python -m wedge_v1 owner-dogfood --corpus "$OWNER_CORPUS"
```

Results land in `wedge_v1/results_owner_dogfood.json` (gitignored). Not Layer-1 evidence.

```bash
python -m wedge_v1 owner-smoke              # example or gitignored owner_corpus/
python -m wedge_v1 owner-smoke --corpus "$WEDGE_OWNER_CORPUS"
```
