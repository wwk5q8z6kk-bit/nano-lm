# wedge_v1 — Local research document intelligence

```bash
python3 -m wedge_v1 smoke
python3 -m wedge_v1 ingest --corpus DIR          # md/txt (+ PDF if pypdf installed)
python3 -m wedge_v1 ask --corpus DIR "…"
python3 -m wedge_v1 find --corpus DIR "exact phrase"
python3 -m wedge_v1 scan --corpus DIR
python3 -m wedge_v1 compare --corpus DIR metformin
python3 -m wedge_v1 report ask "How long before cached entries expire?"
python3 -m wedge_v1 dogfood                      # papers/ classical pack
python3 -m wedge_v1 owner-smoke                  # example corpus contact (5 tasks)
python3 -m wedge_v1 owner-dogfood --corpus wedge_v1/data/corpus
# export WEDGE_OWNER_CORPUS=~/notes && python3 -m wedge_v1 owner-smoke
```

**Owner corpus:** see `data/OWNER_CORPUS.md` / `OWNER_CORPUS.md`. Results are gitignored.

Classical-first; no LM. Not a Layer-1 Evidence Ledger claim until owner promotion.

Optional: `pip install pypdf` for PDF text-layer ingest (`requirements-optional.txt`).

## Owner-corpus dogfood (Active Frontier)

```bash
# Synthetic fixture (CI / no private docs):
PYTHONPATH=. python3 -m wedge_v1 owner-dogfood --demo

# Private folder outside git (never commit PHI):
PYTHONPATH=. python3 -m wedge_v1 owner-dogfood --corpus ~/Documents/my-research-notes
# or: export OWNER_CORPUS=~/Documents/my-research-notes
```

Results: `results_owner_dogfood.json` + `results_owner_failure_gallery.md` (gitignored). Not Layer-1 evidence.
See `OWNER_CORPUS.md`.
