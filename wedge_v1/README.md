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
python3 -m wedge_v1 dogfood                      # 8 tasks on papers/ (classical)
```

**Latest:** dogfood **8/8** on `papers/` with fail-closed abstention. Classical-first; no LM.

Optional: `pip install pypdf` for PDF text-layer ingest (`wedge_v1/requirements-optional.txt`).

Not a Layer-1 Evidence Ledger claim until owner promotion.
