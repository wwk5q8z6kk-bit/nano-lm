# Corpus contact note (ACTIVE FRONTIER)

**Not Layer-1 evidence.** Updated under `continue autonomous` (2026-07-31T18:47:17Z).

| Class | Corpus | n_docs | supported/contradicted | abstain | Artifact |
|-------|--------|--------|------------------------|---------|----------|
| PAPERS_DOGFOOD | `papers/` | 35 | 3/5 | 2/5 | `wedge_v1/results_corpus_contact_papers.json` |
| OWNER_PRIVATE | — | — | — | — | **blocked: set OWNER_CORPUS / pass `--corpus`** |

## Honest sentences (papers dogfood)

- **Useful:** Exact find and span-backed answers on governance docs beat scrolling when looking up known numbers like 0.925.
- **Not useful:** Natural-language research questions often abstain or contradict without a clear primary claim, so it does not yet replace reading.

## Autonomous continue outcome

- Restored missing `frontier/verified_ask_report.py` (smoke was red).
- `python -m wedge_v1 smoke` → `WEDGE_V1_SMOKE_OK`.
- Re-ran papers contact: open clinical probe abstains; TTL/find/compare support.
- `evolve` still recommends **OWNER_CORPUS_CONTACT** — cannot fake private docs.
- LM probe still **NOT_INDICATED**.

## Exact next owner action

```bash
export OWNER_CORPUS=/path/to/private/docs   # N>=10, no PHI in git
python -m wedge_v1 owner-ready --corpus "$OWNER_CORPUS"
python -m wedge_v1 contact --corpus "$OWNER_CORPUS" --class OWNER_PRIVATE \
  --useful "..." --not-useful "..." -o wedge_v1/.private/results_corpus_contact_owner.json
```
