# Decision record — Wedge v1 noisy-track diagnostic

**Auth:** `AUTHORIZE_WEDGE_V1_NOISY_TRACK`
**Trigger:** Owner "continue" after Phase 3 idle
**RESULT:** `wedge_v1/results_wedge_v1_noisy_diagnostic.json`
**Verdict:** `NOISY_INGEST_NORMALIZE_SUFFICIENT`

| Arm | U | checks | E-class |
|-----|---|--------|---------|
| Clean primary | ≈0.891 | — | — |
| Noisy raw | ≈0.458 | 36/50 | stressed |
| Noisy + OCR normalize | ≈0.859 | 49/50 | 3/3 ok |

Recover gap vs clean ≈0.032 ≤ δ=0.05.

**Interpretation:** Ingestion/normalize is the correct first-principles fix for OCR noise.
Do not credit LM for fixing glyph corruption. Primary U remains clean-track.
LM still not indicated.
