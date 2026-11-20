# Decision record — Wedge v1 Phase 2 classical baseline

**Date:** 2026-07-31
**Auth:** `AUTHORIZE_WEDGE_V1_CLASSICAL_BASELINE`
**Trigger:** Owner "continue"

## Result

| Metric | Value |
|--------|-------|
| U | ≈ 0.926 |
| Q | 1.0 |
| E | 0.0 |
| R | ≈ 0.079 |
| L | ≪ 1s |
| C | 1.0 (reference) |
| Task checks | 40/40 pass |

Artifact: `wedge_v1/results_wedge_v1_classical.json` (+ `trajectory/` copy).

## Interpretation (scoped)

On the **frozen clean synthetic** mini-corpus, classical+verify already delivers high U.
E-class items correctly **abstain** (T35 paraphrastic, T36 implicit, T39 coref) — those are
the only honest Phase 3 candidates, not a license for LM-first productization.

This does **not** claim open-world or clinical readiness.
