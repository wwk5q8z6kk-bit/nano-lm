# E1 L/C Reconstruction Protocol (DESIGN ONLY)

**Status:** `DESIGN_DRAFT` — not authorization to run.  
**Purpose:** Close or permanently split `C_E1_MEASUREMENT.reproducibility = PUBLIC_PARTIAL`.  
**Gate:** G2 design now; G3 clean-clone replay only with explicit owner auth.

## First principle

Decision reproducibility (KILL from public utilities) ≠ cost-term reproducibility (\(L\), \(C\)).

## Scope

Reconstruct only:

- \(L\): latency / review-time term as defined in E1 prereg / `trajectory/e1/common.py`
- \(C\): normalized compute term as frozen in E1 utility

**Out of scope:** retraining models; changing \(U\) weights; new generative runs.

## Required artifacts (checklist)

1. Exact formula citations (file + symbol).
2. Device/normalization table (MPS/CUDA/CPU) used historically.
3. Mapping from stored item/method JSONs → component values.
4. Tolerance bands for float recompute.
5. Clean-clone script path (to be added only under auth).
6. Pass/fail: match published utilities within ε **or** document irreducible gap → keep PUBLIC_PARTIAL with new note.

## Exit

- Upgrade ledger reproducibility to `PUBLIC_REPRODUCIBLE`, **or**
- Split claim: `C_E1_DECISION_REPRO` vs `C_E1_COST_REPRO` with honest statuses.

## Non-authorization

```text
AUTHORIZE_E1_LC_REPLAY = NOT_PRESENT
```
