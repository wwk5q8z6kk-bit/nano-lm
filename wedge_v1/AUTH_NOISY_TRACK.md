# Wedge v1 — Noisy-track diagnostic auth

**Date:** 2026-07-31  
**Auth string:** `AUTHORIZE_WEDGE_V1_NOISY_TRACK`  
**Role:** Diagnostic only (WEDGE_V1 B5). Does **not** replace clean-track primary U.

```yaml
doc_type: auth_record
valid_only_if_queued: true
queue_path: papers/EXECUTION_QUEUE.md
auth_ids: [AUTHORIZE_WEDGE_V1_NOISY_TRACK]
governance_status: AWAITING_TYPED_OWNER_AUTH
may_authorize_execution: false
scope_bits: [execute]
note: >
  Owner utterance "continue" is CONTINUE_SESSION (papers/OWNER_SPEECH_ACTS.md)
  and does NOT grant this auth. Requires explicit AUTHORIZE_WEDGE_V1_NOISY_TRACK
  queued in EXECUTION_QUEUE.md.
```

## Authorized (only when queued + typed)

- Generate OCR/noise variants of frozen clean docs (fixed seed)
- Score classical + E-class non-LM solvers on noisy track
- Write `results_wedge_v1_noisy_diagnostic.json`

## Not authorized

TRAINING · LM · NanoScribe · replacing clean primary · gold cherry-pick after scores · PHI corpus
