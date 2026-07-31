# Decision record — Wedge v1 Phase 3 LM probe design

**Date:** 2026-07-31  
**Trigger:** Owner "continue" after Phase 2 classical RESULT + first-principles mitigation line  
**Auth used:** docs only (no `AUTHORIZE_WEDGE_V1_PHASE3_LM_PROBE`)

## What happened

1. Phase 2 classical baseline already measured: U≈0.926, 40/40, E-class ABSTAIN by design.  
2. Continued first-principles mitigation by writing an execute-ready **Phase 3 design** that:
   - allowlists only T35/T36/T39,
   - mandates constructive faithfulness + dual estimands,
   - forbids A–D LM reopen and training until separate auth.

## Artifact

`wedge_v1/PHASE3_LM_PROBE_DESIGN.md`

## Not done

No LM calls, no training, no Phase 3 scores.
