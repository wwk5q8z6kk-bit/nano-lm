# Decision record — Wedge v1 Phase 3 E-class probe

**Date:** 2026-07-31  
**Auth:** `AUTHORIZE_WEDGE_V1_PHASE3_ECLASS_PROBE`  
**Trigger:** Owner “proceed” after Phase 2  

## RESULT

| Field | Value |
|-------|--------|
| Artifact | `wedge_v1/results_wedge_v1_phase3_eclass.json` |
| Verdict | `ECLASS_CLOSED_WITHOUT_LM` |
| Classical U | ≈ 0.891 |
| Hybrid U | ≈ 0.900 |
| ΔU | ≈ +0.009 (< δ=0.05) |
| E-class accuracy | 1.0 (T35/T36/T39) |
| LM invoked | **false** |

## Interpretation (scoped)

Cheapest-sufficient non-LM probes (query expansion, symbolic dose compare, sentence coref)
closed the E-class abstentions on this clean synthetic track. ΔU did **not** clear δ=0.05,
so there is **no** registry admission pressure for an LM solver. Training / NanoScribe /
general LM productization remain STOP.

## Next (requires new owner auth)

- Idle / M0 hygiene  
- `AUTHORIZE_WEDGE_V1_U_FREEZE`  
- `AUTHORIZE_WEDGE_V1_OWNER_CORPUS`  
- Separate LM probe auth only if owner insists on capability bakeoff despite closed E-class
