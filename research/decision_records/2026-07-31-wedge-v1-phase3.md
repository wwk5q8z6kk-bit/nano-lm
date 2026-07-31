# Decision record — Wedge v1 Phase 3

**Auth:** `AUTHORIZE_WEDGE_V1_PHASE3_LM_PROBE`
**Cascade:** non-LM first (query expand → symbolic → sentence coref) → LM only if needed
**RESULT:** `wedge_v1/results_wedge_v1_phase3.json`
**Verdict:** `ECLASS_CLOSED_WITHOUT_LM`

| Task | Method | Outcome |
|------|--------|---------|
| T35 | query synonym expansion | PRESENT `300 seconds` |
| T36 | symbolic multi-doc dose compare | PRESENT 500→850 |
| T39 | sentence-level coref | PRESENT 2 bindings |

`lm_invoked=false`. Training not authorized. A–D classical-only.
ΔU≈0 vs Phase 2 (C bump 1.05); material result is **E-class closed without generation**.
