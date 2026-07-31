# wedge_v1 — Local research document intelligence

Phase 1 lock: `papers/WEDGE_V1.md`  
Phase 2: `AUTH_RECORD.md` → `results_wedge_v1_classical.json`  
Phase 3 E-class: `AUTH_PHASE3.md` → `results_wedge_v1_phase3_eclass.json`

```bash
python -m wedge_v1.build_corpus
python -m wedge_v1.run_classical_baseline
python -m wedge_v1.run_phase3_eclass
```

**Latest:** E-class closed with query expansion + symbolic compare + coref-lite.  
LM **not** indicated (`lm_still_needed=false`). Hybrid ΔU < δ — no LM registry admit.

Not a Layer-1 Evidence Ledger claim until owner promotion.
