# Experiment U-04 — NanoScribe `C12–C14` Grounded Verifier Recall Audit & Live Qwen2.5-1.5B Grounded Menu Evaluation

- **Experiment ID:** `EXP-U04`
- **Status:** Preregistered before execution
- **Hardware / Cost:** Local Apple Silicon M3 Max (`cpu` / `mps`), `$0.00` paid compute (uses local cached `Qwen2.5-1.5B-Instruct`)

## Primary Questions

1. **Part 1 (Forensic Diagnosis of `76/276` Gold Drops & `42/276` Menu Misses):** Why does `Gold x Real Verifier` silently drop `76/276` (`27.5%`) asserted (`S`/`A`) gold slots across `C12` (`39/104`), `C13` (`29/92`), and `C14` (`8/80`), while maintaining `0/7` hazard admissions?
2. **Part 2 (Minimal Non-Hazard Verifier/Extractor Fix + Zero-Regression Verification):** Can a minimal, general, non-hazard fix to `nanoscribe/grounded.py` recover rejected valid gold spans on `C12–C14` with **0/7** hazard admissions, **0** `N-03` custody violations, and `pytest -q nanoscribe` 100% green?
3. **Part 3 (Live `Qwen2.5-1.5B-Instruct` on `grounded` Menu):** Compare live `LocalCompactPredictor` (`Qwen2.5-1.5B-Instruct`, `candidate_policy="grounded"`, `decoding="constrained"`, `prefix_cache=True`) vs deterministic `grounded` heuristic / rule baseline vs old `context` saved runs (`8/276`) on **`C14_retention_validation_20261026` (`24` cases)** (expanding to `C12–C13` if runtime fits). Separate Track 1 (raw Qwen choices before verifier) vs Track 2 (end-to-end verified note custody & asserted retention).
