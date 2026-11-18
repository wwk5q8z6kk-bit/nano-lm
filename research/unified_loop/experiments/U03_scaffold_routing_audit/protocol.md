# Experiment U-03 — Factorial Scaffold Attribution (`E23 C7-Omega`) & Routing Paradox Diagnosis (`E24 HEM-ATRD`)

- **Experiment ID:** `EXP-U03`
- **Status:** Preregistered before execution
- **Hardware / Cost:** Local Apple Silicon M3 Max (`cpu`), `$0.00` paid compute
- **Related Hypotheses:** `H-15` (Gauge-Covariant Token-Graph), `H-17` (Frontier De Bruijn), `H-21` (`HEM-ATRD`), `H-31` (Lyapunov Guard)

## Primary Questions

1. **Part A (`E23 C7-Omega` Neural vs Symbolic Attribution):** In `scripts/run_e23_dynamic_debruijn_cognition.py`, arms `B1..B4` receive Python-computed `SymbolicBeliefState` binary oracle features (`sym_feats`: discovered, inspected, `verified_fault_entity`, patched, `needs_clarify`, `tests_verified_pass`), whereas `B0` receives `sym_feats=0`. In a clean factorial sweep (`sym_feats` `ZERO` vs `ON` x `canonical_mode` x `shield`) across `3` seeds plus a `0`-param deterministic solver:
   - How much of `C7-Omega`'s `1.000` Deep-OOD (`N=14, 4-hop`) performance survives in pure neural weights when `sym_feats=0` (`A4_frontier_sym0` and `A6_frontier_sym0_shield`)?
   - How much does adding `sym_feats=ON` alone help `raw` or `root_levels`?
   - Kill criterion for pure-neural `C7` claim (`H-17`): **KILL pure-neural attribution** if `A4_frontier_sym0` loses `>30 pp` vs `A5_frontier_symON` (`sym_feats` drives most of the gain).

2. **Part B (`E24 HEM-ATRD` vs `X5 Shannon` Routing Paradox):** In `reports/E24/hybrid_entropy_margin_routing.json`, `H1_x5_full_vocab_shannon` achieved `1.000 ± 0.000` counterfactual 8-way recall, while `H2_x8` dropped to `0.430 ± 0.494`, `H3` to `0.729 ± 0.245`, and `H4_hem_atrd_topk_shield` to `0.839 ± 0.280` (with `hard_query_route_frac = 0.951`). Across `5` seeds (`0..4`), measure per-seed:
   - Pass-1 CF accuracy (`pass1_cf_acc`), unconditional Pass-2 CF accuracy (`raw_pass2_cf_acc`), training-time query routing fraction (`train_last50_query_route_frac`), eval routing fraction, and Lyapunov shield rejection rate (`eval_routed_rejected_frac`).
   - Test `H5_hem_train_warmup` (routing warmup / train floor so Pass-2 does not starve during training) and `H6_hem_train_warmup_margin_tol` to verify whether training-time routing starvation or eval-time Lyapunov over-rejection caused `H4`'s `16.1%` deficit vs `X5`.
