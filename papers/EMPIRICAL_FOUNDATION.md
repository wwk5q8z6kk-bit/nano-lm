# Empirical Foundation + Invariants (lockfile)

*Owner-mandated (Master Plan action #2), 2026-07-20. SHORT by design. Locks the
non-negotiables. Any change to this file requires an owner-authored commit.*

## The measured foundation (immutable JSONs in trajectory/)
1. **Field localization:** held-value copying failure lives in open-vocabulary slots;
   closed fields exactly zero across all instances (template-vs-value control).
2. **The factorial:** own-160M diluted — fullFT/200M 16.9±1.7 · LoRA/200M 7.1±1.2
   (seed band ±1.3) · fullFT/3.2B 7.0±1.0 · LoRA/3.2B 4.24±0.91 (seed band 0.00)
   ≈ pythia-160m 3.5±0.7. Interaction (weak base × full FT); tokenizer/architecture
   ~innocent.
3. **Slot diversity is causal:** +66.7 pts held-type recall (D5→D80), monotonic,
   position innocent, pre-registered.
4. **Per-type categorical competence:** flips are discrete per (model, lexical type);
   aggregates are composition arithmetic (three models, two stacks, identical metrics
   from identical states). Variance is boundary-localized.
5. **Shared residual floor:** ~15–18 clean points in BOTH stacks after every escape —
   the program's central object. Leading candidate: **lexical interference** (coverage
   refuted pre-run: wool flips with novel tokens; bee stings moves at constant
   coverage; equal-novel types split 0/100).
6. **Verification works:** grounding+absence verifiers → 100% presented precision at
   19% review load on the measured task (the existence proof for the architecture).
7. **Measurement discipline:** multi-instance eval (single-instance is biased —
   inst0's hardness is composition); training-run variance must be bounded; clean vs
   diluted metrics differ 4–5×; per-item outputs must be logged.

## The 12 invariants (from the Master Plan; enforcement = review default-reject)
(1) small model ≠ system · (2) no default global context · (3) provenance+uncertainty
on every claim · (4) contradictions first-class · (5) typed versioned memory ·
(6) authorized writes only · (7) traceable decisions · (8) independent failure ·
(9) no automatic noise growth with components · (10) functional with one small model ·
(11) these failure modes are permanent regression tests · (12) mitigation claims are
pre-registered and measured on the existing instruments.

## Regression suite (what "passes" means)
A change passes iff: clean held-out-value gap not worsened (anchors + corner configs);
per-type flip table reproduced on the frozen instruments (m0–m4, sweep_eval);
determinism cross-checks intact; provenance complete for every presented claim.
Instruments: rescore_anchors.py · batched_scorer.py (validated fast path) ·
gen_slot_diversity_eval.py pools/instances · the immutable results JSONs.
