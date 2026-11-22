# G06b: localised protection keeps corrections and the rule (2026-11-22)

`nullspace_local.py`. Predictions L1–L6 committed in `5adc2b0` before any run; scorer in `b916aba` before any result. Seeds 8001–8003. Raw files: `results/nullspace_local/`; scored output: `results/nullspace_local/verdict.json`. [Observed]

**Question.** G06 (`NULLSPACE-DURABILITY-20261122.md`) found that projecting later decoder updates off *all* old entities' addresses keeps corrections (98.5%) but costs rule inference for new entities (66% against 99.7%). Is that cost a property of protection, or of protecting too much in too few dimensions? Suggested by the memory-line session; the curve over α was suggested by the ontogenesis session.

**Answer.** Of protecting too much. Protecting only what a correction needs keeps both, at the original width:
- protecting the corrected relation's output: 94% of corrections last, inference 99.3%;
- with more width (m = 96), protecting corrected entities: 99.6%, 100% and 99.1%.

| Arm | Width | Corrections 20+ phases later | Old facts, corrected entities | Unshown relations of new entities | Decoder room left |
|---|---|---|---|---|---|
| No protection | 24 | 21.5% | 83.5% | 99.7% | — |
| All old entities, α = 1 (= G06) | 24 | 98.5% | 100% | **66.2%** | 0.03% |
| All old entities, α = 10 | 24 | 91.1% | 99.1% | 91.2% | 0.4% |
| All old entities, α = 100 | 24 | 52.5% | 94.9% | 99.6% | 7% |
| Corrected entities only | 24 | 98.4% | 99.9% | 91.4% | 0.1% |
| **Corrected relation only** | 24 | **93.9%** | 92.1% | **99.3%** | 2% |
| No protection | 96 | 21.6% | 91.9% | 99.8% | — |
| All old entities | 96 | 100% | 100% | 80.7% | 0.3% |
| **Corrected entities only** | 96 | **99.6%** | **100%** | **99.1%** | 1% |
| Corrected relation only | 96 | 99.6% | 93.0% | 99.9% | 36% |

Means of three seeds. New facts were learned at 99.9–100% in every arm. Old facts on uncorrected entities stayed at 99.9–100% in every arm.

## Against the registered predictions

- L1 (reproduction): **held**. `none` at m = 24 matches G06 seed for seed. The all-entity projector matches G06 `owm_1` to four decimals.
- L2 (protecting corrected entities at m = 24 costs as much as protecting all, inference ≤ 0.80): **failed**, in the favourable direction: 91.4%. My crowding argument overstated the cost: with the soft projector, a direction used only by corrected entities is not fully frozen.
- L3 (protecting the corrected relation at m = 24: corrections ≥ 0.90 and inference ≥ 0.85): **held**, 93.9% and 99.3%.
- L4 (the same at m = 96: corrections ≥ 0.90, inference within 0.03 of no protection): **held**, 99.6% and 99.9% against 99.8%.
- L5 (protecting all old entities still costs inference at m = 96, ≤ 0.85): **held**, 80.7%.
- L6 (α traces a monotone trade-off): **held**. Corrections 98.5% → 91.1% → 52.5% and inference 66.2% → 91.2% → 99.6% as α goes 1 → 10 → 100.

The registered reading for L3 or L4 holding applies: protecting only the corrected outputs separates durability from rule use; the G06 cost came from protecting everything.

## What this shows

- **The trade-off reported in G06 was not fundamental.** It came from what was protected. Guarding every old entity freezes the decoder directions the rule needs for new entities. Guarding only the corrected answers leaves those directions free.
- **Which scope to protect is a second trade-off, inside the good region.**
  - Protecting the corrected relation keeps the most room (2% at m = 24, 36% at m = 96) and the best inference.
  - But it leaves the corrected entities' other answers exposed (92–93% kept, from 83.5% unprotected).
  - Protecting the corrected entities keeps those answers too (99.9–100%), and needs width to keep inference (91% at m = 24, 99% at m = 96).
- **Uniform weak protection (α = 10) is worse than localised protection** at the same width: 91% and 91%, against 94% and 99%.
- This answers the memory-line session's question from the decoder side: shared structure can carry the rule while a private, localised guard holds the corrections, provided the guard covers only what was corrected.

## Limits

Bilinear model, one synthetic world, three seeds, one write rule (computed address edit), α = 1 for the localised arms. The projectors are exact linear algebra on a bilinear readout; a deep network's readout is not linear in its weights, so this is a mechanism result, not evidence it scales. OWM and output-localised protection are published ideas. What is measured here is that, in this world, localisation removes the G06 cost. Not a language-model result.

## Next discriminating tests

1. **A nonlinear readout:** the same arms with a two-layer decoder, protecting with gradient projection (OGD) rather than an exact projector.
2. **The cost of the guard:** a projector per relation is m × m per relation, recomputed each phase. Measure time and memory against simply storing corrections (the external store, which has both properties trivially).
3. **Many more corrections:** raise from 16 per phase until the localised guard crowds out room, to find where the separation breaks.
