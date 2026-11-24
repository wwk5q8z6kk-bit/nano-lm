# G06: protecting old answers during later training (2026-11-22)

`nullspace_durability.py`. Predictions N1–N6 committed in `9a6db1d` before any run; scorer committed in `bc56027` before any run. Seeds 8001–8003. Raw files: `results/nullspace_durability/`; scored output: `results/nullspace_durability/verdict.json`. [Observed]

**Question.** Corrections written by the computed address edit protect the whole entity at writing (97%, INVESTIGATION.md Question 9 review checks), yet only 21.5% survive 20+ phases. The decay comes through the shared decoder, which every later phase trains. If later decoder updates are kept out of the directions that old entities' addresses use (orthogonal weight modification, OWM; Zeng et al. 2019, published prior art), do corrections last? What does that cost?

**Answer.** Corrections last, at 98.5–98.8% after 20+ phases. The cost is the rule. With the decoder held still in those directions, new entities still learn the facts they are shown (95–100%), but their unshown relations are inferred correctly only 56–66% of the time, against 99.7% without protection. Durability of old answers and use of the rule for new entities trade off through the shared decoder.

| Arm | New facts learned | Correction right at writing | Correction 20+ phases later | Old facts, corrected entities | Old facts, other entities | Unshown relations of new entities | Decoder room left |
|---|---|---|---|---|---|---|---|
| none (reproduces `constrained`) | 100% | 99.8% | 21.5% | 83.5% | 99.9% | **99.7%** | — |
| OWM, α = 1 | 99.9% | 98.7% | **98.5%** | **100%** | 100% | 66.3% | 0.03% |
| OWM, α = 0.01 | 94.8% | 98.9% | 98.8% | 94.1% | 94.7% | 56.2% | 0% |
| OWM, α = 0.0001 | 94.5% | 98.9% | 98.8% | 94.0% | 94.4% | 56.1% | 0% |
| Rehearse every correction (no projection) | 100% | 96.8% | 97.4% | **40.4%** | 99.9% | 98.4% | — |

Means of three seeds; per-seed values are in the result files. Old facts: regular facts 20+ phases old. "Decoder room left" is trace(P)/m at the last phase.

## Against the registered predictions

- N1 (`none` reproduces the recorded `constrained` arm exactly): **held**. Per-seed values 0.2243, 0.2383 and 0.1815 are identical to the recorded ones.
- N2 (strongest projection: corrections ≥ 0.80 later and corrected entities' old facts ≥ 0.95): **failed**, narrowly. Corrections held (98.8%), but old facts on corrected entities were 94.0%. At α = 1 both parts hold (98.5%, 100%); that setting was not the one predicted.
- N3 (strongest projection keeps learning new facts within 0.05): **failed**, narrowly: 94.5% against 100%.
- N4 (strongest projection keeps the rule usable for new entities within 0.05): **failed**, badly: 56.1% against 99.7%. **This is the main cost, and I did not predict it.**
- N5 (rehearsing corrections: ≥ 0.90 durable, new learning within 0.02): **held**: 97.4% and 100%. It also cut the corrected entities' other old facts from 83.5% to 40.4%, which N5 did not measure.
- N6 (strongest projection leaves ≤ 5% of the decoder free): **held**. Every projected arm ends with essentially no free decoder direction, because 24 dimensions are spanned after the first phase.

The registered reading for "N2 holding with N3 failing" was that durability and acquisition trade off. What actually trades off is durability against *inference* for new entities; learning their shown facts is nearly free.

## What this shows

- **Decay is a property of unprotected shared weights, not of the write.** The same written corrections that fall to 21.5% stay at 98.5% once later decoder updates cannot move old addresses' outputs.
- **The decoder carries the rule.** Freezing it in old directions (which, in 24 dimensions, is all directions) leaves new entities able to memorise their three shown facts through their own addresses. They no longer land where the rule can answer their other five relations. Generalisation for new entities needs a decoder that keeps adapting.
- **Rehearsing corrections is the wrong simple baseline here.** It keeps the corrections, but it repeats the damaging single-fact write every phase. The corrected entities lose most of their other answers (40%). Rehearsing a correction without the entity's other answers reproduces the original interference.
- **No arm gets all three:** durable corrections, intact old facts and usable rules for new entities. α = 1 comes closest on the first two (98.5%, 100%) and gives up a third of the inference.

## Limits

Bilinear model in 24 dimensions, one synthetic world, three seeds, one write rule (the computed address edit). In this world the decoder is spanned after the first phase, so the projection becomes a near-freeze. A wider model (m much larger than the number of entities) would leave room and might not show the inference cost. OWM is published, so the mechanism is not new; what is new is the measured trade-off in this program's world. Not a language-model result.

## Next discriminating tests

1. **Width:** repeat with m = 256 and m = 1024. If room stays above zero and inference recovers while durability holds, the cost is dimensional crowding, not a property of protection.
2. **Protect only the corrected entities:** project later updates off the span of corrected entities' addresses only (16 per phase), not all old addresses. This keeps most of the decoder free for the rule.
3. **Rehearse the corrected entity whole:** rehearse each correction with the entity's other current answers (from the model itself, as the computed edit does), not alone.
