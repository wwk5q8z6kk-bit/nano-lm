# Step-budget sweep: results

Run 2026-11-22 under [PROTOCOL-MEMRULE-STEPS.md](PROTOCOL-MEMRULE-STEPS.md), committed in `c67b4b7`. An addendum written before any result was read (`3c8d0d7`) changed how S1 is interpreted; it is applied below. Seeds 4101–4105, shared typed path, learning rate 1e-3. The 200-step values are the earlier typed M = 0 runs. Raw files: `results/memory_rule_steps/`. [Observed]

| Steps per phase | Acquisition | Inference, new entities | Retention | Corrections |
|---|---|---|---|---|
| 200 | 0.836 | 0.737 | 0.770 | 0.034 |
| 1,000 | 0.884 | 0.783 | 0.809 | 0.035 |

Per seed, corrections at 1,000 steps equal those at 200 steps on four seeds and are 0.006 higher on the fifth.

## Against the registered prediction

- S1 (corrections at 1,000 steps at most 0.05 above 200 steps): **held**, five of five.

## Reading, under the addendum

As registered, S1 holding refutes the step-budget account. The addendum, recorded before any result was read, explains why that reading is wrong: more steps for every phase also make same-type neighbours' accumulated values larger by about the same factor, so the correction's budget relative to them does not change. What S1 shows is narrower: **a uniform change in budget does not move corrections**, here or (from `RESULTS-MEMRULE-DUAL.md`) for a tenfold learning rate. The relative-budget account is tested separately in `PROTOCOL-MEMRULE-WRITEPASS.md`.

Five times the steps raised acquisition (0.84 to 0.88), inference (0.74 to 0.78) and retention (0.77 to 0.81) together, much as the tenfold learning rate did. The default of 200 steps at 1e-3 under-trains this memory in this world. Earlier comparisons between arms are unaffected, because every arm in a comparison used the same budget.

## Limits

- One additional step count.
- Thresholds set after earlier tests; the interpretation was revised before any result was read, and both the registered and revised readings are stated.
- Synthetic world, given type information, value table. Not evidence about a language model.
- All five runs finished on the first attempt, on a heavily loaded machine; timings are not measurements.
