# Protocol: do corrections take when written alone, and does giving them a larger relative budget make them take?

Written 2026-11-22, before any confirmatory run. Follows [RESULTS-MEMRULE-DUAL.md](RESULTS-MEMRULE-DUAL.md) (`a2f61a5`) and the addendum to [PROTOCOL-MEMRULE-STEPS.md](PROTOCOL-MEMRULE-STEPS.md) (`3c8d0d7`). The write-alone falsifier and the collateral caveat were proposed by another session (nano-lm-d5).

## Accounts being separated

In the typed shared-path memory, corrections take about 3% of the time, scored right after the phase that wrote them. Three accounts are on record:
1. **Conflict inside one objective:** the correction loses to the phase's same-type new facts in the summed loss.
2. **Relative budget:** a correction gets about the same movement as each earlier write (Adam moves each value by roughly the learning rate per step), so it cannot outweigh values that same-type neighbours accumulated over many phases.
3. **Reachability** (nano-lm-d5's limit in a frozen nonlinear decoder): the target value cannot win from these slots at all.

Writing corrections alone in the loss removes (1). Giving the correction pass ten times the steps, and nothing else, tests (2). If corrections take with a larger relative budget, (3) does not apply here.

## What I had seen when writing this

Development seed 4100 only:

| Correction pass | Corrections before → after the pass | Corrected entities' other facts, before → after | Same-type peers' facts, before → after | Late inference on new entities | Late acquisition |
|---|---|---|---|---|---|
| 200 steps (same as phase) | 0.031 → 0.038 | 0.757 → 0.757 | 0.784 → 0.784 | 0.736 | 0.848 |
| 2,000 steps | 0.025 → 0.944 | 0.729 → 0.716 | 0.763 → 0.759 | 0.607 | 0.773 |

## Design

`memory_rule_writepass.py`: shared typed path, learning rate 1e-3, 200 steps per phase on the phase's new shown facts only. Then, with a fresh Adam, a pass with only the phase's 16 corrections in the loss, of either 200 or 2,000 steps. Seeds 4101–4105, 10 runs. Deviation from the proposal: the 16 corrections are written together, not one pass each.

Measured each phase, last 10 phases averaged: corrections before and after the pass; collateral at writing (the corrected entities' other shown facts; shown facts of other entities of the corrected types; inference on this phase's new entities); and the usual late acquisition, inference and retention.

## Predictions

"Holds" means on at least four of the five seeds.

- W1. With a 200-step pass, corrections after the pass are at most 0.05 above before. Removing the majority alone does not make them take, so account (1) is not the main cause.
- W2. With a 2,000-step pass, corrections after the pass are at least 0.80. Account (2) is supported and (3) does not apply here.
- W3. Collateral at writing is small with the 2,000-step pass: corrected entities' other facts fall by at most 0.05, and same-type peers' facts by at most 0.02.
- W4. The cost appears later instead. Late inference on new entities (measured before each phase's correction pass) is at least 0.05 lower with the 2,000-step pass than with the 200-step pass.

## Limits

- The type information is given, not learned. Synthetic world, value table, frozen addresses.
- Thresholds were set after one development seed.
- Ten times the steps is one point. Where corrections start to take, and whether a smaller boost buys them with less later cost, is not measured.
- The correction pass knows which facts are corrections. That is a write-time label, the same kind as an explicit edit request; it is not available to a learner that must detect corrections itself (see `INVESTIGATION.md`, Question 8).
- Results under `results/memory_rule_writepass/`, created exclusively.
