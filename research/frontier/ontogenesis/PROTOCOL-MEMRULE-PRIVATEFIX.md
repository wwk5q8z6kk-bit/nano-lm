# Protocol: write corrections into a private slot instead of the shared ones

Written 2026-11-22, before any confirmatory run. Follows [RESULTS-MEMRULE-WRITEPASS.md](RESULTS-MEMRULE-WRITEPASS.md) (`9ee8701`). There, corrections forced into the shared typed slots took 0.92 of the time, but later inference on new entities fell from 0.74 to 0.61, because the corrected slots are shared with same-type entities. The write was already as local as the addressing allows. That pointed to writing somewhere else instead.

## Design

`memory_rule_privatefix.py`. Every query reads two separate value tables and sums them: the shared typed slots, and 8 private uniformly random slots of its own table. Each phase trains both tables on the phase's new shown facts (learning rate 1e-3, 200 steps, Adam reset). Then the phase's corrections are written alone in the loss with a fresh Adam, updating either both tables or the private table only. Seeds 4101–4105.

Arms, 15 runs:
- `private`, 200-step correction pass: the baseline for later costs
- `private`, 2,000-step correction pass
- `both`, 2,000-step correction pass

## What I had seen when writing this

Development seed 4100 only:

| Correction pass | Corrections after the pass | Later inference, new entities | Retention | Collateral at writing (own, peers) |
|---|---|---|---|---|
| private, 200 steps | 0.038 | 0.736 | 0.832 | 0.000, 0.000 |
| private, 2,000 steps | 0.925 | 0.716 | 0.896 | 0.000, 0.000 |
| both, 2,000 steps | 1.000 | 0.650 | 0.885 | 0.016, 0.001 |

A private correction still needs a larger budget, because it must outweigh the shared path's accumulated value in the sum. Once it has one, it costs little later inference on this seed.

## Predictions

"Holds" means on at least four of the five seeds.

- P1. Private 2,000-step corrections reach at least 0.80 after the pass.
- P2. Private 2,000-step late inference on new entities is no more than 0.03 below the private 200-step arm. Most of the later cost is avoided.
- P3. Both-table 2,000-step late inference is at least 0.05 below the private 200-step arm. The cost reappears when the shared table is written.
- P4. Collateral at writing with the private 2,000-step pass is at most 0.01 on the corrected entities' other facts and on same-type peers.

## Limits

- The correction pass is told which facts are corrections and to write them privately. A learner that must detect corrections itself would not have that label.
- The private table is per-fact storage, close to a hashed exact store. Two tables double the value storage of the single-path memory.
- Type information is given; value tables, frozen addresses, synthetic world. Not evidence about a language model.
- Thresholds were set after one development seed.
- Results under `results/memory_rule_privatefix/`, created exclusively.
