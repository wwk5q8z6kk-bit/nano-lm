# Protocol: choosing private or shared writes without an edit label

Written 2026-11-22, before any confirmatory run. Follows [RESULTS-MEMRULE-PRIVATEFIX.md](RESULTS-MEMRULE-PRIVATEFIX.md) (`e265596`). There, corrections given an explicit label and written to a private table with a 2,000-step pass took 0.89 of the time, at a cost of 0.02 later inference. This test removes the label. Agreed with nano-lm-ad, which owns `INVESTIGATION.md`, Question 8, on these conditions: own branch, new files, `routing_loo.py` untouched, a labelled ceiling and a matched random router as baselines, false positives reported separately, no tuning on held-out seeds.

## Design

`memory_rule_routing.py`. Same two-table memory (shared typed slots plus private random slots, summed). Each phase the memory gets the phase's facts unlabelled: new entities' shown facts and that phase's 16 corrections, mixed.
- `none`: all facts trained jointly into both tables, 200 steps.
- `detector τ`: a fact is flagged when, before this phase's training, the memory's argmax differs from the given value **and** its top probability is at least τ. Unflagged facts are trained jointly, 200 steps. Flagged facts are then written alone to the private table, 2,000 steps.
- `random τ`: the same number of facts as the detector flags in that phase, chosen at random from the phase's facts and written the same way. This control can fail.
- Labelled ceiling: `private`, 2,000 steps from `results/memory_rule_privatefix/` (same seeds, same world, corrections only, with the label).

Seeds 4101–4105. Arms: none; detector at τ = 0.1 and 0.2; random at τ = 0.1 and 0.2. Twenty-five runs.

## How τ was chosen, development seed 4100 only

| Arm | Corrections | Inference, new entities | Acquisition | Fraction flagged | Regular new facts flagged (false positives) | Exceptions flagged | Corrections flagged |
|---|---|---|---|---|---|---|---|
| none | 0.056 | 0.737 | 0.902 | — | — | — | — |
| detector, τ = 0 | 0.619 | 0.048 | 1.000 | 0.951 | 0.950 | 0.966 | 0.950 |
| detector, τ = 0.05 | 0.681 | 0.544 | 0.996 | 0.467 | 0.374 | 0.979 | 0.956 |
| detector, τ = 0.1 | 0.875 | 0.707 | 0.968 | 0.176 | 0.065 | 0.627 | 0.913 |
| detector, τ = 0.2 | 0.531 | 0.729 | 0.926 | 0.067 | 0.003 | 0.281 | 0.538 |
| detector, τ = 0.4 | 0.131 | 0.736 | 0.904 | 0.015 | 0.000 | 0.059 | 0.131 |
| detector, τ = 0.6 | 0.056 | 0.737 | 0.902 | 0.003 | 0.000 | 0.018 | 0.019 |
| random, τ = 0 | 0.575 | 0.048 | 0.964 | 0.951 | — | — | — |

At τ = 0 the detector collapses. An empty memory disagrees with every fact in phase 0, everything goes private, and the shared path never learns the rule. τ = 0.1 and τ = 0.2 trade corrections against inference, so both are registered rather than one picked after the fact.

## Predictions

"Holds" means on at least four of the five seeds.

- R1. Detector τ = 0.2: corrections at least 0.40, and inference no more than 0.03 below `none`.
- R2. Detector τ = 0.1: corrections at least 0.75, and inference no more than 0.05 below `none`.
- R3. Each detector beats its count-matched random router on corrections by at least 0.20, with inference no lower.
- R4. False positives (regular new facts flagged) are at most 0.01 at τ = 0.2 and at most 0.10 at τ = 0.1.
- Recorded without a prediction: the gap to the labelled ceiling.

## Limits

- **A stationary world.** No rule ever changes. nano-lm-d5's G04 (PR #204) shows that in a world where rules change, a router that sends surprises to private slots keeps the old rule. This detector would do the same.
- The detector also flags exceptions, sending irregular facts private. Here that is what one wants; it is not a correction detector as such.
- τ was chosen on one development seed. Five held-out seeds is a small confirmation.
- Type information is given; value tables, frozen addresses, synthetic world. The private table is per-fact storage. Not evidence about a language model.
- Results under `results/memory_rule_routing/`, created exclusively.
