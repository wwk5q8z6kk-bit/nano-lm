# Two read paths without a router: results

Run 2026-11-22 under [PROTOCOL-MEMRULE-DUAL.md](PROTOCOL-MEMRULE-DUAL.md), committed in `1f05f7e` before any confirmatory run. Seeds 4101–4105, last 10 of 40 phases. Raw files: `results/memory_rule_dual/`. The typed single-path arms at learning rate 1e-3 are reused from `results/memory_rule_sharing/` (same seeds, same world), as registered. [Observed]

## Means over five seeds

| Arm | Learning rate | Acquisition | Inference, new entities | Retention | Corrections |
|---|---|---|---|---|---|
| Shared (typed) path only | 1e-3 | 0.836 | 0.737 | 0.770 | 0.034 |
| Private (random) path only | 1e-3 | 1.000 | 0.042 | 0.959 | 0.481 |
| Shared + private, one table | 1e-3 | 0.888 | 0.720 | 0.816 | 0.040 |
| Shared path only | 1e-2 | 0.903 | 0.790 | 0.822 | 0.042 |
| Shared + private, one table | 1e-2 | 0.962 | 0.762 | 0.876 | 0.221 |

Every cell's seeds lie within 0.07 of each other; per-seed values are in the raw files.

## Against the registered predictions

- E1 (at 1e-3 the second path keeps at least 0.9 of inference and adds at least 0.03 acquisition): **held**, five of five. Inference kept 0.974–0.982 of its single-path value; acquisition rose 0.043–0.064.
- E2 (at 1e-2, acquisition at least 0.95 with inference at least 0.9 of the 1e-3 single path): **held**, five of five.
- E3 (at 1e-2, the second path adds at least 0.03 acquisition over the single path at the same rate, with inference no more than 0.02 lower): **failed** on all five seeds, narrowly. Acquisition rose 0.050–0.067, but inference fell 0.027–0.029.
- E4 (corrections stay at most 0.30 with both paths at 1e-2): **held**, five of five (0.206–0.269).

## Reading

The development seed's headline, two paths giving 0.97 acquisition with inference intact, came mostly from the learning rate. Raising the rate tenfold on the single shared path lifts acquisition from 0.84 to 0.90 and inference from 0.74 to 0.79. At that rate the second path adds a further 0.06 acquisition and 0.18 corrections, and costs about 0.03 inference. By the registered rule that cost is too large, so it is not a free addition. It is still a much better exchange than spreading reads was in the previous test, where reaching 0.96 acquisition cost two thirds of inference.

What each change buys, at a glance:
- The higher learning rate improves acquisition and inference together. The earlier tests all ran at 1e-3, which turns out to be too low for this memory in this world. No other rate was tried; 1e-2 is not claimed to be best.
- The private path improves acquisition and is the only thing that moves corrections (0.04 to 0.22).
- Neither fixes corrections. The best arm with any inference applies one correction in five.

## Not predicted, and against my own account of corrections

The protocol said corrections fail because a correction must outvote neighbours' accumulated values within one phase's step budget. The single shared path at ten times the learning rate, which moves each value about ten times further per step, applies corrections no more often (0.034 at 1e-3, 0.042 at 1e-2). A step budget predicts a rise. So the budget account is not supported for the shared path. A more likely account, not tested: in the same phase the corrected fact shares slots with new same-type facts, and the summed loss is won by the majority, so the correction loses inside a single objective rather than for lack of steps. That is a conflict, not a budget, and it differs from both my original account and nano-lm-d5's reachability limit (frozen nonlinear decoder, flat across 100–3,000 steps; commit `d6e05c3` on `research/g06-nullspace-durability`).

## Limits

- The type information in the shared addresses is given, not learned.
- Thresholds were set after one development seed.
- The private path is per-fact storage, close to a hashed exact store sharing the same table.
- Only two learning rates were tried, one of them the inherited default.
- A value table, frozen addresses and a synthetic world. Not evidence about a language model.
- All fifteen runs finished on the first attempt. The machine was shared and loaded; timings are not measurements.
