# Spreading memory reads in the rule world: results

Run 2026-11-22 under [PROTOCOL-MEMRULE-SHARING.md](PROTOCOL-MEMRULE-SHARING.md), committed in `0764fb6` before any confirmatory run. Seeds 4101–4105, last 10 of 40 phases. Raw files: `results/memory_rule_sharing/`. [Observed]

## Means over five seeds

| Arm | Acquisition | Inference, new entities | Inference, old entities | Retention | Corrections | Effective slots |
|---|---|---|---|---|---|---|
| typed, M = 0 | 0.836 | 0.737 | 0.734 | 0.770 | 0.034 | 1,829–2,152 |
| typed, M = 2 | 0.827 | 0.641 | 0.635 | 0.762 | 0.042 | 2,971–3,459 |
| typed, M = 4 | 0.858 | 0.496 | 0.490 | 0.795 | 0.085 | 5,313–6,006 |
| typed, M = 6 | 0.959 | 0.246 | 0.248 | 0.898 | 0.244 | 10,220–10,879 |
| typed, M = 8 | 1.000 | 0.042 | 0.045 | 0.959 | 0.481 | 14,830–14,879 |
| random, M = 0 | 0.622 | 0.106 | 0.109 | 0.584 | 0.196 | 2,825–3,300 |

Seed-to-seed spread is small: within 0.04 for every cell. Per-seed values are in the raw files.

## Against the registered predictions

- H1 (typed addresses generalise, at least 0.50; random ones do not, at most 0.15): **held**, five of five.
- H2 (fully random slots destroy generalisation, at most 0.10): **held**, five of five (0.038–0.051).
- H3 (inference never rises and acquisition never falls by more than 0.02 per dose step): **held**, four of five. Seed 4105's acquisition fell 0.024 from M = 0 to M = 2.
- H4 (some dose keeps acquisition at least 0.95 and inference at least 0.8 of its undiluted value): **does not hold**, none of five. The best candidate on every seed was M = 6: acquisition 0.946–0.966, but only 0.33–0.34 of the inference. M = 2 keeps 0.86–0.88 of the inference but leaves acquisition unchanged (0.82–0.84).

## Reading

In this world spreading reads trades generalisation for acquisition, and the trade is worse than one for one. Inference falls by 0.10, 0.15, 0.25 and 0.20 over successive doses, while acquisition does not move until M = 6 and only reaches 0.95 once two thirds of the generalisation is gone. The arbitrary-fact result, where spreading half of each query's reads looked like a free remedy, does not carry over to a world with structure.

The same structure that generalises also blocks individual facts. With typed addresses undiluted, the memory applies only 3% of corrections, below its 20% with random addresses. A correction written to slots shared with same-type entities is outvoted by them. Corrections rise steadily with the dose to 0.48 at M = 8. Acquisition (0.84) is below 1 for a related reason: the 10% exceptions and same-type neighbours compete for the same slots.

So for a fixed-address memory, where an entity's facts live and where its type's regularities live is the same decision. Making addresses shared generalises and resists corrections; making them private acquires and corrects but cannot generalise. Resetting the optimiser (used in every arm here) does not remove that. Separating the two, for example a shared generalising read plus a private exact read with a learned choice between them, is a design question this test raises, not one it answers. The routing results in `INVESTIGATION.md`, Question 8, are the existing work on that choice.

## Limits

- The type information in the addresses is given, not learned.
- Thresholds were set after one development seed.
- Frozen addresses, a value table and a synthetic world. Not evidence about a language model.
- The machine was heavily loaded by unrelated work; timings are not measurements.
- No run was interrupted. All thirty finished on the first attempt.
