# Revisable addresses without full replay: results

Run 2026-11-22 under [PROTOCOL-MEMRULE-TRANSPORT.md](PROTOCOL-MEMRULE-TRANSPORT.md), committed in `0bf6daf` before any confirmatory run. Seeds 4101–4105. Raw files: `results/memory_rule_transport/`. Full re-index, given types and derived codes are reused from earlier results on the same seeds. [Observed]

## Means over five seeds

| Arm | Inference, new entities | Acquisition | Retention | Corrections | Data replayed or transported |
|---|---|---|---|---|---|
| Re-cluster, full re-index (exact store) | 0.767 | 0.901 | 0.927 | 0.144 | 34,560 facts replayed |
| Given types, no re-index | 0.737 | 0.836 | 0.770 | 0.034 | none |
| **Re-cluster, drop moved entities' values** | **0.699** | 0.830 | 0.754 | 0.039 | none |
| Re-cluster, transport own predictions | 0.683 | 0.822 | 0.752 | 0.039 | 1,422–1,482 keys distilled |
| Derived codes, frozen | 0.302 | 0.893 | 0.822 | 0.034 | none |

474–494 entities changed group over the 40 phases; final purity 0.977–0.983.

## Against the registered predictions

- T1 (drop within 0.03 of given types without re-index): **did not hold**, two of five. Gaps 0.015–0.067, mean 0.038.
- T2 (transport does not beat drop by 0.02): **held**, five of five. Transport was lower on every seed (0.004–0.023).
- T3 (full re-index beats drop by at least 0.03): **held**, five of five.
- T4 (drop at least 0.35 above frozen derived codes): **held**, five of five.

## Reading

Revisable addresses work without any replay, but not quite as well as given types: re-clustering alone reaches 0.699, against 0.737 for given types and 0.302 for frozen codes. Most of the full re-index's further gain, to 0.767, comes from replaying the exact store, which is extra training on every fact.

Transporting a moved entity's values by distilling the memory's own old predictions made things slightly worse on every seed. The old prediction for a moved entity comes from slots it is leaving, so it carries the wrong group's rule as often as its own facts. Writing that into the new group's shared slots adds noise to the group's rule. Simply letting the moved entity's facts lapse, and letting the new group's rule answer for it, is better here.

So in this memory, what makes generalisation work is that the grouping can be revised; moving old values to new addresses adds nothing. That argues against investing in online transport for this kind of memory. For a memory where individual facts matter more than the rule (exceptions, corrections), dropping them would cost more, and this test does not measure that cost separately.

## Limits

- One transport method (self-distillation of the soft prediction). A transport that used the entity's own written values would need an exact store, which this test excluded on purpose.
- Label matching across re-clusters is part of the method; without it most entities would move at every re-cluster.
- Clustering sees creation-time facts only. Synthetic world, value table. Not evidence about a language model.
- All ten runs finished on the first attempt.
