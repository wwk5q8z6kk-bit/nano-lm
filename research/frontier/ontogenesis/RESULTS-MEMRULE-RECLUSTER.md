# Revisable addresses by re-clustering and full re-index: results

Run 2026-11-22 under [PROTOCOL-MEMRULE-RECLUSTER.md](PROTOCOL-MEMRULE-RECLUSTER.md), committed in `95d3c37` before any confirmatory run. Seeds 4101–4105. Raw files: `results/memory_rule_recluster/`. The given-types-without-re-index and derived arms are reused from `results/memory_rule_sharing/` and `results/memory_rule_derived/` (same seeds, same world). [Observed]

## Means over five seeds

| Arm | Acquisition | Inference, new entities | Retention | Corrections | Final purity |
|---|---|---|---|---|---|
| Given types, same re-index (control) | 0.900 | 0.783 | 0.923 | 0.151 | 1.000 |
| EM, 32 clusters | 0.901 | 0.767 | 0.927 | 0.144 | 0.977–0.983 |
| EM, 16 clusters (the true number) | 0.877 | 0.724 | 0.915 | 0.154 | 0.857–0.981 |
| Given types, no re-index (earlier) | 0.836 | 0.737 | 0.770 | 0.034 | — |
| Derived codes, frozen (earlier) | 0.893 | 0.302 | 0.822 | 0.034 | — |

## Against the registered predictions

- Q1 (EM with 16 clusters within 0.03 of the control on inference): **did not hold**, none of five; 0.034–0.089 below.
- Q2 (EM with 32 clusters within 0.03 of the control): **held**, four of five; 0.764–0.772 against 0.766–0.800.
- Q3 (EM with 16 clusters at least 0.30 above frozen derived codes): **held**, five of five.
- Q4 (EM with 16 clusters reaches purity at least 0.95): **did not hold**, one of five (0.857–0.981).

## Reading

Revisable addresses recover almost all of the given-type generalisation, without any label: discovered types with 32 clusters reach inference 0.767, against 0.783 for given types under the same re-index schedule, and against 0.302 for frozen data-derived codes. The two-thirds gap left by frozen codes was a limit of freezing, not of the information in the data.

The surprise is in the cluster count. With exactly as many clusters as hidden types, EM sometimes merges two types into one cluster and splits another (purity down to 0.86), and inference suffers. With twice as many clusters, it splits types rather than merging them, purity stays at 0.98, and inference follows. Over-splitting is cheap for this memory, because two clusters of the same type still each learn the type's rule from their own members. Merging is costly, because one cluster then holds two rules. So guessing too many types is safe here, and guessing exactly right is not. On the development seed the 16-cluster arm happened to do well (purity 0.986), and the prediction was written from it.

Retention and corrections rise in every re-indexed arm (corrections 0.14–0.15 against 0.03) because the re-index replays the exact store of current values. That comes from the replay, not from the clustering, as the control shows.

## Limits

- **Full re-index, not online transport.** Every 5 phases the value table is reset and retrained on an exact store of every current fact: 34,560 facts replayed over 40 phases. This shows what revisable addresses can recover, not that they can be maintained cheaply. Moving stored values to new addresses without full replay (the blueprint's Family A) is the open question.
- Clustering sees facts as shown at creation only; it never sees corrections and is not adapted to rule changes.
- Thresholds Q1 and Q4 were set from a development seed that was kinder to 16 clusters than the held-out seeds.
- 3 of 8 relations shown, 16 types, 32 values, value table, synthetic world. Not evidence about a language model.
- All fifteen runs finished on the first attempt.
