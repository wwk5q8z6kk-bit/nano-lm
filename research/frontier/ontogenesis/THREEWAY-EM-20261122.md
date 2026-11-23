# The three-way router on learned clusters (2026-11-22)

Joint with the memory-line session: its EM re-clustering, label matching and settings, copied and cited. Protocol, development check, predictions E1–E5 and scorer committed in `3a0e17b` before any confirmatory run. Seeds 7601–7605 (new). Raw files: `results/threeway_em/`; scored output: `results/threeway_em/verdict.json`. [Observed]

**Question.** The earlier three-way studies gave the memory the hidden types for free. Does the router still work when the memory learns its own clusters? The memory groups entities by EM over their creation-time facts every 5 phases, with 32 clusters, labels matched across re-clusters, and no replay.

| Arm | Stable rule | Old entities revised | New entities, 1–5 phases | Exceptions kept | Noise resisted | Missed changes (of 12) |
|---|---|---|---|---|---|---|
| Given types, detector | 77.0% | **70.0%** | 68.8% | 94.8% | 93.8% | 1.2 |
| Given types, no detector | 78.2% | 12.6% | 21.2% | 98.2% | 91.6% | — |
| **EM clusters, detector** | 70.7% | **50.4%** | 53.4% | 97.0% | 87.8% | 5.2 |
| EM clusters, no detector | 71.1% | 14.5% | 21.5% | 99.3% | 86.0% | — |

EM: purity 99.3%, 23.4 of 32 clusters used, and a changed type split into "before" and "after" clusters in 21% of checks (per seed: 17%, 25%, 17%, 23%, 25%).

## Against the registered predictions

| | Prediction | Result | Seeds meeting the bound |
|---|---|---|---|
| E1 | EM purity ≥ 0.95 and before/after split ≤ 0.20 | **failed, narrowly.** Purity 0.993 held; split 0.213 | 2 of 5 |
| E2 | the detector adds ≥ 0.25 revision with EM | **held**, +0.36 | 4 of 5 |
| E3 | EM costs 0.05–0.30 of revision against given types | **held**, 0.196 | 3 of 5 |
| E4 | EM costs ≤ 0.10 of stable rule | **held**, 0.063 | 5 of 5 |
| E5 | exceptions ≥ 0.93 and noise ≥ 0.90 in every arm | **failed.** Exceptions held (94.8–99.3%); noise with EM clusters 86.0–87.8% | 0 of 5 |

## What this shows

- **The router works on clusters the memory learns itself, at a measured cost.** With EM clusters and the change detector, it revises half of old entities after a rule change, against 70% with given types and 14% with no detector. It keeps 97% of exceptions. Stable-rule accuracy is 6 points lower than with given types.
- **The change detector remains the essential part:** it adds 36 points of revision with EM clusters.
- **The memory-line session's predicted pitfall does occur, but not often:** after a rule change, EM sometimes separates a type's before-change and after-change members (21% of checks). It did not happen on the development seed. Its members do not get revised together, which is one plausible reason EM misses more changes (5.2 of 12 against 1.2).
- **Noise resistance drops with learned clusters** (86–88% against 92–94%). Exceptions are unaffected. A plausible reading, not tested: with less confident shared answers, a wrong label disturbs the cluster's shared values more. The seen-twice rule itself does not depend on the addresses.
- **Seed variance again:** the given-type router revised 70% on these seeds, against 79% on the development seed and 77–82% in earlier studies.

## What this means for the typed-address caveat

The caveat shrinks from "the whole design may not transfer" to a measured cost:
- about 20 points of revision;
- about 6 points of stable rule;
- about 6 points of noise resistance;
- about four more missed changes per twelve.

The design itself transfers.

## Next

1. **Why EM misses more changes.** Two candidates, not exclusive:
   - the memory-line session's over-splitting hypothesis: each cluster learns its rule from half the data, so signatures are noisier;
   - the before/after split measured here.

   A cheap test is 16 clusters for the detector's grouping only.
2. **Noise with learned clusters:** check whether the noise loss comes from shared-table disturbance, using the shared-only readout.


## Post-hoc diagnostic: why EM misses more changes (rerun of the registered seeds, reproduced exactly)

Per rule change, over the five registered seeds (60 changes): was it detected, did EM ever split its type into before/after clusters, and over how many clusters was the type spread? Raw: `results/threeway_em/diagnostic_per_change.json`.

| | Detected | Missed |
|---|---|---|
| Type split before/after at some re-cluster | 7 | 10 |
| Never split | 27 | 16 |
| Mean clusters per changed type | 2.62 | 2.62 |

- Splits are more frequent among missed changes (10 of 26, 38%) than among detected ones (7 of 34, 21%), but most misses (16 of 26) happen without any split.
- Over-splitting (the memory-line session's hypothesis) does not separate detected from missed changes: both average 2.62 clusters per type.
- Most of the extra misses under EM are therefore unexplained by either mechanism. Masking a flagged relation from clustering (the memory-line session's proposed fix for splits) would address at most the split-related third.

## The router's revision across seed sets, by configuration (requested by the ontogenesis session)

Pooling is valid only across identical configurations. Mean, with the minimum and maximum over seeds in brackets.

| Configuration | Seeds | Old entities revised |
|---|---|---|
| Three-way, typed addresses, normal revision, private 16,384 | 7301–7305 | 86% [78%–91%] |
| same | 7401–7405 | 71% [58%–78%] |
| same | 7501–7505 | 82% [71%–89%] |
| Three-way, typed addresses, share ≥ 0.2, private 16,384 | 7501–7505 | 79% [70%–85%] |
| Three-way, typed addresses, share ≥ 0.2, private ×16 | 7501–7505 | 77% [71%–81%] |
| Three-way, given types (32-prototype construction), share ≥ 0.2 | 7601–7605 | 70% [60%–75%] |
| Three-way, EM clusters, share ≥ 0.2 | 7601–7605 | 50% [31%–66%] |

Pooled over the 15 seeds of the one configuration run three times (typed addresses, normal revision): **79% [58%–91%]**. Single five-seed means of this router should be read as ±10 points.
