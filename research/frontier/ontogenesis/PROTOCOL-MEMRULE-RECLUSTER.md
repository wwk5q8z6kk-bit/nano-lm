# Protocol: revisable addresses by periodic re-clustering and full re-index

Written 2026-11-22, before any confirmatory run. Follows [RESULTS-MEMRULE-DERIVED.md](RESULTS-MEMRULE-DERIVED.md) (`f22c020`). There, addresses built from shown facts and frozen at creation recovered about a third of the given-type inference, and a greedy online clustering collapsed. The question here is whether the missing two thirds is a limit of the information or of frozen addresses.

## What I had seen when writing this

Development seed 4100 only.

1. **Batch EM recovers the types.** EM for a mixture of categoricals over all entities' facts shown at creation (16 clusters, 5 restarts, no labels) recovered the hidden types with purity 0.912 after 5 phases (320 entities), 0.919 after 10, 0.924 after 20 and 0.986 after 40. The information is there.
2. **Pilots of `memory_rule_recluster.py`** (re-cluster and re-index every 5 phases, 1,000 re-index steps):

| Arm | Late acquisition | Inference, new entities | Retention | Corrections | Final purity | Facts replayed |
|---|---|---|---|---|---|---|
| EM, 16 clusters | 0.904 | 0.768 | 0.930 | 0.163 | 0.986 | 34,560 |
| EM, 32 clusters | 0.909 | 0.777 | 0.937 | 0.175 | 0.984 | 34,560 |
| given types, same re-index schedule | 0.913 | 0.787 | 0.930 | 0.150 | 1.000 | 34,560 |
| (earlier) given types, no re-index | 0.850 | 0.740 | 0.783 | 0.038 | — | 0 |
| (earlier) derived, frozen | 0.904 | 0.310 | 0.827 | 0.044 | — | 0 |

The re-index retrains on an exact store of every current fact, so it raises retention and corrections by replay alone. The fair comparison for discovered against given types is therefore the given-types arm with the **same** re-index schedule, which was added for that reason.

## Design

`memory_rule_recluster.py`: single shared path, 16,384 slots read 8 at a time, Adam 1e-3, 200 steps per phase on that phase's facts, reset each phase. Entity vector = prototype of the entity's cluster plus noise. Every 5 phases: re-cluster every entity seen so far by EM over its facts shown at creation (no labels), rebuild every address, reset the value table, and retrain it for 1,000 full-batch steps on the exact store of all current facts written so far. Between re-indexes, a new entity takes the cluster with the highest posterior under the current EM parameters. Seeds 4101–4105.

Arms, 15 runs:
- EM, 16 clusters (the true number of types is used, which is a mild oracle)
- EM, 32 clusters (does not need the number of types)
- given types with the same re-index schedule (control)

Reused, same seeds: given types without re-index, and derived frozen codes.

## Predictions

"Holds" means on at least four of the five seeds.

- Q1. EM with 16 clusters reaches inference within 0.03 of the given-types re-index control.
- Q2. EM with 32 clusters does the same, so knowing the number of types is not needed.
- Q3. EM with 16 clusters exceeds the frozen derived codes on inference by at least 0.30.
- Q4. EM with 16 clusters reaches final purity of at least 0.95.

## Limits

- **The re-index is the blueprint's "full re-index" baseline, not an online method.** It needs an exact store of every fact and replays it all every 5 phases (34,560 facts over 40 phases here). It shows what revisable addresses can recover, not that they can be maintained cheaply. A transport that moves stored values to new addresses without full replay (Family A) is the open question this leaves.
- Clustering sees only facts as shown at creation. It never sees corrections, and is not adapted to rule changes.
- 3 of 8 relations shown; 16 types; 32 values; synthetic world; value table. Not evidence about a language model.
- Results under `results/memory_rule_recluster/`, created exclusively.
