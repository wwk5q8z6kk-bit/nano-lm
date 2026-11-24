# Protocol: revisable addresses without full replay

Written 2026-11-22, before any confirmatory run. Follows [RESULTS-MEMRULE-RECLUSTER.md](RESULTS-MEMRULE-RECLUSTER.md) (`09706ff`). There, re-clustering plus a full re-index (reset the value table, retrain on an exact store of every current fact) recovered the given-type inference, at the cost of replaying 34,560 facts. The question here is how much of that needs the replay at all.

## Design

`memory_rule_transport.py`: the same memory, EM with 32 clusters, re-cluster every 5 phases. The table is never reset and no values are stored outside it. After each re-cluster the new labels are matched to the old ones (Hungarian assignment on co-membership), so an entity keeps its prototype unless its group really changed. Only the entities whose group changed ("moved") get new addresses. Two modes:
- `drop`: the moved entities' new addresses are written nothing. Their stored facts are lost from the read, and only later training refills them.
- `transport`: before moving, the memory reads its own soft prediction for each written key of a moved entity from the old addresses (an index of which keys were written, no values), then trains the new addresses towards it (KL), 1,000 full-batch steps on the moved keys only.

Seeds 4101–4105, 10 runs. Reused from the same seeds: full re-index with 32 clusters (`results/memory_rule_recluster/k32-em-*`), given types without re-index (`results/memory_rule_sharing/typed-m0-*`), and frozen derived codes (`results/memory_rule_derived/`).

## What I had seen when writing this

Development seed 4100 only:

| Arm | Inference, new entities | Acquisition | Retention | Corrections | Entities moved | Keys transported |
|---|---|---|---|---|---|---|
| drop | 0.717 | 0.842 | 0.767 | 0.050 | 434 | 0 |
| transport | 0.697 | 0.827 | 0.767 | 0.038 | 434 | 1,302 |
| full re-index (earlier pilot) | 0.777 | 0.909 | 0.937 | 0.175 | — | 34,560 replayed |
| given types, no re-index (earlier) | 0.740 | 0.850 | 0.783 | 0.038 | — | 0 |
| derived, frozen (earlier) | 0.310 | 0.904 | 0.827 | 0.044 | — | 0 |

On this seed, re-clustering alone, with no replay and no transport, nearly matches given types without re-index. Transporting the moved entities' predictions did not help.

## Predictions

"Holds" means on at least four of the five seeds.

- T1. `drop` inference is within 0.03 of given types without re-index. Revisable clustering alone, with no replay, recovers the given-type inference at the same training budget.
- T2. `transport` does not beat `drop` on inference by 0.02 or more.
- T3. The full re-index beats `drop` on inference by at least 0.03. The rest of its gain comes from replaying the exact store.
- T4. `drop` inference is at least 0.35 above frozen derived codes.

## Limits

- The label matching assumes cluster identity is worth keeping across re-clusters. Without it, every re-cluster would move most entities.
- `transport` distils the memory's own prediction, errors included. It is one simple transport, not the best possible.
- `drop` loses the moved entities' facts from the read; its retention is measured, not assumed.
- Clustering sees creation-time facts only. Synthetic world, value table. Not evidence about a language model.
- Results under `results/memory_rule_transport/`, created exclusively.
