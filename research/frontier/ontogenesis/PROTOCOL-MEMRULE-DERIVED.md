# Protocol: addresses built from the data instead of from given types

Written 2026-11-22, before any confirmatory run. Every result in the typed-memory line (`RESULTS-MEMRULE-SHARING.md` through `RESULTS-MEMRULE-ROUTING.md`, and nano-lm-d5's joint three-way study) gave the memory each entity's hidden type. That is the largest standing limit on all of them. This test asks how much of the typed addresses' generalisation the memory gets when it must build entity vectors from what it is shown.

## Design

`memory_rule_derived.py`, `derived` codes. An entity's vector is the sum of a fixed random vector for each (relation, value) pair shown at its creation, normalised to the same expected norm as the earlier arms. It is fixed at creation; later corrections do not change it. No type label is used. Everything else as the earlier single-shared-path arm: `cab_s.py`'s world, 16,384 slots read 8 at a time, learning rate 1e-3, 200 steps per phase, Adam reset. Seeds 4101–4105; five runs. The typed (oracle) and random arms are reused from `results/memory_rule_sharing/` (same seeds, same world).

## What I had seen when writing this

Development seed 4100 only:

| Codes | Acquisition | Inference, new entities | Retention | Corrections |
|---|---|---|---|---|
| typed (given types) | 0.850 | 0.740 | 0.783 | 0.038 |
| derived (from shown facts) | 0.904 | 0.310 | 0.827 | 0.044 |
| random | 0.666 | 0.103 | 0.637 | 0.225 |

Also tried on the development seed and abandoned: label-free online clustering. A new entity joined the existing cluster whose (relation, value) counts best predicted its shown facts, or opened a new cluster. It collapsed to 3–6 clusters with type purity 0.15–0.24 at three settings of the new-cluster cost. With three shown facts and 32 values, a cluster mixing two types still predicts better than a fresh one, so greedy online assignment never splits. Recovering types needs global, revisable inference, which these frozen addresses cannot take. `cluster_online` stays in the script as a record; it is not a registered arm.

## Predictions

"Holds" means on at least four of the five seeds, comparing each seed's derived run with that seed's typed and random runs.

- G1. Derived inference is at least 0.15 above random and at least 0.25 below typed: real but partial generalisation from data.
- G2. Derived acquisition is at least that of typed. Data-derived addresses concentrate less than given types (development seed: 1,788 against 2,117 effective slots).
- Recorded without a prediction: the fraction of the typed–random inference gap recovered, (derived − random) / (typed − random).

## Limits

- One way of building codes from data, chosen for simplicity, not tuned.
- Entities share shown relations by chance: 3 of 8 relations shown means two entities share about 1.1 shown relations on average. That bounds how similar same-type codes can be. A world showing more relations per entity would make this easier.
- Value table, frozen addresses, synthetic world. Not evidence about a language model.
- Results under `results/memory_rule_derived/`, created exclusively.
