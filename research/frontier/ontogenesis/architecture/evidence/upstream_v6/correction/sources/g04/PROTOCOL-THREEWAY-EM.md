# The three-way router on learned clusters instead of given types (registered before any confirmatory run)

Joint test with the memory-line session. It answers the typed-address caveat of the earlier three-way studies. That session's revisable addresses re-cluster entities by EM over their creation-time facts every 5 phases, with labels matched across re-clusters and no replay (`memory_rule_recluster.py` `features`, `em`; `memory_rule_transport.py` `match`; copied unchanged from research/memory-saturation-factorial and cited). In its stationary world they recovered given-type inference without labels. Here the world has rule changes, exceptions and noise.

**Design.** `threeway.run(arm="threeway", narrow=True, share=0.2, codes=..., detector=...)`. Both address modes use the memory-line session's construction: a cluster prototype plus noise, 32 prototypes. `given32` uses the hidden types as clusters; `em` uses EM clusters. The private table is the standard size in every arm (address source and table size are separate questions; the ×16 table's effect is reported in `THREEWAY-NARROW-20261122.md`). Narrow revisions use the entity's cluster (`em`) or type (`given32`), so the router stays label-free in `em`.

| Arm | Addresses | Change detector |
|---|---|---|
| `given_det` | given types | on |
| `given_nodet` | given types | off |
| `em_det` | EM clusters | on |
| `em_nodet` | EM clusters | off |

**The memory-line session's pitfall,** measured: EM clusters on creation-time facts, so after a rule change it might split a type into "before" and "after" clusters. `changed_types_split` is the fraction of (re-cluster, changed type) checks where the type's before-change and after-change members fall in different majority clusters.

## Development check (seed 1 only)

| Arm | Stable rule | Old entities revised | New entities, 1–5 phases | Exceptions kept | Noise resisted | Purity | Split | Flags / false / missed |
|---|---|---|---|---|---|---|---|---|
| given_det | 79.9% | 79.0% | 83.0% | 95.7% | 95.1% | — | — | 29 / 10 / 1 |
| given_nodet | 79.3% | 15.2% | 27.7% | 99.0% | 94.2% | — | — | — |
| em_det | 72.8% | 60.4% | 68.6% | 96.7% | 92.0% | 99.4% | 0.0 | 24 / 14 / 5 |
| em_nodet | 73.1% | 23.6% | 32.1% | 100% | 92.4% | 99.4% | 0.0 | — |

## Predictions (seeds 7601–7605, new; held on the five-seed mean, with seeds meeting the bound reported)

- E1 **EM recovers the types without a before/after split:** purity at least 0.95, and `changed_types_split` at most 0.20, in `em_det`.
- E2 **The change detector is still needed with learned clusters:** `em_det` revises at least 0.25 more old entities than `em_nodet`.
- E3 **Learned clusters cost some revision, not most of it:** `given_det` minus `em_det` on old entities revised is between 0.05 and 0.30.
- E4 **Learned clusters cost at most 0.10 of stable rule:** `given_det` minus `em_det` on stable rule is at most 0.10.
- E5 **Exceptions and noise are unaffected by the address source:** exceptions kept at least 0.93 and noise resisted at least 0.90, in all four arms.

**Reading.** E1–E4 holding: the router works on clusters it learns, at a measured cost. The typed-address caveat then shrinks to that cost, rather than holding for the whole design.
