# Protocol: two read paths without a router in the rule world

Written 2026-11-22, before any confirmatory run. Follows [RESULTS-MEMRULE-SHARING.md](RESULTS-MEMRULE-SHARING.md) (`6333bcf`). There, a fixed-address memory whose addresses carry the hidden type generalised (0.74 inference) but acquired (0.84) and corrected (0.03) poorly. Spreading its reads traded one for the other, and no dose kept both.

## Question

If every query reads two sets of slots and sums them, without deciding between them, does the memory keep the shared path's generalisation and gain the private path's acquisition? The two paths are a shared set at the typed projection addresses and a private set of uniformly random slots. Unwritten private slots stay at zero, so an unseen fact is answered by the shared path alone. `INVESTIGATION.md`, Question 8, tested the alternative of a write-time router; a router's false alarms cost generalisation there. This arm has no router.

## What I had seen when writing this

Development seed 4100 only, not results:

| Arm | Learning rate | Acquisition | Inference, new entities | Retention | Corrections |
|---|---|---|---|---|---|
| typed single path (earlier test) | 1e-3 | 0.850 | 0.740 | 0.783 | 0.038 |
| dual, one shared table | 1e-3 | 0.901 | 0.718 | 0.832 | 0.063 |
| dual, two tables (double storage) | 1e-3 | 0.902 | 0.737 | 0.834 | 0.056 |
| dual, one shared table | 1e-2 | 0.968 | 0.757 | 0.884 | 0.188 |

Also seen: `memory_rule_dual.py shared_only` at 1e-3 reproduced the earlier typed single-path result on seed 4100 exactly, so the two scripts agree. The single path at 1e-2 has **not** been run on any seed. That is the control below, and I do not know its result.

The two-table arm bought nothing on the development seed and costs about four times the run time on this loaded machine. It is dropped.

## Design

`memory_rule_dual.py`. World, memory, step budget and endpoints as in `PROTOCOL-MEMRULE-SHARING.md`; Adam reset each phase. Confirmatory seeds 4101–4105, the same seeds as the earlier test, so its typed M = 0 and M = 8 results (learning rate 1e-3) are reused as references and comparisons are paired by seed.

New arms, 15 runs:
- `dual_pool` at 1e-3
- `dual_pool` at 1e-2
- `shared_only` at 1e-2, the control that separates the learning rate from the second path

## Predictions

"Holds" means on at least four of the five seeds.

- E1. `dual_pool` at 1e-3 keeps inference at least 0.9 of typed M = 0's, and raises acquisition by at least 0.03 over it.
- E2. `dual_pool` at 1e-2 reaches acquisition at least 0.95, with inference at least 0.9 of typed M = 0's at 1e-3.
- E3, the decisive comparison, no direction predicted: is `dual_pool` at 1e-2 at least 0.03 above `shared_only` at 1e-2 on acquisition, with inference no more than 0.02 below it? If `shared_only` at 1e-2 does as well, the gain on the development seed came from the learning rate, not the second path.
- E4. Corrections stay low for `dual_pool` at 1e-2: at most 0.30. A correction must outvote values that same-type neighbours have pushed into the shared slots over many phases, within one phase's 200 steps. A pass is consistent with that account, not proof of it.

## Limits, stated in advance

- The type information in the shared addresses is given, not learned.
- Thresholds were set after one development seed, which showed a pattern in E1, E2 and E4's direction.
- The private path is per-fact storage. With one shared table its slots collide with the shared path's; the storage is the same as a single-path arm, but this is close to adding a hashed exact store. Whether it is "learning" or "storing" is not settled by this test.
- A value table, frozen addresses and a synthetic world. Not evidence about a language model.
- Result files are created exclusively under `results/memory_rule_dual/`.
