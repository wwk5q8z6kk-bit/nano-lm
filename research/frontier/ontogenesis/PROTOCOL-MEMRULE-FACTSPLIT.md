# Protocol: what dropping moved values costs individual facts

Written 2026-11-23, before any run of this measurement. `RESULTS-MEMRULE-TRANSPORT.md` (`ab782b6`) left one cost unmeasured: when a re-cluster moves an entity and its values are dropped rather than replayed or transported, what happens to its individual facts? Exceptions and corrected facts cannot be recovered from the group's rule, so they should suffer most.

## Design

A measurement only. `memory_rule_factsplit.py` scores, at the end of a run, retention of every shown fact split three ways:
- regular: followed its type's rule at creation and was never corrected;
- exception: broke the rule at creation and was never corrected;
- corrected: corrected at least once, scored against its latest value.

The scoring call is added to `memory_rule_transport.py` and `memory_rule_recluster.py`. Nothing else in those scripts changes, so the registered `late` results must reproduce exactly. That is checked for every run, and any mismatch is reported as a failure of the rerun, not used.

Arms, the same held-out seeds 4101–4105 (15 reruns): `drop`, `transport` (`memory_rule_transport.py`), and full re-index with 32 clusters (`memory_rule_recluster.py`).

## Predictions

"Holds" means on at least four of the five seeds.

- F1. Exceptions: `drop` keeps at least 0.20 fewer than the full re-index.
- F2. Regular facts: `drop` is within 0.10 of the full re-index. The group's rule answers for them.
- F3. Corrected facts: `drop` keeps no more than the full re-index.
- Recorded without a prediction: `transport` against `drop` on each kind.

## Limits

- Post-hoc in the sense that the seeds were already used for `RESULTS-MEMRULE-TRANSPORT.md`. The new quantity has not been looked at on any seed, including the development seed.
- End-of-run snapshot only, not averaged over late phases.
- Synthetic world, value table. Not evidence about a language model.
- Results under `results/memory_rule_factsplit/`, created exclusively.
