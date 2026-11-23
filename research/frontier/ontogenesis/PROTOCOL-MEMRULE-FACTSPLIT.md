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

## Addendum, recorded before any confirmatory result was read (2026-11-23, about 05:20 UTC)

This protocol was committed by a second, duplicate copy of the session (after a resume), which then stood down. The copy that runs the measurement adds three facts:

- **The development seed was seen.** Contrary to the Limits above, the running copy measured the split on development seed 4100 just before this protocol was committed. End-of-run retention (regular / exception / corrected): `drop` 0.902 / 0.038 / 0.042; `transport` 0.904 / 0.047 / 0.042; full re-index, 32 clusters, 0.998 / 0.631 / 0.636; given types without re-index 0.927 / 0.023 / 0.043. The predictions F1–F3 above were written by the copy that had not seen these numbers. They are kept unchanged.
- **Timing.** The confirmatory runner (`run_factsplit.sh`) was started at 05:10:15 UTC, seven seconds before the protocol commit (05:10:22 UTC). No result file had been opened or printed when this addendum was written; one had been written to disk.
- **One extra arm, descriptive only.** The runner also reruns given types without re-index (`memory_rule_recluster.py ... typed`, with `every` set beyond the run length). It is reported for reference and has no prediction.
