# Corrections to the G04 studies after code review (2026-11-23)

Cursor Bugbot reviewed PR #204 and raised seven findings; one more was raised on PR #197. Each was checked against the code. Six were real bugs, one is a false positive that exposed a real reproducibility gap, and one is a counting inconsistency. The corrected code is version 2 of `threeway.py` and `rule_change_detect.py`. Every affected study is rerun with it by `run_g04_v2.py` into `results_v2/`. The original result files in `results/` are kept unchanged, and the original code is in the git history.

| Finding | Verdict | Fix | Studies affected |
|---|---|---|---|
| A (type, relation) pair can change twice, so the earlier change was scored against a stale value | **Real.** Repeats occur in 9 registered seeds: 7201, 7203, 7302, 7403, 7601–7604 | Each change is scored only until the next change of the same pair. Old-entity revision is scored only for the change still in force. A flag counts as true only for the change in force when it fires | 2, 3, 4, 6 |
| Missed changes keyed on (type, relation), dropping the change phase | **Real** | Keyed on (phase, type, relation) | 3–6 |
| Z5 compared exceptions of all ages against `exception_kept` (10+ phases old) | **Real** | Same age restriction for both | 3 |
| Narrow revisions OR'd the slot masks of all pairs revised in one phase, and keyed the router's mask by member type rather than the flagged group | **Real** | Each revised pair is trained separately under its own mask, keyed by the group's majority cluster (or type) | 5, 6 |
| Study 2 used override-store answers (stored exceptions) as decoder targets | **Real** | Targets on the other relations come from the model alone | 2 |
| Study 2 reused predictions after a mid-phase revision | **Real** | Predictions are refreshed after each accepted candidate | 2 |
| The cost-study arms "never set" their table size or switches | **False positive.** The arms were run with explicit settings by an inline runner, and the result files confirm it (for example, `oracle_x16` has a 0.002 leakage gap) | The runner is now committed: `run_g04_v2.py` records every arm's settings | — |
| PR #197: the bundle block says 113 local branches, the totals say 112 | **Counting inconsistency.** 113 refs existed at bundling; 112 were counted later in the same session | Wording corrected in the manifest | G00 only |

Study 1 (`RULE-CHANGE-20261122.md`) is not affected: none of its seeds repeats a pair, and none of the other findings touches its code.

## Corrected results

Added below as each rerun finishes, with old and new figures side by side.
