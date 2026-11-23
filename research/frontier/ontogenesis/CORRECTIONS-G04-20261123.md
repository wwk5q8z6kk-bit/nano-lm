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

### Prediction verdicts, original → corrected (same registered predictions, same seeds)

| Study | Changed verdicts | Unchanged |
|---|---|---|
| 2: bilinear detector (Y1–Y5) | none | Y1–Y4 held, Y5 failed |
| 3: three-way router (Z1–Z8) | none | Z1–Z5, Z8 held; Z6, Z7 failed |
| 4: cost decomposition (C1–C5) | none | C1–C4 held, C5 failed |
| 5: narrow revisions (N1–N5) | **N4 held → failed (1 of 5 seeds); N5 held → failed (0 of 5 seeds)** | N1–N3 held |
| 6: learned clusters (E1–E5) | none (E2 now 3 of 5 seeds, E3 now 4 of 5) | E2–E4 held; E1, E5 failed |

### Headline figures, original → corrected (five-seed means; three seeds for study 2)

| Study, arm | Old entities revised | Stable rule | Exceptions kept | Noise resisted | False alarms / missed |
|---|---|---|---|---|---|
| 2: detector (bilinear) | 86.9% → **92.8%** | 94.5% → 95.5% | 100% → 100% | 97.1% → 97.7% | 47.7 / 2.3 → **23.3 / 1.7** |
| 2: seen twice | 58.2% → 60.8% | 95.9% → 95.9% | 100% | 97.7% | — |
| 2: ceiling | 92.9% → 97.1% | 99.4% | 100% | 99.4% | — |
| 3: three-way | 85.8% → 86.6% | 72.6% → 72.6% | 89.1% | 91.1% | 27.8 / 1.4 → same |
| 3: private first | 42.6% → 41.9% | 75.3% | 91.5% | 90.8% | 15.2 / 6.2 → 15.2 / 6.4 |
| 4: oracle and router | changes ≤ 0.6 points | unchanged | unchanged | unchanged | unchanged |
| 5: oracle, share ≥ 0.2 | 92.5% → 92.0% | 76.7% → 76.8% | 99.9% | 95.4% → 95.1% | — |
| 5: three-way, share ≥ 0.2 | 79.0% → 78.5% | **77.4% → 74.1%** | **95.4% → 89.9%** | 94.5% → 92.5% | **15.8 → 32.8** / 2.4 → 2.0 |
| 5: three-way, share ≥ 0.2 + ×16 | 76.7% → 81.3% | **80.3% → 74.6%** | **94.1% → 85.8%** | 94.4% → 90.8% | **19.6 → 57.8** / 2.8 → 1.6 |
| 6: given types + detector | **70.0% → 81.5%** | 77.0% → 72.4% | 94.8% → 88.9% | 93.8% → 91.8% | 14.2 → 27.2 / 1.2 → 1.6 |
| 6: EM clusters + detector | 50.4% → 52.8% | 70.7% → 67.5% | 97.0% → 93.5% | 87.8% → 84.9% | 14.4 → 20.8 / 5.2 → 6.2 |

## What the corrections change

1. **Withdrawn:** "both fixes bring the three-way router's stable-rule cost under 3 points" (`THREEWAY-NARROW-20261122.md`, N5).
   - With correct per-pair masks, the router with narrow revisions raises about twice as many false alarms (33–58 per run).
   - It keeps fewer exceptions (86–90%), and its stable rule is 74–75%, not 80%.
   - The original result depended on the bug: the OR'd mask made each revision wider, and so stronger.
   - A likely mechanism, not tested: weaker, properly narrow revisions leave a falsely flagged group still disagreeing, so the detector fires on it again.
2. **Stands:** narrowing helps when revisions are triggered correctly. The oracle arms are practically unchanged (N1–N3): 9 points of stable rule recovered for about 7 points of revision. The open problem is now the detector's repeated firing under narrow revisions, not the revision itself.
3. **Revised upward:**
   - The bilinear change detector (study 2) revises 93% of old entities with half the false alarms. Using stored exceptions as targets, and stale predictions, had been hurting it.
   - With given types (study 6), the router revises 82%, not 70%; the repeated-pair scoring had depressed it.
4. **The learned-cluster gap is larger than reported:** 82% (given types) against 53% (EM clusters), a 29-point gap rather than 20.
5. **Studies 3 and 4 stand as reported.**

## Follow-up diagnostic: the suggested mechanism for the withdrawn result is wrong (post hoc)

Above I suggested that properly narrow revisions make the detector fire repeatedly on falsely flagged groups. On the narrow study's registered seeds, with the corrected code (rerun reproduces `results_v2` exactly; raw: `results_v2/threeway_narrow/diagnostic_repeat_flags.json`):

| Router | False alarms | Distinct falsely flagged groups | Repeat flags on an already-flagged false group | True flags | Repeat true flags |
|---|---|---|---|---|---|
| normal revision | 34.6 | 20.8 | 13.8 | 20.2 | 10.2 |
| narrow revision (share ≥ 0.2) | 32.8 | 20.2 | 12.6 | 20.6 | 10.6 |

- **Narrowing does not raise false alarms** relative to normal revision; the counts are equal. The apparent doubling was relative to the buggy original, whose OR'd masks made each revision wider and suppressed re-firing. My suggested mechanism is therefore not supported.
- **Re-firing is general.** About 40% of false alarms and about half of true flags repeat a group already flagged, under either revision. Those repeats add revision volume without new information.
- **What remains of N4:** narrowing lifts the router's shared-only stable rule by only 1.6 points (0.752 → 0.768), against the 4 points predicted.
