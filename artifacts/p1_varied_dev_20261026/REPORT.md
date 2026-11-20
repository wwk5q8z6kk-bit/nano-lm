# Varied medication-conflict development run — 2026-10-26

The bounded context path now recognizes complete blanket medication denials
(`No medication.`, `No meds!`, `I deny taking medicine.`), common direct
questions, short duration replies and `I take` / `I am taking` replies. A use
observation plus a blanket denial is retained as a conflict in either order,
including when the model selects only one side. Qualified/named/other-person
denials remain uncertain. Source-derived repairs are not credited as raw model
success. Frozen adapters, screens, checkpoints and recorded results were not
modified.

## Paired model comparison

24 invented, exposed development dialogues, six strata with four cases each;
120 fields. The implementing agent authored and labelled the cases before
inference. This is neither blind evaluation nor clinical/human annotation.
See [protocol](protocol.md), [annotation lock](annotation_lock.json),
[sources](sources.json) and [gold](gold.json). The labels contain 44 supported,
4 absent, 4 conflicting, 12 uncertain and 56 missing fields.

Pinned base Qwen2.5-1.5B-Instruct revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, greedy offline inference on local Apple
MPS. The same saved generation supplies each run's raw and checked scores.
Both inference runs used the fixed source snapshot preserved in
[constrained-02/source_snapshot](constrained-02/source_snapshot); the later shared
checkout changes were excluded. Run status is complete and source/input hashes
passed in both provenance records. Incremental paid compute: **$0**; no cloud
resources started or campaign allocation consumed. Inference plus setup took
56.90 seconds constrained and 38.31 seconds free.

| Measure | Constrained | Free generation |
|---|---:|---:|
| Raw exact grounded fields | 102/120 | 10/120 |
| Checked exact grounded fields | 117/120 | 39/120 |
| Raw parser failure cases | 0/24 | 22/24 |
| Raw contract transport failures | 0/24 | 0/24 parsed outputs |
| Review drafts | 24/24 | 13/24 |
| Abstentions (including failed inference) | 0/24 | 11/24 |
| Failed inference cases | 0/24 | 7/24 |
| Checked medication conflict recall | 4/4 | 2/4 |
| Correct assertions emitted in notes | 48/48 | 26/48 |
| Emitted assertion annotation mismatches | 0 | 0 |

All-missing and all-uncertain controls score 56/120 and 12/120 respectively.
The constrained raw model recalls **0/4** conflicts and makes eight asserted
field mismatches. The checked source-preservation policy removes those eight
mismatches and salvages 15 fields, losing no correct assertions. The three
remaining checked errors are allergy `uncertain` where the labelled source is
`missing` (varied-06, varied-10, varied-16). No assertion is made for those fields.

These results demonstrate the exercised development path only. They do not
supersede the earlier independent development cohort's failed transfer result.
The recognizer, binder, candidate menus and checker share bounded lexical rules;
there is no independent semantic verifier. Free generation remains unreliable.
Use the explicit development path:

```sh
python3 -m nanoscribe.scribe --transcript /absolute/path/to/transcript.txt \
  --compact-checkpoint /absolute/path/to/pinned/local/snapshot \
  --context-policy context --compact-decoding constrained --format text
```

[Analysis](analysis.json), [recompute script](analyze_saved.py),
[constrained comparison](constrained-02/comparison.json),
[free comparison](free-01/comparison.json), and per-run encounters retain all
outputs, errors, raw decisions, checked fields and final notes. Keep parser
failures in all field denominators.

## Integrity and concurrent work

The initial CPU attempt, `constrained-01`, generated 24 encounters but failed
its final source-integrity check because another active chat modified the shared
checker and question rules during generation. That attempt remains preserved
and is excluded from the table. The two completed runs used an isolated copy of
its run-start source snapshot under `/tmp/p1-varied-locked-20261026`.

The shared checkout subsequently gained separate clause-retention work. A
minimal merge retained both sets of duration templates, removing a duplicate
map key that had shadowed `{} ago.`. [Shared replay](shared-replay-01/comparison.json)
replayed the 24 constrained saved generations through a hash-pinned snapshot of
the combined code: 117/120 checked fields, 24 review drafts and all four conflicts
retained. This is an integration replay, not another model run or validation set.
[Replay script](replay_shared.py) and provenance preserve that distinction.

## Software verification and preservation

- Regression RED: 13 failures / 4 passes before this repair (`red.log`).
- Fixed inference snapshot: 220 focused tests passed (`locked-tests.log`).
- Before concurrent edits: 513 NanoScribe/runner/comparison tests passed (`tests.log`).
- Core suite: 1,347 passed, 2 skipped (`core-tests.log`).
- Combined checkout: all 17 new regressions passed (`merged-regressions.log`).
- Intermediate combined suite: 556 passed, 3 failures (`merged-tests.log`). Those
  failures concern earlier expectations for unknown/mixed questions under the
  concurrently introduced clause-retention behavior; the combined full suite
  was not green at that check. A later shared-suite check had 577 passes and
  two failures in the concurrently added clause-retention cases
  (`final-shared-tests.log`); the 17 medication/question/duration regressions
  continued to pass. This report does not claim integration completion.
- Active-state and documentation integrity checks passed. Direct fabric 8/8,
  C3 7/7 and NanoScribe software benchmark passed earlier in this turn.
- SHA-256 comparison: all 699 tracked protected files unchanged; see
  [preservation record](preservation.json) and before/after manifests.

Next work: finish the concurrent integration checks, then evaluate the selected
constrained path on separately authored, blinded, varied dialogues. Keep raw
model accuracy separate from deterministic repairs, retain missing/uncertain
errors, and preserve all previous evidence. No training or paid runtime is
needed to resolve the remaining local integration issues.
