# Compact model → checked scribing: development pilot

The revised 20-dialogue pilot produces **14 useful review drafts, six abstaining
drafts and zero final-path failures**, evenly split across canonical and
paraphrased questions. The checked extraction scores **95/100 grounded exact
fields**, with zero false assertions among its 78 presented fields. The final
note check is stricter: it presents 56 supported/absent claims and quarantines
all assertions in six drafts. This is an exposed, bounded development result,
**not broad scribing readiness or clinical validation**.

The model is cached **base Qwen2.5-1.5B-Instruct**, revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, not the historical fine-tuned adapter
and not Native Nano. No new training or cloud resource was launched. Incremental
paid compute is **$0**; local CPU inference used PyTorch 2.11.0 / Transformers
5.5.3. Checkpoint/tokenizer hashes, prompts, raw generations, stage timings,
contract outputs, encounters and drafts are retained in each completed run.
Historical provider balances were not treated as current.

## Instrument admission

[dialogues.json](dialogues.json) contains 20 invented, exposed development
examples: ten worlds paired across canonical/paraphrased Doctor questions, with
identical Patient replies and targets. [spec.json](spec.json) independently sets
the 25-cell quota table: 66 supported, 14 absent, 12 missing, four uncertain and
four conflicting fields. Source/target binding, all quotas, unique sources/IDs,
world pairing and development split are checked before inference.

The test-only oracle scores 100% grounded exact fields; the all-missing negative
control scores 12%. Neither enters scribing. The legacy deterministic reference
scores 56%: all 50 canonical fields plus six missing paraphrase fields.
[annotation_review.json](annotation_review.json) is a separate logical review by
the **same coding agent**, not independent human/clinician review or IAA. It
replaced an ambiguous concurrent-medications label with an explicit unresolved
positive-versus-denial conflict. No pilot case can later be a confirmatory test.

## Changes tested

The explicit opt-in `constrained` decoder generates one field at a time from a
finite token trie. Evidence choices are exact, uniquely located Patient text
under a recognized preceding question. Duration and severity choices respect the
bounded field types; conflict choices require two distinct Patient turns. State
selection remains learned; proposals are never replaced with gold or silently
repaired after generation. M, U, A, S and C remain available where applicable.

The development `context` policy recognizes a bounded vocabulary of question
paraphrases. Unknown, compound, historical, future and third-party contexts
abstain. The existing bounded reply grammar and conflict resolution bind raw
proposals; wrong states become uncertain. Complete-observation and note/schema
checks then run before a review draft can assert content. Duration/severity
uncertainty uses narrow, field-specific full-reply forms so a longer reply cannot
hide a correction to another field.

This is a **hybrid learned/grammar-constrained path**. The decoder, binder and
note checker share the question classifier; their context handling is not an
independent semantic validation. Reply templates and the small vocabulary remain
strong limitations. Legacy defaults remain available.

The frozen `nano_ai/adapters/state_span.py` is byte-identical to the prior
committed source. The new development adapter lives in
`nanoscribe/context_solver.py`; historical H3/H4/H5/H6 source-pin tests pass.
Run 09 generated against an intermediate equivalent adapter seam, whose exact
snapshot is retained. [current_path_audit.json](current_path_audit.json) records
20/20 identical full drafts when those saved proposals are replayed through the
final separate adapter, with source/code hashes and **no new model inference**.
The additional challenge runs actual model inference through that final path.

## Observed pilot results

Failures count as wrong on all five fields; denominators never exclude them.
Raw is strict proposal/contract evaluation without binding. Checked is the
adapter output **before** the stricter final note check.

| Run / stage | Failures / 20 | State correct / 100 | Grounded exact / 100 |
|---|---:|---:|---:|
| 02: rules, raw | 20 | 0 | 0 |
| 02: rules, checked | 20 | 0 | 0 |
| 03: worked examples, raw | 15 | 24 | 24 |
| 03: worked examples, checked | 14 | 19 | 19 |
| 08: constrained/context, raw | 4 | 75 | 68 |
| 08: constrained/context, checked | 0 | 81 | 81 |
| 09: typed constrained/context, raw | 2 | 85 | 85 |
| 09: same raw generations, legacy binder | 0 | 54 | 54 |
| 09: typed constrained/context, checked | 0 | 95 | 95 |

Run 09 is syntactically parseable on 20/20 cases, but only **18/20 raw proposals
are contract-valid**. The two failures incorrectly label positive loratadine
reports as absence. The binder quarantines these; it does not credit them as
correct raw outputs. Raw proposal errors also include uncertainty labeled missing
and a medication conflict reduced to a denial. The checked extraction retains
two duration conflicts, quarantines the two medication conflicts, preserves all
four uncertain states, and misses one missing-state label. Its five state errors
are explicit in the confusion tables. No raw failure is removed from scoring.

| Final path | Run 03 canonical / 10 | Run 03 paraphrase / 10 | Run 09 canonical / 10 | Run 09 paraphrase / 10 |
|---|---:|---:|---:|---:|
| Useful checked draft, human review required | 2 | 0 | 7 | 7 |
| Abstaining draft | 2 | 2 | 3 | 3 |
| Failed inference, no note | 6 | 8 | 0 | 0 |

The final note check rejects all assertions in the uncertainty-as-missing,
positive-as-absent and unresolved medication-conflict worlds. It presents 46
supported values and ten explicit denials across the other 14 drafts, retaining
missing, uncertainty and duration-conflict review items. Thus 95% checked-field
accuracy is **not** 95% note completeness or clinical reliability.

Rules-only generation took 76.758 seconds; worked examples 131.215 seconds;
run 09 took 316.130 seconds, including loading and pipeline work after provenance
hashing. Concurrent local tests affected throughput: these are observations, not
speed benchmarks or evidence that CPU is always the fastest backend.

## Attempt preservation and additional challenge

`run-01` preserves the initial annotation attempt. `run-04` through `run-07` are
interrupted development attempts with STATUS notes and exact snapshots. They
exposed a legacy-question guard issue, Qwen's combined `:S` token boundary,
cross-field source menus and uncertainty-check scope. They are not complete
cohorts or discarded successful runs. Run 08 exposed duration-prefix/severity
punctuation issues and single-turn conflict choices; run 09 adds typed choices
and separate-turn conflict selection. All revisions respond to exposed cases;
this is not a preregistered treatment comparison.

Six additional invented cases in [challenge-01/cases.json](challenge-01/cases.json)
exercise new question wording and values, explicit uncertainty/denial,
medication conflict, third-party subject, future plan and compound questions.
Expected outcomes are stored before inference and never passed to the model.
Actual inference through the final adapter passed **5/6** in 43.874 seconds.
New wording/values and uncertainty/denial produced useful drafts. Third-party,
future and compound questions abstained. The medication-conflict case selected
only the denial; the binder quarantined it and the final note abstained from all
assertions, losing an otherwise valid chief complaint. This fails the expected
conflict-retention/usefulness outcome, with no unsafe assertion presented. The
failure is preserved in [challenge-01/results.json](challenge-01/results.json),
with snapshots and [provenance](challenge-01/provenance.json). These cases are
exposed development checks, not held-out clinical validation.

## Decision and next work

Keep the constrained/context route as an explicit development option. Structured
syntax and bounded paraphrase transfer now work end to end; general semantic
state selection, medication-conflict retention, longer free-form dialogues and
independent annotation remain open. Additional GPU time alone does not solve
these semantic limits. Any training must use a separate development corpus and
retain these exposed cases as regression checks, with a new independently
reviewed instrument before confirmatory use.

H6/H7 artifacts, terminal decisions, consumed evaluations and freeze tags remain
unchanged. Results are not pooled with historical compact, OpenResearch, C2, H6
or invalid-gold H7 scores.

## Reproduce

```sh
python3 scripts/run_p1_pilot.py
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 python3 scripts/run_p1_pilot.py \
  --checkpoint /absolute/path/to/pinned/local/snapshot \
  --decoding constrained --binder context --output /new/development/result/directory
python3 -m nanoscribe.scribe --transcript /absolute/path/to/transcript.txt \
  --compact-checkpoint /absolute/path/to/pinned/local/snapshot \
  --compact-decoding constrained --context-policy context --format text
python3 artifacts/p1_pilot_v1/audit_current_path.py
python3 -m pytest nanoscribe/test_context.py nanoscribe/test_constrained.py \
  nanoscribe/test_compact.py scripts/test_run_p1_pilot.py -q
```

Model loading is lazy/offline, with no remote model code, and requires an existing
local checkpoint. No targets, fixture specs or gold evidence enter inference.
