# Fresh development validation after question/scope repairs

**Outcome: failed transfer.** The bounded software repairs pass their regression
tests and full verification suite, but neither model mode produces a usable
asserted field on these 24 fresh encounters. Both abstain on all 24 notes.
Checking discards correct raw assertions. Higher aggregate checked scores are
uncertainty-state matches, not improved usable scribing.

This is synthetic development evidence, not H6/H7, a confirmatory evaluation,
clinical validation, or a representative estimate. The previous development
cohort and every frozen artifact remain unchanged. This cohort differs from
the previous one; their results are not a paired before/after treatment effect.

## Separation, annotation and admission

One new-context AI author created 24 encounters, 284 turns, in six strata:
ordinary question variation, field-local uncertainty, irrelevant/unattributed
context, medication/allergy scope, correction/conflict, and mixed/ambiguous
questions. Each stratum has four cases. The author did not inspect implementation,
previous cases, annotations or inference outputs.

Two separate new-context AI agents annotated all 120 fields from source and
rubric only. A third agent adjudicated before inference, blind to implementation
and model answers. These are independent contexts, not independent model
families, human raters or clinician dual-IAA. The coordinator fixed the code
using separate injected development cases and independent patch review, then
recorded 170 implementation hashes before inspecting this cohort.

Original annotators agreed on state for 119/120 fields (99.17%), normalized
first-span value for 72/80 fields both called supported (90%), and complete
evidence sets for 100/120 fields (83.33%). Adjudication resolves 20 state/evidence
differences. It also changes eight shared A/A labels for conventional
no-known-allergies wording to U under the rubric's qualified-knowledge rule.
These overrides are **not** independent annotation agreements. Original files,
all decisions and metric definitions are retained in [adjudication.md](adjudication.md)
and [agreement.json](agreement.json). Boundary conventions are not fully uniform;
lexical exact mismatches must not be described as adjudicated clinical errors.

Primary gold has 80 S, 12 A, 27 U, one C and zero M fields. All 24 cases contain
usable gold assertions. All 120 fields remain in every primary denominator.
The primary annotation-dictionary oracle scores 120/120 and the all-M control
scores 0/120. This verifies scoring wiring, not end-to-end representability.
Every one of the 12 gold A fields fails the unchanged legacy denial transport
contract, across 11 cases; only 13 entire cases pass secondary transport.
Natural-language denials remain A in the primary labels. There is no missing-state
specificity coverage, and conflict recall is based on just one field.

[protocol.md](protocol.md) and [annotation_lock.json](annotation_lock.json)
record the pre-inference procedure and input hashes. Sources, original labels,
process notes, adjudication, gold, rubric and protocol were locked before either
run. No implementation or label tuning followed inference.

## Execution and primary raw-versus-checked results

Cached base Qwen2.5-1.5B-Instruct revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306` ran offline on local Apple MPS, greedy,
with a 256-token bound (per slot for constrained generation). The original
few-shot free prompt and structured constrained prompt are saved separately.
No response hit the generation token limit. Paid compute: **USD 0**. Free mode
took 38.94 seconds and constrained mode 53.01 seconds, including loading and
scoring. Python 3.14.7, PyTorch 2.11.0, Transformers 5.5.3.

Each mode invokes the predictor once per source; constrained prediction uses
slot-by-slot generation. The same captured response supplies raw,
checked and final-note measurements. Raw constrained output already incorporates
decoder rules; it is not an unaided model baseline. Raw primary scoring uses
the unchanged strict whole-record parser, so a malformed response fails all five
raw fields even if local recovery can parse some of them.

| Measure | Free raw | Free checked | Constrained raw | Constrained checked |
|---|---:|---:|---:|---:|
| Correct state /120 | 7 | 26 | 16 | 26 |
| Correct normalized content /120 | 3 | 26 | 9 | 26 |
| Grounded exact field /120 | 2 | 26 | 9 | 26 |
| Complete evidence exact field /120 | 2 | 8 | 7 | 12 |
| Emitted extraction assertions | 10 | 0 | 11 | 0 |
| Exact-correct assertions /92 gold assertions | 2 | 0 | 3 | 0 |
| Asserted state/content mismatches | 7 | 0 | 8 | 0 |
| Recalled conflict /1 | 0 | 0 | 0 | 0 |
| Whole-record parse/inference failures /24 | 22 | 1 | 0 | 0 |

Grounded-exact scoring requires correct content and exact evidence for S/A/C;
U/M receive credit from state alone. All 26 checked grounded-exact credits in
each mode are U. Complete-evidence scoring additionally tests all uncertainty
evidence. Free checked outputs contain 115 U fields and one failed encounter;
constrained checked outputs contain 106 U and 14 M. **Final notes emit zero
assertions in both modes and abstain on 24/24 usable sources.** Zero asserted
mismatches is a consequence of zero assertions, not evidence of useful safety.
By the metric definition, a constant-U output would score 27/120 grounded-exact,
above either checked result. This is an arithmetic counterfactual, not another
inference run; the all-M control alone does not expose that degenerate baseline.

| Primary paired change | Free | Constrained |
|---|---:|---:|
| Exact-correct assertions retained | 0 | 0 |
| Exact-correct assertions lost | 2 | 3 |
| Exact-correct assertions salvaged | 0 | 0 |
| Normalized-correct assertions lost | 3 | 3 |
| Asserted content mismatches removed | 7 | 8 |
| Asserted content mismatches introduced | 0 | 0 |

The apparent exact-field salvage totals (26 free, 20 constrained) mix state
changes and parse recovery; none is a usable assertion. Free recovery avoids
21 whole-record failures and admits 37 individual slots from those responses,
but checking rejects every admitted free slot. Recovery count means syntactic
admission, not semantic correctness.

## Remaining failure mechanisms

The old global-unsafe flag fires on **zero of 24** sources when replaying the
locked source policy. Whole-note abstention now follows local rejection across
all fields. The grammar recognizes 44 of 142 clinician turns as singleton
requests. This is descriptive implementation coverage, not an attribution
accuracy metric: some turns are statements or compound requests.

An independent output audit found that 76/120 constrained slots had only the
two fallback choices M and U[]. They include 56 gold S, six A and the sole C.
Correct concrete output was therefore unavailable for at least 63 fields before
model selection. Candidate/recognizer coverage is part of the failure, alongside
model selection, source quoting and checker rejection.

For example, free scope-08 duration `two days` and scope-15 severity `moderate`
are exact-correct raw fields discarded for binder state mismatch. Constrained
scope-07 complaint, scope-09 severity and scope-16 complaint suffer the same
loss. Free outputs also misclassify positive acetaminophen as absent, promote
unattributed notebook text, retain a sister's corrected allergy, and omit part
of a multi-complaint report. Exact-source formatting alone cannot repair these.

A separate **post-hoc** diagnostic scores the locally parse-admitted raw slots
already recorded in the unchanged adapter trace. It neither repairs text nor
replaces the strict primary baseline. Free mode has 47 admitted raw slots,
including **16 exact-correct assertions and 19 normalized-correct assertions;
checking loses all of them**. Constrained mode loses its three exact-correct
assertions. This exposes additional over-abstention hidden by whole-record raw
failure accounting. See [posthoc_diagnostics.json](posthoc_diagnostics.json)
and its [reproduction script](posthoc_diagnostics.py).

## Prespecified annotation sensitivity

The same saved responses were rescored under each original annotation set and
under a variant restoring only the eight shared no-known-allergies A labels.
No inference or implementation change was made.

| Annotation policy | Free raw / checked grounded-exact /120 | Constrained raw / checked grounded-exact /120 | Free / constrained checked complete-evidence /120 |
|---|---:|---:|---:|
| Adjudicated primary | 2 / 26 | 9 / 26 | 8 / 12 |
| Original A | 2 / 17 | 5 / 17 | 4 / 8 |
| Original B | 2 / 18 | 6 / 18 | 4 / 7 |
| Restore only eight shared A labels | 2 / 18 | 6 / 18 | 4 / 7 |

All variants still have zero checked assertions and 24/24 note abstentions in
both modes. The label ambiguity materially changes aggregate scores, but not
the failed-utility conclusion. See [annotation_sensitivity.json](annotation_sensitivity.json)
and [analyze_saved.py](analyze_saved.py).

## Breakdowns

Grounded-exact counts below use primary gold. Checked successes are all U;
they must not be read as recovered facts.

| Field (24 each) | Free raw / checked | Constrained raw / checked |
|---|---:|---:|
| Chief complaint | 0 / 0 | 2 / 0 |
| Duration | 1 / 3 | 0 / 3 |
| Severity | 1 / 6 | 1 / 5 |
| Medication | 0 / 2 | 0 / 2 |
| Allergy | 0 / 15 | 6 / 16 |

| Stratum (20 fields each) | Free raw / checked | Constrained raw / checked |
|---|---:|---:|
| Ordinary question variation | 0 / 3 | 0 / 2 |
| Field-local uncertainty | 1 / 6 | 3 / 6 |
| Irrelevant and unattributed context | 0 / 2 | 2 / 2 |
| Medication/allergy scope | 1 / 3 | 1 / 3 |
| Correction and conflict | 0 / 3 | 1 / 4 |
| Mixed or ambiguous questions | 0 / 9 | 2 / 9 |

## Verification, preservation and next work

The [repair report](../p1_scope_repair_dev_20261023/REPORT.md) records question
attribution, local quarantine, strict field recovery and independent code review.
Separate injected regressions verify valid-field retention within those bounded
patterns. The full suite passes **1,843 tests, with two environment skips**;
direct fabric, trajectory and synthetic scribing benchmark checks also pass.
Passing these tests does not establish natural-dialogue model quality.

All **1,099 protected file hashes** remain unchanged, including frozen evidence
and the entire earlier independent development cohort. Both runs use exactly
the locked 170 source/test/harness hashes. Input locks and snapshots are verified
before and after inference. Final preservation and documentation gate records
are in the adjacent repair artifact directory. No frozen experiment was rerun,
no freeze tag was created or moved, and no paid compute, training or private data
was used.

Saved evidence: [free comparison](free-01/comparison.json),
[constrained comparison](constrained-01/comparison.json), per-field pairs,
encounters, diagnostics, generation records, prompts, input/source snapshots and
provenance in those run directories. The independent reviewer records its
assessment in [independent_result_review.md](independent_result_review.md).

The general valid-field retention problem remains open. Further development
must address question/answer interpretation, complete reply segmentation and
concrete candidate coverage, including denials and conflicts, using separate
new cases and another fresh independently annotated cohort. These exposed
results remain immutable diagnostics; they are not a P1 quality pass.
