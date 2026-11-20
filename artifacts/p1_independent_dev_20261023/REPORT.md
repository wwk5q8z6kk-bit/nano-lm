# Fresh P1 development validation — 2026-10-23

The repaired scribing path did not transfer successfully to these fresh cases.
Both decoding modes abstained on **24/24 encounters**, although every encounter
had usable annotated content. Free generation failed the exact-source parser
on every response. Constrained generation produced parseable responses, but
checking left all 120 fields uncertain (98 changed; 22 retained) and discarded three raw
assertions with correct normalized content. This is a failed development
transfer result, not evidence of P1 readiness or clinical safety.

The source author and two annotators worked in separate, blinded agent contexts.
Annotations and adjudication were locked before inference. These are AI-agent
annotations, not independent human or clinician evaluation. No historical
evidence, H6/H7 evaluation, freeze tag, or model weight was changed.

## Primary results

All 24 encounters and all 120 fields remain in every relevant denominator.
The primary scorer counts every field of an unparseable response as failed.
Its free-generation score therefore measures admissible extraction, not zero
semantic understanding. Raw and checked results within each mode come from
the **same generation**, without retries.

| Measure | Free raw | Free checked | Constrained raw | Constrained checked |
|---|---:|---:|---:|---:|
| Correct state / 120 | 0 | 0 | 13 (10.83%) | 11 (9.17%) |
| Correct normalized content / 120 | 0 | 0 | 13 (10.83%) | 11 (9.17%) |
| Grounded exact fields / 120 | 0 | 0 | 10 (8.33%) | 11 (9.17%) |
| Complete evidence exact fields / 120 | 0 | 0 | 4 (3.33%) | 0 |
| Parse-admissible supported/absent assertions | 0 | 0 | 3 | 0 |

Normalized content requires the correct state and, for supported values, the
normalized first annotated value. Grounded exact scoring additionally requires
an annotated exact evidence span for supported/absent fields and every
annotated contradictory span for conflicts; missing/uncertain fields need only
the correct state. Complete evidence scoring requires the entire evidence set,
including uncertainty evidence. Thus **11/120 after constrained checking is
exactly the prevalence of gold uncertainty**, not successful content recovery.

| End-to-end measure | Free | Constrained |
|---|---:|---:|
| Raw parse failures / 24 | 24 | 0 |
| Checked inference failures / 24 | 24 | 0 |
| Whole-note abstentions despite gold usable content / 24 | 24 | 24 |
| Correct supported/absent claims actually emitted / 104 | 0 | 0 |
| Conflict state recall / 1 | 0 | 0 |

Constrained outputs have draft objects, but their disposition is ABSTAIN and
they contain no asserted content. They are not useful review drafts. Zero
erroneous final assertions here is a consequence of emitting no assertions.

Machine-readable primary results and field/stratum breakdowns:
[free comparison](free-01/comparison.json),
[constrained comparison](constrained-01/comparison.json).
Every original response, parse error, checker trace, and final draft is retained
in the corresponding [free](free-01/encounters.json) and
[constrained](constrained-01/encounters.json) encounter files.

## What checking retained and lost

The constrained raw response asserted only three supported values. All three
matched normalized gold content, but differed from the annotated boundary:

| Case / field | Raw exact patient quote | Annotated quote | Checked state |
|---|---|---|---|
| fresh-02 / chief complaint | `cough` | `A cough` | uncertain |
| fresh-11 / chief complaint | `moderate headache` | `A moderate headache` | uncertain |
| fresh-17 / chief complaint | `headache` | `A headache` | uncertain |

Checking lost **3/3 normalized-correct assertions** and retained none. The
strict paired metric calls these three exact mismatches removed, because of
the article boundaries; it does **not** establish removal of three false
clinical claims. There were no normalized-content assertion mismatches removed
or introduced. The five grounded exact fields described as salvaged are
missing-to-uncertain transitions matching gold uncertainty. The four grounded
exact fields lost are genuinely missing fields converted to uncertainty.
Complete evidence accuracy fell from 4/120 to 0/120.

Constrained state counts by field are shown below. The checked matches in every
row are exclusively uncertainty labels; free primary counts are zero throughout.

| Field (24 each) | Raw state correct | Checked state correct | Raw grounded exact | Checked grounded exact |
|---|---:|---:|---:|---:|
| Chief complaint | 3 | 0 | 0 | 0 |
| Duration | 1 | 1 | 1 | 1 |
| Severity | 1 | 1 | 1 | 1 |
| Medication | 3 | 4 | 3 | 4 |
| Allergy | 5 | 5 | 5 | 5 |

| Authored stratum (20 fields each) | Raw state correct | Checked state correct | Raw grounded exact | Checked grounded exact |
|---|---:|---:|---:|---:|
| Conventional controls | 4 | 0 | 3 | 0 |
| Medication uncertainty/nonuse | 3 | 3 | 3 | 3 |
| Medication conflict/coexistence | 1 | 1 | 0 | 1 |
| Allergy uncertainty/negation/coexistence | 2 | 3 | 2 | 3 |
| Partial information | 1 | 3 | 0 | 3 |
| Context complications | 2 | 1 | 2 | 1 |

## Failure localization

A posthoc, source-only replay of the existing context policy flags
**24/24 sources with global unknown context**. Ordinary questions such as
“What brought you in?” and “How long has it been there?” fall outside its
bounded question vocabulary. “Any medication allergies?” is also unattributed
because the question contains both medication and allergy cues.

`inspect_context` marks unsupported question/reply scope globally unsafe;
`extract_context_fields` then produces uncertainty for every field. In the
constrained traces this yields 98 transitions to uncertainty and 22 accepted
evidenceless uncertainty fields. Constrained candidate construction also uses
the same question attribution policy, limiting available supported candidates
before model selection. This is a limitation of the combined grammar/model/
checker path; the difference between free and constrained output cannot be
attributed solely to the language model or decoder.

The [context diagnostic](context_diagnostic.json) records every source and
question classification. Relevant implementation paths are
[question attribution and context extraction](../../nanoscribe/context.py),
[source inspection](../../nanoscribe/check_note.py), and
[constrained choices](../../nanoscribe/compact.py). Product code was not tuned
after observing these outputs.

Free generation has additional problems. Its first parser failures comprise
19 nonexact patient quotes, one ambiguous quote, one malformed conflict
payload, and three wrong slot counts. None are generation exceptions or
token-limit failures. For example, fresh-01 normalizes `Three days` to
`3 days`, breaking exact-source transport. Other failures concern meaning:

- fresh-04 assigns `penicillin` to both medication absence and allergy absence,
  although the allergy is positive.
- fresh-09 misses the unresolved medication conflict.
- fresh-10 labels medication use on different days as conflict, whereas the
  scalar schema rubric uses uncertainty for coexisting values.
- fresh-22 asserts `severe` from an unconfirmed clinician statement.
- fresh-24 substitutes penicillin for the affirmed ibuprofen medication and
  labels the positive penicillin allergy as absence.

An independent reviewer added a clearly **posthoc literal state-label**
diagnostic to avoid conflating complete parse failure with absence of any
state information. It finds 89/120 matching free labels (119 available) and
13/120 matching constrained labels (120 available). Free field counts are
24/24 chief complaint, 21/24 duration, 21/24 severity, 12/24 medication, and
11/24 allergy. None of the gold uncertainty, conflict, or missing fields have
matching free labels. The diagnostic ignores all payload and evidence validity;
it is not validated semantic accuracy and does not replace the primary scores.
See the reproducible [script](posthoc_state_labels.py) and
[complete diagnostic](posthoc_state_labels.json).

## Independent annotation and limitations

The [source author](authoring.md) wrote 24 fresh synthetic encounters across
six strata, without implementation, previous cases, labels, or model outputs.
Two new-context agents separately used the same
[annotation rubric](annotation_rubric.md), without each other's labels or
inference results. Their process records are retained for
[annotator A](annotation_process_a.md) and [annotator B](annotation_process_b.md).
They could see the authored strata. Separate contexts provide procedural
blinding, not independence of model family or training.

Pre-adjudication state agreement was **120/120 (100%; Cohen's kappa 1.0)**;
complete evidence-set agreement was **115/120 (95.83%)**. The coordinator had
implementation context but had not seen outputs for these cases. All five
evidence disagreements were resolved before inference; a further unanimous
conflict evidence set was narrowed to its two incompatible observations.
No states changed and no cases were excluded. Both original annotations and
all reasons remain in the [adjudication record](adjudication.md) and
[structured decisions](adjudication.json).

Gold contains 89 supported, 15 absent, four missing, 11 uncertain, and one
conflicting field. Its 104 supported/absent fields span all 24 cases. Fourteen
of the 15 natural-language denials fail the legacy contract's fixed absence
vocabulary; **they remain absent in primary gold**, rather than being changed
to fit the implementation. Each run records these transport errors separately.
The semantic oracle scores 120/120; the all-missing negative control scores
only the four genuinely missing fields. These validate the scorer's mechanics,
not the model or the clinical adequacy of the rubric.

The five-slot scalar schema represents multiple coexisting medications or
allergies as uncertainty. Evidence boundaries, temporal qualifiers, and patient
affirmations to clinician-supplied names affect strict content scores; an exact
annotation mismatch is not automatically a false medical assertion. Particular
interpretive choices for fresh-23 and fresh-24 are documented in adjudication.
One conflict and four missing fields are insufficient for reliable subgroup
estimates. This small, synthetic, exposed development set is not a clinical,
population, confirmatory, or human dual-IAA evaluation.

## Execution and reproducibility

The [protocol](protocol.md), sources, both original annotations, rubric,
adjudication, and gold were locked at **2026-10-23 23:35:14 UTC**, before the
first inference at **23:40:01 UTC**. The annotation-lock SHA-256 is
`a3764f3bb23440a9810db51ede32e27aaf8264e238d9f7fbfa3c0dad85209f60`.
No case selection, retries, prompt tuning, training, or inference-code changes
were performed using these outputs.

Both runs use the cached base **Qwen/Qwen2.5-1.5B-Instruct**, revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, with no adapter. Model safetensors
SHA-256: `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`.
Execution was offline, greedy, float16 on local Apple Silicon MPS, with Python
3.14.7, torch 2.11.0, and transformers 5.5.3. The existing few-shot free prompt
and existing structured constrained prompt differ, and the latter adds source
grammar priors. This is not a decoder-only controlled comparison.

The recorded `max_new_tokens=256` bound applies per free response or per
constrained slot-choice generation call, **not to the whole constrained
encounter**. Per-case token counts and timings are saved. Total measured run
times, including setup and hashing, were 176.826 seconds free and 101.577
seconds constrained, or **278.403 seconds combined; incremental paid compute
USD 0**. No network download or cloud job was used.

The [runner](../../scripts/run_p1_independent_dev.py) captures 167 Python source
files, all 12 input files including the lock, prompts, and seven checkpoint-file
hashes. Both runs have identical recorded input, source, and checkpoint hashes.
Exact prompts and hash-verified preserved input/source snapshots are inside each run directory.
Provenance: [free](free-01/provenance.json),
[constrained](constrained-01/provenance.json).

An independent pre-inference [harness review](harness_review.md) corrected
conflict evidence completeness, noncanonical manifest-path admission, and
failure-status recording using unrelated toy tests. All 31 harness tests passed
before inference. The reviewer subsequently audited both persisted runs and
reproduced the posthoc label diagnostic without model calls.

## Verification and preservation

The full verification completed with **1,720 passed and two skipped**: NanoScribe,
Nano Core, Nano AI, benchmarks, fabric/trajectory/wedge, existing script tests,
and the new harness tests. Direct fabric and trajectory checks passed. The
historical NanoScribe simulation benchmark also passed; it loads no model and
is separate from this failed transfer validation. Active-state, documentation,
and diff checks passed after the reporting changes. Commands and logs are
indexed in [verification.json](verification.json).

All **699 files in the protected pre-validation manifest** retain identical
SHA-256 hashes. All 11 locked annotation inputs, both input snapshots, and both
167-file source snapshots verify; live inference sources still match the runs.
See [integrity.json](integrity.json), [before](protected_before.json), and
[after](protected_after.json). H6/H7 and prior pilot/challenge evidence remain
unchanged. These development integrity checks do not create a freeze tag.

## Next engineering work

The immediate gaps are source question attribution and global uncertainty
propagation, followed by field-level recovery from malformed exact-source
responses. Medication/allergy handling must continue to distinguish use,
denial, uncertainty, temporal coexistence, explicit correction, and genuine
unresolved conflict. These repairs need new development cases, followed by
another separately authored and annotated validation set. Keep this run's
inputs and outputs intact as exposed diagnostic evidence. Independent human
and clinician review remains open; no P1 exit or treatment conclusion follows
from these results.
