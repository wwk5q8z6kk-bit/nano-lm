# P1 annotation transport and next bounded work

## Implemented engineering repair

`nano_ai/training/annotated_evaluation.py` converts supplied state/span targets
into evaluator-owned `FixtureCase` gold, using exact Patient spans and the Nano
contract. It does not reconstruct labels from question vocabulary. The caller
must supply independently specified counts for every field/state cell, including
zeros. Empty/duplicate corpora, malformed or ambiguous spans, invalid denials,
metadata disagreement and quota mismatch fail before inference.

Use `evaluate_annotated(solver, examples, expected_counts=counts)` for a **future**
instrument. `counts` maps each `FieldName` to every `FieldState` and its integer
count. It must come from the instrument specification, not from the same parsed
targets being checked. The existing solver-neutral evaluator passes only
`NanoInput` to inference. Targets stay on the evaluator side; this module is
never an inference adapter and must not be wired into NanoScribe extraction.

The independent toy suite exercises all five states and exact conflict offsets,
a perfect test-only oracle, and an all-missing negative control. Oracle success
establishes plumbing correctness, not model quality or semantic annotation
correctness. The module cannot detect a semantically wrong but well-formed
source annotation; independent annotation review remains necessary.

No frozen H6/H7 evaluator, source recipe, consumed screen or result is changed.
H7 remains consumed with invalid transported gold; no treatment conclusion is
recovered by rescoring it.

## Bounded exploratory pilot (executed)

The recipe below was executed as a separate exposed development diagnostic.
See the [20-dialogue pilot report](../../artifacts/p1_pilot_v1/REPORT.md).
Annotation review was a separate logical pass by the same coding agent, not an
independent annotator. The executable checkpoint was pinned base Qwen2.5-1.5B,
not the historical fine-tuned compact adapter. Two complete prompt runs cost $0;
the revised prompt reached two useful checked review drafts but failed most
cases. Constrained generation and broader independently checked source context
are the next engineering work; this is not P1 admission.

1. Author 20 new public/synthetic **development-only** dialogues with novel
   question wording, all five field states, explicit Patient spans and at least
   four conflict and four uncertainty instances. Keep source world IDs together.
   Have a separate annotation review verify the actual text, then record the
   independently counted 25-cell field/state table and content hashes.
2. Admit the instrument only if the source-target round trip, perfect oracle,
   all-missing negative control, duplicate-world/split checks and independent
   quota check pass. Add question-paraphrase pairs to catch vocabulary coupling.
   These exposed development examples cannot later become confirmatory tests.
3. Run the checked reference solver first. Select one preserved compact learned
   adapter only after verifying its checkpoint, runtime and input/output contract.
   Keep raw proposals, final contract outputs, state confusion, exact evidence
   errors, failures and stage timings separate. Do not pool scores with historical
   H6/H7 or differently defined compact/OpenResearch results.
4. Pass that adapter to the existing transcript-only NanoScribe entrypoint and
   inspect encounter records and source-checked review drafts on the same public
   development cases. Abstentions and unresolved conflicts remain visible.
   This is an engineering transfer diagnostic, not P1 admission or clinical proof.

Budget: the next pilot starts with **$0 incremental paid compute**, local CPU
development and inspection of preserved artifacts. No retraining or fresh sealed
screen is part of this recipe. If the selected learned runtime requires GPU,
record its exact runtime, wall-clock estimate, rate, stop condition and reconciled
campaign spend before selecting Kaggle or a bounded RunPod job. RunPod speed is
a valid reason to choose it within an active experiment budget. The standing
$180 autonomous envelope, $200 hard cap and $20 owner reserve apply; account-wide
spend is not evidence of campaign spend or remaining campaign balance.

A new scientific screen needs its own identity, disjoint data, reviewed annotation
transport, preregistered comparison and experiment-scoped authorization where
required. It must not reuse or rename the consumed H7 screen.

Verification:

```bash
python3 -m pytest nano_ai/tests/test_annotated_evaluation.py -q
```
