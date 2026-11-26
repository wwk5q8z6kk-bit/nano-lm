# Fresh P1 development validation — 2026-10-23

This is a new, exposed development diagnostic of the repaired local working
tree. It is not H6/H7, a confirmatory experiment, a clinical validation, or an
estimate of real-world prevalence. All encounters are synthetic and contain
no private data. No historical evidence or freeze tags may be changed.

## Independence and order

1. A source author in a new agent context writes 24 natural-language encounters
   without reading implementation, earlier examples, labels, or model outputs.
2. Two other new-context agents independently annotate the identical source
   file using annotation_rubric.md. Neither sees the other annotation or any
   inference output. They may not inspect implementation or previous evidence.
3. The coordinating agent compares labels, records agreement and every
   adjudication, and resolves targets before model inference. That agent has
   implementation context; this limitation is explicit. No cases are excluded
   to accommodate the implementation's grammar. Unsupported gold transport is
   reported, not silently converted to uncertainty.
4. Sources, original annotations, adjudicated gold, rubric, protocol, and
   adjudication record are SHA-256 locked before the first scored inference.
   These are local development integrity hashes, not scientific freeze tags.
5. Run the cached base Qwen/Qwen2.5-1.5B-Instruct revision
   989aa7980e4cf806f80c7fef2b1adb7bc71aa306 locally, offline, greedy, with a
   256-token bound. Test both existing few-shot free generation and existing
   typed constrained decoding. No prompt tuning, model training, case
   selection, or inference-code changes based on these outputs in this run.
6. Within each decoding mode, compare the raw response with context-checked
   fields and the final draft from exactly the same cached generation. Raw
   constrained output already includes decoder rules and is not an unaided
   model baseline. The free-generation arm supplies that complementary view.
7. Preserve every raw response, error, diagnostic, draft, per-field comparison,
   timing, source/code/model hashes, and code/input snapshots in new output
   directories. Never overwrite an existing run. Recheck input and historical
   evidence hashes after execution.

Both annotators are AI agents, not clinicians or independent human raters.
Separate contexts provide procedural blinding, not independence of model
family or training. All agreement claims must identify this limitation.

## Primary comparisons

Denominators are all 24 encounters and all 120 fields, including failures.
Report state accuracy separately from strict normalized-value/exact-evidence
accuracy. Report valid asserted fields retained or lost by checking, invalid
assertions removed or introduced, conflict recall, malformed raw responses,
and whole-note abstentions despite at least one gold S/A field. Tabulate by
field and authored stratum. Evidence boundary disagreement is distinct from
state disagreement. Do not infer clinical safety from a small zero-error count.

The five-slot scalar schema cannot encode coexisting distinct medication or
allergy values; annotate those U with both source observations. Natural
explicit denials remain A even when the legacy output contract rejects their
wording. Expose contract transport errors separately. Primary raw semantic
scoring admits syntactically well-formed natural denials without that fixed
vocabulary; report separately whether the output contract accepts them. A raw
syntax failure remains a failed response in aggregate metrics; do not drop it.

The oracle must score perfectly under the independent scorer. The all-missing
negative control must expose failure on non-missing fields. Test admission and
paired scoring before model execution. Review results for measurement defects;
any correction must retain original outputs and document its chronology.

Incremental paid compute budget: USD 0. No network model downloads.
