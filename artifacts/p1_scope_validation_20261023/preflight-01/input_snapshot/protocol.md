# Fresh scope-repair development validation — 2026-10-23

This is a new, exposed synthetic development diagnostic, not H6/H7, a
confirmatory experiment, clinical validation, or representative prevalence
estimate. Historical evidence, including the previous independent development
cohort and all its results, remains immutable. No private data is used.

## Independence and ordering

1. A new-context author creates 24 encounters in six requested strata without
   implementation, earlier cases, annotations, or inference outputs.
2. Two other new-context AI agents independently label all five fields from
   sources.json and annotation_rubric.md only. They cannot coordinate or inspect
   implementation, previous evidence, or each other's labels.
3. Product implementation and its independent code review use separate injected
   cases. The coordinator does not inspect this cohort until the code is fixed
   and records implementation hashes before examining annotations.
4. A third new-context AI agent adjudicates all disagreements before inference,
   seeing the rubric, source and both original annotations, but no code or
   model output. The coordinator reviews structural admission afterward. No examples are
   excluded or relabelled to fit implementation. Natural-language denials stay A
   even where the unchanged legacy transport contract rejects them.
5. SHA-256 lock sources, original annotations, process notes, adjudication,
   gold, rubric, authoring note and this protocol before any scored inference.
   These local integrity hashes are not scientific freeze tags.
6. Run cached base Qwen/Qwen2.5-1.5B-Instruct revision
   989aa7980e4cf806f80c7fef2b1adb7bc71aa306 offline, greedy, with a 256-token
   generation bound (per slot in constrained decoding). Use existing prompts,
   free generation and the repaired typed constrained decoder. USD 0 paid
   compute. No training, downloads or tuning against this cohort.
7. Compare raw output, context-checked output and final note from exactly one
   cached prediction per source per mode. Raw constrained output already uses
   decoder rules; it is not an unaided model baseline. Raw strict parse failures
   remain failures in aggregate metrics, even if checked field recovery succeeds.
8. Preserve responses, errors, diagnostics, comparisons, timing, source/input/
   code/model hashes and snapshots in new directories. Do not overwrite or rerun
   to select a favorable result. Verify inputs and protected evidence afterward.

Both annotators are AI agents, not independent human or clinician raters.
Separate contexts provide procedural blinding, not independent model families.
Adjudication is also blind to implementation; the coordinating agent has
implementation context and must not tune implementation or labels after review.

## Measurements

All 24 encounters and 120 fields remain denominators, including failures.
Report state, normalized-value and exact-source grounding separately; usable
assertions retained, lost or salvaged by checking; incorrect assertions removed
or introduced; conflicts; parse errors; note abstention with usable gold; and
breakdowns by field and stratum. Do not equate parser validity or internal
checker agreement with semantic correctness. A small zero-error sample cannot
establish clinical safety. Two compatible medicine/allergy values use U because
the scalar schema cannot represent both; explicit resolved corrections use the
resolved state rather than C. The oracle must score perfectly, and an all-M
negative control must fail on nonmissing fields before model execution.

## Annotation sensitivity, specified before inference

The independent adjudicator changed eight shared A/A labels for conventional
no-known-allergies wording to U under the rubric's qualified-knowledge rule.
Keep that adjudicated gold as the primary analysis, and rescore the same saved
model responses under (1) each complete original annotation set and (2) a
variant restoring only those eight ALG labels and their original A evidence.
Report all variants; do not select a favorable label policy or run inference
again. These analyses also expose boundary sensitivity. The primary agreement
statistics always use the two original, unmodified annotation sets.
