# Independent implementation review

Reviewer: delegated `retention_patch_review` agent, separate from the implementation agent.

Scope: read-only review of the development question/reply attribution, constrained menus, and checked-note path. Reproductions were independently authored synthetic development cases. No files under `artifacts/p1_retention_validation_20261026` were read, and no prior validation sources, gold labels, or model outputs were read. No implementation files were edited by the reviewer. This is an AI implementation review, not clinical evaluation or human annotation.

## Findings and verified repairs

1. **Temporal adjunct included in medication value.** `Which medicine did you try?` / `I took naproxen today.` previously bound `naproxen today` and rejected the correct `MED:S[naproxen]`. Retest: the correct medication remains supported with value `naproxen`; `this morning` also remains supported.
2. **Non-medication relation admitted as medication.** `I took an allergy test.` previously passed as a supported medication. Retest: `MED:S[an allergy test]` becomes uncertain, with no asserted medication value.
3. **Consecutive patient-turn retraction lost its target.** Following `I took ketoprofen.`, a separate patient turn containing `That was last year.`, `That was before this episode.`, `It was hypothetical.`, or `That belonged to my brother.` previously retained ketoprofen as supported. All four exact reproductions now produce uncertain medication.
4. **Unanswered request became missing after an explicit different-field answer.** A medication question answered `I have nausea.` previously produced supported complaint plus missing medication. Retest: nausea remains supported and medication is uncertain.
5. **Efficacy follow-up erased medication use.** Following `I took naproxen.`, `Has the medication helped?` / `I could finally sleep.` previously made medication uncertain. Retest: naproxen remains supported.
6. **Allergy half of relation-word guard missing.** `Any allergies?` / `The medication gives me hives.` and `An allergy test gives me hives.` initially passed as supported named allergens. Final targeted retest after the shared MED/ALG guard: both become uncertain with no asserted allergy value. Positive control `Latex gives me hives.` remains supported with value `latex`.

All nine exact reproductions for findings 1–5 passed before the final allergy-guard repair. The reviewer also ran the four focused test files `test_reply_retention_dev.py`, `test_constrained.py`, `test_question_attribution.py`, and `test_question_coverage_dev.py`: **185 passed** at that intermediate reviewed state. This count is not the final full-suite result.

After the implementation lock, the reviewer reran only the remaining allergy issue and its named-trigger control as requested: **3/3 passed**, with independent final note verification passing in every case. No new blocker was found in this bounded final retest. This does not establish unrestricted clinical-language correctness or fresh-cohort model quality.

## Review discipline

The implementation agent received each actionable reproduction before the corresponding repair. The reviewer checked the final behavior through `scribe(..., context_policy="context", solver=compact_solver(..., binder="context"))`, injecting the exact proposed model slot and inspecting the returned checked note. This validates the source-to-note boundary rather than merely checking a helper's return value. Frozen evidence and prior reports were not modified.

Recorded UTC: 2026-10-26T13:44:32.660505+00:00
