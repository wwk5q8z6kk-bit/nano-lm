# P1 question attribution and field-preservation repair

This development repair follows the earlier v2 repair and its preserved failed
transfer validation. It changes the opt-in NanoScribe context path. Frozen H6/H7
adapters, results and previous development evidence are outside the edit scope.

## Behavior

Question attribution now uses anchored request structures and bounded symptom
noun phrases. Allergy questions containing medication modifiers stay allergy
questions. Compound, third-person, historical, future and instruction prompts
cannot authorize positive assertions. Lexical targets can quarantine fields but
cannot establish evidence.

Unknown replies quarantine requested or mentioned fields, while unrelated source
context remains available for review. Explicit pure other-person/old-episode
questions do not erase current patient facts; mixed scope or first-person
corrections still invalidate the implicated fields. Named corrections and bounded
immediate anaphoric/self-retractions are checked. Explicit unbounded reassignment
of the patient's source still quarantines the whole note.

Complete simple replies have exact source spans, with guards against non-use,
qualifiers, coexisting values, unnamed pills and reaction-only allergy replies.
This is a bounded lexical policy, not a general semantic or clinical verifier.
It can still miss unfamiliar corrections or reject valid unfamiliar language.

Exactly five correctly ordered output slots permit field-local parse recovery.
Each isolated slot must pass the unchanged frozen strict parser and unique
patient-span locator. Invalid slots become uncertain, every owner of duplicated
evidence is rejected, and malformed overall structure still fails. No fuzzy
quotation repair or guessed source offsets are used. Source conflicts retain
all observations, even when a malformed or incorrect proposal omitted one side.

Versions: context-source-scope-v3; nanoscribe.context-state-diagnostics.v3;
typed-context-source-trie-v7. Diagnostics retain the original response, raw
slots, strict and local parse errors, and acceptance decisions. Recovery count
means parse-admitted slots, not semantically correct or usable fields.

## Separate development and review

New injected development cases are in test_question_attribution.py,
test_partial_context_parse.py and test_scope_repair.py. Existing tests that
required unnecessary whole-note abstention now require local quarantine and
retention of the valid complaint. Source reassignment tests still abstain.

The initial attribution set reproduced 35 failures; field recovery reproduced
12; the first scope set reproduced 18. Independent patch review found mixed
scope corrections being skipped, bare nonvalues being asserted, unbounded
reassignment missed, pronoun/self-retractions missed, and a duration enum error.
Regression cases cover these failures. Final focused NanoScribe plus independent
validation-harness check: 406 passed (final-targeted.log).

implementation_lock.json records 170 source/test/harness hashes after these
repairs, before root inspection of the new validation cohort. The cohort was
authored and labelled by separate new-context agents; a third blinded agent
adjudicated annotations. Model outputs did not inform further edits in this run.
See the separate [validation report](../p1_scope_validation_20261023/REPORT.md)
for model results; passing software tests do not establish model quality.

## Verification and preservation

Full-suite commands and logs are retained in verification.json and verification/.
Final results: 1,843 passed, two environment skips. This includes 375 NanoScribe,
316 Nano Core, 499 learned-core, 500 benchmark, 32 foundation and 121 script
tests. Direct fabric (8/8), trajectory (7/7) and synthetic scribing benchmark
checks also pass. The final NanoScribe group was rerun after the last two
self-retraction regressions; original logs remain alongside the final log.
The final reporting gate record includes documentation checks, diff whitespace
and rechecks after documentation updates. Direct fabric, trajectory and synthetic
benchmark checks overlap with pytest and are not added to the test total.

protected_before.json records 1,099 file hashes: all previously protected
evidence paths and the entire earlier independent development validation,
including its source/input snapshots. [integrity.json](integrity.json) confirms
all 1,099 hashes unchanged, both model runs matching the 170-file implementation
lock, and every saved input/source snapshot matching its recorded hash.
Documentation and whitespace gates pass in [final_gates.json](final_gates.json).
The external checkpoint symlink is not traversed or modified. No tags, frozen
screen runs, paid compute, training, private data or clinical validation are part
of this repair.
