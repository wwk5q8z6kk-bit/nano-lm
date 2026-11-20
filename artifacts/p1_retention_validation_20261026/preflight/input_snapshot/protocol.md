# Fresh retention development validation — 2026-10-26

This new synthetic development diagnostic is not a frozen evaluation, confirmatory experiment, clinical validation or representative prevalence estimate. All historical evidence remains immutable. No private data is used.

## Independence and ordering

1. A fresh-context author created 24 encounters in six strata without implementation, prior cases, annotations or model outputs.
2. Two fresh-context AI agents independently annotated the sources using the existing annotation rubric only; neither saw the other annotations or code.
3. The implementation coordinator used separate new regression cases and an independent patch reviewer. The coordinator has not inspected the new cohort or its annotations while developing. Source hashes are recorded in implementation_lock.json before that inspection.
4. A third fresh-context AI agent adjudicates source and annotation disagreements without implementation or model access. All 24 cases remain in the analysis. Natural-language denials retain their annotated states even when the unchanged transport contract rejects their wording.
5. Lock sources, both original annotations, gold, process notes, adjudication, rubric, authoring, this protocol and implementation_lock.json with SHA-256 before scored inference. These local integrity hashes are not scientific freeze tags.
6. Run cached base Qwen/Qwen2.5-1.5B-Instruct revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 offline, greedy, with the existing 256-token limit (per slot for constrained decoding), prompts, free generation and development constrained decoder. No training, downloads or paid compute.
7. Raw and checked outputs use the same single prediction per case per mode. The constrained raw path already uses grammar rules; it is not an unaided-model baseline. Strict parse failures remain failures in primary metrics. Separately report isolated raw-slot recovery without changing primary denominators.
8. Preserve every generation, error, diagnostic, code/input/model hash and snapshot. Do not rerun to select favorable outputs, tune code on this cohort or modify annotations after observing results.

AI agents supply procedural blinding, not independent model families or clinician/human dual-rater evaluation.

## Measurements and controls

All 24 cases and 120 fields remain denominators. Report state, normalized content, exact evidence and complete evidence separately; asserted fields retained, lost and salvaged; assertion mismatches removed or introduced; conflicts; final note assertions and whole-note abstention; by-field and by-stratum scores; and concrete constrained-menu availability. Run annotation-oracle and all-M negative controls before inference; report an all-U negative control in analysis. Compare saved responses against each original annotation set as a sensitivity analysis, without selecting a favorable policy. Do not equate internal checker agreement with semantic correctness. Earlier cohorts are different cases and do not establish a paired treatment effect. Preserve all prior run artifacts and frozen evidence.
