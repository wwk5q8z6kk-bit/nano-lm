# Active research

Updated 2026-11-20. The owner redirected NanoLM to general-purpose frontier model development. NanoScribe is outside the active program.

## Status (canonical — mirrored in `ACTIVE_NOW.json`)

| Field | Value |
|-------|-------|
| `program_execution_status` | `FRONTIER_MODEL_RESEARCH_ACTIVE` |
| `capability_frontier` | `GENERAL_PURPOSE_LANGUAGE_MODEL` |
| `current_gate` | `MATCHED_GENERAL_MODEL_BASELINE` |
| `evidence_core` | `FROZEN_UNTOUCHED_BY_DOC_PR` |
| `training_backend` | `KAGGLE_GOOGLE_CLOUD_AND_RUNPOD` |
| `training_status` | `ACTIVE` |
| `local_zero_cost_exploratory_training` | `ALLOWED` |
| `routine_runpod_training` | `ALLOWED_WITHIN_ACTIVE_EXPERIMENT_BUDGET` |
| `materially_costly_run` | `EXPERIMENT_SCOPED_AUTHORIZATION` |
| `confirmatory_evidential_run` | `PREREG_PLUS_EXPERIMENT_SCOPED_AUTHORIZATION` |
| `phi_on_cloud` | `NOT_AUTHORIZED` |
| `phi_or_private_data` | `NOT_AUTHORIZED` |
| `clinical_claims` | `FORBIDDEN_WITHOUT_EXTERNAL_HUMAN_VALIDATION` |

## Current evidence gate

Compare authenticated native checkpoints on common public text, then test a candidate architecture against a strong matched baseline. Record training data lineage, evaluation overlap, context, accuracy or loss, speed and memory. Preserve negative and inconclusive results. A research target or a tool simulator score does not establish model capability.

The original pretraining and SFT code is integrated. The newer architecture, post-training and scaling studies have separate code and evidence lineages. The [model research map](research/MODEL_RESEARCH_PROGRAM.md) identifies the relevant research branches. The [execution plan](EXECUTION_PLAN.md) defines the decision each experiment informs.

## Historical evidence

H6 is terminal and rejected. H7's consumed screen has an invalid gold instrument. Both retain their original decisions. The former application has no current engineering or compute allocation. Its evidence and detailed status are preserved in the [Earlier scribing program](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/46c65f0044e81ac73251e18a5a7d69a7299a8079/docs/ACTIVE_NOW.md).
