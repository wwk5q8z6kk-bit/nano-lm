# Execution plan

## General-model experiments

| Work | Design | Decision |
|---|---|---|
| Recover native baselines | Validate the architecture, tokenizer, checkpoint hash and training provenance for recovered native weights. Evaluate the same text and context across eligible models with a simple control. Report domain-level results and sample sizes. | Identify the strongest reproducible reference and the limits of the comparison. |
| Test architecture choices | Compare the candidate to a competitive dense or hybrid reference. Match tokens, training data, tokenizer and evaluation protocol. Record actual throughput, compute and cache accounting. Replicate promising differences across seeds. | Choose which architecture warrants the next larger training run. |
| Test generalization | Use separate held-out sources and evaluate language modeling, instruction following and reasoning. Track overlaps, include simple baselines and report model-only and tool-assisted outcomes separately. | Identify a specific capability gap for pretraining or post-training. |

These experiments are exploratory. Attach code and data revisions, seed, termination condition, artifact destination and budget to paid runs. Keep input rights and possible pretraining overlap visible. Use local resources or free Kaggle when they fit, Google Cloud within the verified credit allocation, and RunPod under its separate campaign limits. Confirmatory research retains the preregistration and authorization rules in [active state](ACTIVE_NOW.md).

## Historical scribing program

NanoScribe is outside active execution. Preserve open work and historical results for a separate disposition. The previous plan records phases A–C, their artifacts and the clinical evaluation boundaries. [Earlier scribing program](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/46c65f0044e81ac73251e18a5a7d69a7299a8079/docs/EXECUTION_PLAN.md)

## Verification

Run the repository checks in [AGENTS.md](../AGENTS.md) and the experiment-specific replay documented with each result.
