# Model research program

The active research compares native NanoLM language models, starting with general-language baselines. Native weights, third-party base models after adaptation, and systems with tools each have distinct evaluation contracts.

## Research map

| Lineage | Source | What can be tested |
|---|---|---|
| Original 3.15M native LM | Integrated `pretrain/` and `sft/` | Causal LM pretraining and alignment mechanisms at small scale. |
| Native dense and hybrid checkpoints | Recovered experimental weights | Strict loading, tokenizer identity, common-text evaluation, throughput and memory. Check checkpoint and training provenance before relating runs. |
| Architecture discovery | [PR 174](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/174), [PR 175](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/175), [PR 176](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/176) | Cross-branch implementations and controlled comparisons. Similar names can represent different backbones. Use the exact revision and experiment manifest. |
| Post-training and evaluation | [PR 171](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/171) | Distinguish native model training, external-weight adaptation, verifier behavior and measured student outcomes. |
| Foundation model research proposals | [PR 172](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/172) | Candidate studies and evaluation designs. Parameter targets and proposed architectures are hypotheses. |
| Historical scribing | Integrated `nanoscribe/`, `nano_ai/` and frozen artifacts | Reproduction of the former application's scoped findings. |

## Comparisons

Use identical scored text, context and tokenizer when possible. If tokenizers differ, account for the exact scored bytes before comparing bits per byte. Compare quality with training tokens, parameter count, wall time and memory. State known or unknown data overlap, seed count and the contribution of any external model or runtime.

The [execution plan](../EXECUTION_PLAN.md) identifies which comparison to run next. The [charter](../PROJECT_CHARTER.md) defines the research objective.
