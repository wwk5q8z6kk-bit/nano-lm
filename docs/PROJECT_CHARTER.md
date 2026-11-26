# Project charter

Build a general-purpose language model that can compete at the frontier. NanoLM's research question is how much language understanding, reasoning and adaptability its own trained weights can deliver at a given training and inference cost. Frontier capability is the objective. Establishing it requires independently reproducible comparisons against contemporary models on multiple domains.

## What counts as progress

- Train or recover native Nano checkpoints and establish their identity, training data, tokenizer and measured behavior.
- Compare architectures at matched data and token budgets. Report parameters, compute, memory, throughput and quality together so a change in one can be weighed against its cost.
- Measure transfer to held-out sources and tasks. Report results for the base model, adapted model and any system using tools or retrieval separately. Teachers and external weights can supply useful comparisons.
- Scale candidates when a repeatable improvement justifies the next experiment. Model size, architecture and deployment hardware remain research choices.

## Program boundary

NanoScribe was the prior medical application. It is outside the active research, product and compute program. Its code and evaluations remain available to reproduce historical results. A future standalone application would need its own scope and evidence. The former scribing, summarization and charting sequence places no prerequisites on general-model research.

The original 3.15M from-scratch LM anchors the model lineage. Paper α, E1, E4 and the terminal H6/H7 decisions retain their original scientific scope. Public or explicitly licensed research data are eligible under their terms. Private owner data and PHI are outside authorization.

The current experiments, compute rules and evidence requirements are in the [execution plan](EXECUTION_PLAN.md) and [active state](ACTIVE_NOW.md).
