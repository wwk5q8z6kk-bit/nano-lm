# Experiment Strategy

## Sequential pipeline

```text
premise → regime → utility → kill gate → architectural response
```

Never run an experiment that cannot change the roadmap.

## Preregistration

- Write hypothesis, instrument, pass/fail bars, and utility **before** execution
- Commit prereg to git when possible (`papers/PREREG_*`, `trajectory/PREREG_*`, component `PREREG_*.md`)
- Post-hoc bar movement is bar-chasing

## Construct validity

- Distinguish agent-applied rubric audits from human/clinician evaluation
- E3-style audits are **not** dual-clinician IAA

## Regime discipline

For general-model studies, specify the capability being tested, a baseline, matched inputs and a result that would change the next training decision. The [execution plan](../EXECUTION_PLAN.md) defines current comparisons.

E1 closed its original task. E4 tested the frozen utility for the historical R★ regime and returned KILL. Those verdicts retain their original scope.

## Compute backends

Use local CPU or Apple Silicon for development, preprocessing, evaluation and affordable training. Prefer free Kaggle when the runtime fits. Google Cloud can support finite jobs within the verified credit allocation. RunPod provides exact GPU runtimes, persistent storage and managed inference under its separate campaign budget. Record the chosen backend and expected cost with each paid experiment.

```text
research question
→ experimental design
→ preregistration when evidential
→ dataset + model manifest
→ local preflight
→ cheapest valid backend environment
→ training / adaptation
→ checkpoints
→ evaluation
→ independent recomputation
→ artifact sync
→ cost record
→ teardown
→ failure-to-architecture update
```

Using RunPod itself is established project infrastructure. A new **materially costly or confirmatory** run still follows experiment-scoped budget/authorization in the active plan. No PHI/private clinical data on RunPod. Details: [infrastructure/RUNPOD.md](../infrastructure/RUNPOD.md).

## Reproducibility

Tags, manifests, content-addressed JSON: [infrastructure/REPRODUCIBILITY.md](../infrastructure/REPRODUCIBILITY.md)
