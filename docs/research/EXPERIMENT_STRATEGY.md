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

## Regime discipline (post-E1)

E1 closed the **old closed task**. New generative experiments require:

1. Written regime where classical is insufficient
2. Frozen utility for that regime
3. Kill gate result — E4 executed this for R★ v1 (KILL)

## Compute / RunPod

**RunPod is Nano’s primary GPU training and experimental-compute backend** (active). Local Apple Silicon/CPU covers development, smoke tests, analysis, preprocessing, evaluation, and small/cheap experiments.

```text
research question
→ experimental design
→ preregistration when evidential
→ dataset + model manifest
→ local preflight
→ RunPod environment
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
