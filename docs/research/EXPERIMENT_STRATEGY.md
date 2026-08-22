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

## Paid compute

RunPod is execution backend only. Flow:

```text
question → prereg → manifest → preflight → budget check → explicit auth
→ pod → checkpoints → eval → recompute → verify artifacts → terminate → cost record
```

No paid compute authorized by default. See [infrastructure/RUNPOD.md](../infrastructure/RUNPOD.md).

## Reproducibility

Tags, manifests, content-addressed JSON: [infrastructure/REPRODUCIBILITY.md](../infrastructure/REPRODUCIBILITY.md)
