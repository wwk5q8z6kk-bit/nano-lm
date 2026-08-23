# RunPod

**RunPod is Nano's active primary GPU training and experimental-compute backend.**

Nano uses RunPod for pretrained-model adaptation, controlled training experiments, CUDA-dependent validation, scaling experiments, and other workloads that exceed or are poorly suited to local Apple Silicon.

RunPod is execution infrastructure, not scientific authority: experiment identity comes from the preregistration, manifest, code revision, dataset manifest, configuration, seeds, checkpoints, and resulting immutable artifacts.

**Cross-branch note:** `artifacts/nano_h6/runops/` bundles may exist only on feature branches — **not** on integration base `origin/master` @ `2ad06d2`. See [MODEL_RESEARCH_PROGRAM.md](../research/MODEL_RESEARCH_PROGRAM.md).

## Policy (current program)

The distinction is **cost / risk / evidential significance**, not “free local versus paid cloud.”

| Class | Policy |
|-------|--------|
| Local zero-cost exploratory training (MPS/CPU) | **ALLOWED** |
| Routine RunPod training | **ALLOWED_WITHIN_ACTIVE_EXPERIMENT_BUDGET** |
| Materially costly run | **EXPERIMENT_SCOPED_AUTHORIZATION** |
| Confirmatory / evidential run | **PREREG + EXPERIMENT_SCOPED_AUTHORIZATION** |
| PHI / private data on cloud | **NOT_AUTHORIZED** |

```text
training_backend = RUNPOD
training_status = ACTIVE

routine_runpod_training =
ALLOWED_WITHIN_ACTIVE_EXPERIMENT_BUDGET

materially_costly_run =
EXPERIMENT_SCOPED_AUTHORIZATION

confirmatory_evidential_run =
PREREG_PLUS_EXPERIMENT_SCOPED_AUTHORIZATION

phi_or_private_data =
NOT_AUTHORIZED
```

Do not gate every RunPod training job merely because the GPU is paid. Routine training on the established backend is in-workflow within the active experiment budget.

## Workflow

```text
question → design → local preflight
→ routine RunPod training (within active experiment budget)
→ or, if materially costly: experiment-scoped authorization
→ or, if confirmatory/evidential: prereg + experiment-scoped authorization
→ pod → checkpoints → eval → recompute → verify → terminate → cost record
```

## Related

- [ACTIVE_NOW.md](../ACTIVE_NOW.md) · [EXPERIMENT_STRATEGY.md](../research/EXPERIMENT_STRATEGY.md) · [REPRODUCIBILITY.md](REPRODUCIBILITY.md) · [TOOL_CALLING.md](TOOL_CALLING.md)
