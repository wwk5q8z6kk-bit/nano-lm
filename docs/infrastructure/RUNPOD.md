# RunPod

RunPod is the **GPU execution backend** for training and CUDA experiments — not scientific authority.

**Cross-branch note:** `artifacts/nano_h6/runops/` bundles may exist only on feature branches — **not** on integration base `origin/master` @ `2ad06d2`. See [MODEL_RESEARCH_PROGRAM.md](../research/MODEL_RESEARCH_PROGRAM.md).

## Policy (current program)

| Class | Policy |
|-------|--------|
| Local zero-cost exploratory training (MPS/CPU) | **ALLOWED** |
| RunPod / paid GPU runs | **OWNER_GATED** |
| Frozen confirmatory execution | **PREREG + OWNER_GATED** |
| PHI / private data on cloud | **NOT_AUTHORIZED** |

## Workflow (when authorized)

```text
question → prereg → manifest → preflight → budget check → explicit auth
→ pod → checkpoints → eval → recompute → verify → terminate → cost record
```

## Related

- [ACTIVE_NOW.md](../ACTIVE_NOW.md) · [EXPERIMENT_STRATEGY.md](../research/EXPERIMENT_STRATEGY.md) · [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
