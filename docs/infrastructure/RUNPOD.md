# RunPod

RunPod is the **experimental execution backend** for GPU workloads — not scientific authority.

## Authority

- Preregistration and manifests define what an experiment means
- Result JSONs and tags in `trajectory/` / `artifacts/` are the record
- RunPod logs and checkpoints are inputs to that record

## Required flow (when authorized)

```text
experiment question
→ preregistration
→ run manifest
→ local preflight
→ cost/budget check
→ explicit paid-run authority
→ Pod launch
→ checkpoints
→ evaluation
→ independent recompute
→ artifact verification
→ external sync
→ Pod termination
→ cost record
```

## Default

**NOT_AUTHORIZED** — see [ACTIVE_NOW.md](../ACTIVE_NOW.md).

## No PHI

Do not upload private medical data or PHI to cloud pods.

## Related

- H6 artifacts: `artifacts/nano_h6/runops/`
- Reproducibility: [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
