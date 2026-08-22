# Model Research Program

Three layers run through all capability programs (P1–P9).

## Compute

```text
Original Nano
→ trained from scratch locally on Apple Silicon

Current Nano research
→ local development + RunPod GPU training

Future deployment
→ chosen independently from training infrastructure
```

**RunPod** is the primary GPU training/adaptation/CUDA backend when **owner-gated** paid runs are authorized. Local zero-cost exploratory training on Apple Silicon/CPU is **ALLOWED**. Training on RunPod does **not** imply Nano must deploy in the cloud — compact/local/private deployment remains a long-term optimization axis. See [ACTIVE_NOW.md](../ACTIVE_NOW.md) and [infrastructure/RUNPOD.md](../infrastructure/RUNPOD.md).

## Integration status (this tree)

Base branch: `origin/master` @ `2ad06d2`. Paths below reflect **what exists in this checkout**.

| Component | State | Canonical location (this tree) |
|-----------|-------|--------------------------------|
| Original 3.15M stack | **integrated** | `pretrain/`, `sft/`, `scribe/` |
| Fabric verification | **integrated** | `fabric/` |
| Wedge v1 | **integrated** | `wedge_v1/` |
| Nano AI / H6 transfer | **not yet integrated** | cross-branch — e.g. `cursor/span-port-route-b-182e` |
| Span-port / register work | **not yet integrated** | cross-branch — source branch + commit |
| Measurement-integrity reconciliation | **not yet integrated** | cross-branch — commit `9fe5b6b6` (evaluate separately) |

Until selectively ported, **do not** treat `artifacts/nano_h6/` or `nano_ai/` as present locally. Canonical docs may reference them only as **cross-branch research lineage**.

---

## A — Mechanism models (microscopes)

**Scale:** ~3M–100M from-scratch (e.g. original nano-lm trunk, `scale/`, `stage_m/`)

**Purpose:** controlled experiments, architecture/objective tests, copy/retrieval/induction

**Repo (integrated):** `pretrain/`, `sft/`, `scribe/`, `trajectory/`

## B — Compact production models

**Scale:** efficient pretrained / adapted / distilled

**Cross-branch lineage:** H6 transfer, span-port, register experiments — **not yet integrated** in this tree (`artifacts/nano_h6/`, `nano_ai/` when ported)

## C — Teacher / reference models

Ceilings, synthetic data, judging, distillation — not default deploy.

## Evidence pointers (integrated artifacts)

- Paper α: `papers/latex/`, `trajectory/results_*.json`
- E1: `trajectory/results_e1_utility.json`
- E4: `trajectory/results_e4_utility.json`
