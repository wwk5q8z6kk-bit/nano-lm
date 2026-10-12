# Model Research Program

Three layers run through all capability programs (P1–P9).

## Compute

```text
Original Nano
→ trained from scratch locally on Apple Silicon

Current Nano research
→ local development + Kaggle free GPU + RunPod exact-runtime compute

Future deployment
→ chosen independently from training infrastructure
```

Kaggle and RunPod are complementary active backends. Use Kaggle when its free
runtime fits; use RunPod for exact GPU/runtime requirements, persistent storage,
or managed inference. Routine RunPod training is
**ALLOWED_WITHIN_ACTIVE_EXPERIMENT_BUDGET**. Materially costly runs require
experiment-scoped authorization; confirmatory evidential runs require prereg
plus experiment-scoped authorization. Training venue does not determine
deployment venue. See [ACTIVE_NOW.md](../ACTIVE_NOW.md) and
[infrastructure/RUNPOD.md](../infrastructure/RUNPOD.md).

## Integration status (this tree)

Base branch: `origin/master` @ `e7c51fe`. Paths below reflect **what exists in this checkout**.

| Component | State | Canonical location (this tree) |
|-----------|-------|--------------------------------|
| Original 3.15M stack | **integrated** | `pretrain/`, `sft/`, `scribe/` |
| Fabric verification | **integrated** | `fabric/` |
| Wedge v1 | **integrated** | `wedge_v1/` |
| Nano AI through H6 | **integrated** | `nano_ai/`, `artifacts/nano_h6/` |
| Span-port / register work | **not yet integrated** | cross-branch — source branch + commit |
| Measurement-integrity reconciliation | **not yet integrated** | cross-branch — commit `9fe5b6b6` (evaluate separately) |

H6 is terminal and rejected. Its exact result is authoritative for that
experiment; it is not permission to rerun the one-shot evaluation. Span-port,
P4, and CUAD work remain cross-branch until separately reviewed.

---

## A — Mechanism models (microscopes)

**Scale:** ~3M–100M from-scratch (e.g. original nano-lm trunk, `scale/`, `stage_m/`)

**Purpose:** controlled experiments, architecture/objective tests, copy/retrieval/induction

**Repo (integrated):** `pretrain/`, `sft/`, `scribe/`, `trajectory/`

## B — Compact production models

**Scale:** efficient pretrained / adapted / distilled

**Integrated lineage:** `nano_ai/` through the frozen H6 decision. **Cross-branch
lineage:** later span-port/register experiments, which are not current program
authority.

## C — Teacher / reference models

Ceilings, synthetic data, judging, distillation — not default deploy.

## Evidence pointers (integrated artifacts)

- Paper α: `papers/latex/`, `trajectory/results_*.json`
- E1: `trajectory/results_e1_utility.json`
- E4: `trajectory/results_e4_utility.json`
