# Model Research Program

Three layers run through all capability programs (P1–P9).

## Compute

```text
Original Nano
→ trained from scratch locally on Apple Silicon

Current Nano research
→ local development + RunPod GPU training (multi-surface campaign OS)

Future deployment
→ chosen independently from training infrastructure
```

**RunPod** is Nano's active primary GPU training/adaptation/CUDA backend. Routine RunPod training is **ALLOWED_WITHIN_ACTIVE_EXPERIMENT_BUDGET**. Materially costly runs require experiment-scoped authorization; confirmatory evidential runs require prereg plus experiment-scoped authorization. Local zero-cost exploratory training on Apple Silicon/CPU is **ALLOWED**. See [ACTIVE_NOW.md](../ACTIVE_NOW.md), [infrastructure/RUNPOD.md](../infrastructure/RUNPOD.md), [ACCELERATED_CAMPAIGN.md](ACCELERATED_CAMPAIGN.md).

## Integration status (this tree)

Base: `origin/master` @ `c4822b9` (PR #41 — Qwen inference + three-track harness).  
Frontier branch may add campaign v2 layers — see `integration_base_sha` in [ACTIVE_NOW.json](../ACTIVE_NOW.json).

| Component | State | Canonical location |
|-----------|-------|-------------------|
| Original 3.15M stack | **integrated** | `pretrain/`, `sft/`, `scribe/` |
| Fabric verification | **integrated** | `fabric/` |
| Wedge v1 | **integrated** | `wedge_v1/` |
| Encounter Representation v0 | **integrated** | `nanoscribe/encounter.py` |
| Evidence transport + eval | **integrated** | `nanoscribe/adapt.py`, `test_evidence_transport.py` |
| Qwen adapter + three-track harness | **integrated** | `nanoscribe/adapters.py`, `tracks.py`, `harness.py` |
| Tool calling + agent platform | **integrated** (frontier v2) | `nanoscribe/tool_*.py`, `agent_canary.py` |
| Native Nano screening | **integrated** (campaign) | `nanoscribe/native/` |
| Campaign control plane | **integrated** | `artifacts/campaign/`, `scripts/campaign_control_plane.py` |
| `nano_ai/` / H6 transfer bundles | **cross-branch** | e.g. `cursor/span-port-route-b-182e` — not in tree |
| Span-port register work | **cross-branch** | source branch + commit |
| Measurement-integrity (`9fe5b6b6`) | **cross-branch** | evaluate separately — excluded from doc-reset base |

Until selectively ported, **do not** treat `artifacts/nano_h6/` or `nano_ai/` as present locally.

---

## A — Mechanism models (microscopes)

**Scale:** ~3M–100M from-scratch (e.g. original nano-lm trunk, `scale/`, `nanoscribe/native/`)

**Purpose:** controlled experiments, architecture/objective tests, copy/retrieval/induction

**Repo (integrated):** `pretrain/`, `sft/`, `scribe/`, `trajectory/`, `nanoscribe/native/`

## B — Compact production models

**Scale:** efficient pretrained / adapted / distilled (Qwen student tracks, future Nano deploy)

**Integrated paths:** `nanoscribe/adapters.py` (serverless strong control, compact baseline)  
**Cross-branch lineage:** H6 transfer — **not yet integrated** (`artifacts/nano_h6/` when ported)

## C — Teacher / reference models

Ceilings, synthetic data, judging, distillation — frontier teacher track in [ACCELERATED_CAMPAIGN.md](ACCELERATED_CAMPAIGN.md). Not default deploy.

## Evidence pointers (integrated artifacts)

- Paper α: `papers/latex/`, `trajectory/results_*.json`
- E1: `trajectory/results_e1_utility.json`
- E4: `trajectory/results_e4_utility.json`
