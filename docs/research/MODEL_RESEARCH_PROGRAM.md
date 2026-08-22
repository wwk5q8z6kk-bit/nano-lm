# Model Research Program

Three layers run through all capability programs (P1–P9).

## A — Mechanism models (microscopes)

**Scale:** ~3M–100M from-scratch (e.g. original nano-lm trunk, scale tests)

**Purpose:**

```text
controlled experiments · architecture tests · objective tests
copy/retrieval/induction · memory · calibration
```

**Repo:** `pretrain/`, `sft/`, `scribe/`, `scale/`, `stage_m/`, `trajectory/`

They need not beat 1B+ pretrained models as products.

## B — Compact production models

**Scale:** efficient pretrained / adapted / distilled (e.g. ~0.5B–1.7B transfer experiments)

**Optimize:** quality × reliability × memory × latency × privacy × cost

**Repo:** `artifacts/nano_h6/`, `nano_ai/` (when present)

Current ~1.5B results are **one frontier point**, not the permanent definition.

## C — Teacher / reference models

**Use for:** ceilings, synthetic data, filtering, distillation, judging, adversarial eval, failure discovery

**Not:** default deployed system

## Relationship to P1 scribing

P1 may use:

- mechanism models for ablations
- compact models for drafting under verification
- teachers for synthetic dialogue or judges

Routing follows [SYSTEM_ARCHITECTURE.md](../SYSTEM_ARCHITECTURE.md) — smallest sufficient solver.

## Evidence pointers

- Paper α measurement: `papers/latex/`, `trajectory/results_*.json`
- E1/E4: `trajectory/results_e1_utility.json`, `trajectory/results_e4_utility.json`
- H6 transfer: `artifacts/nano_h6/`
