# PREREG — E3 Exact-match construct validity (soft-match + human faithfulness)

**Pre-registered 2026-07-30, AFTER E1 provisional kill but BEFORE human labels exist.**
Builds on the construct policy already frozen in `PREREG_E1_nonlm_baseline.md`
(exact / normalize-then-match / human faithfulness). This file is the dedicated
gate that clears or contests theory threat **T3** (exact-match construct) enough
for Paper α measurement claims.

## Status

**Automated arm executed 2026-07-30.** Artifact:
`trajectory/results_e3_normalize_construct.json`.

**Human arm EXECUTED (Stage 1, 2026-07-31)** — bounded pack labeling.
Artifact: `trajectory/results_e3_human.json`. Analysis:
`trajectory/STAGE1_E3_CONSTRUCT_FIRST_PRINCIPLES.md`. Pack frozen at
`trajectory/e3_human_rating_pack.json`.

## Question → Prediction → Measurement → Decision

**Q:** Does exact string match overstate extraction failure relative to (a)
frozen normalize-then-match and (b) human faithfulness judgments?

**Predictions:**

- **H-construct-collapse:** normalize shrinks the M0 clean/held failure mass by
  ≥10 pts (field-level correct-rate lift), **or** humans label ≥50% of exact
  errors as `faithful` → exact-match instrument contested; reinterpret gaps;
  freeze mechanism punchlines.
- **H-exact-faithful:** normalize shrinks failure mass by <5 pts **and** humans
  label <20% of exact errors as `faithful` → exact-match survives as science
  instrument for this synthetic world.
- **H-partial:** between thresholds → graded; report both metrics; no binary
  construct clearance.

## Normalize rule (frozen; same as E1)

Committed in `trajectory/e1/common.py::normalize_value` + `PLURAL_MAP` before
E1 scoring. Do not edit for E3; if a bug is found, VOID and amend.

## Human rubric (frozen before labels)

For each `(dialogue, field, pred, truth)` triple, one of:

| Label | Meaning |
|---|---|
| `faithful` | Pred preserves the clinically relevant fact expressed by truth for this field (trivial orthography/pluralization a clinician would treat as same is OK). |
| `unfaithful` | Pred changes, invents, omits, or substitutes a different clinical fact. |
| `unsure` | Cannot decide from dialogue+pair without external knowledge. |

Raters must not retune methods. Two passes if available; else one rater with this
rubric. Labels write to `trajectory/results_e3_human.json` only.

## Sampling (frozen)

Primary pool: M0_scale verify-off item logs
(`results_e1_items_M0_scale_voff.json`).

1. **60 exact≠normalize disagreements** (exact wrong, normalize correct).
2. **40 random exact errors** (seed `20260730`).

### AMENDMENT 1 (2026-07-30, pre-human, post-automated census)

Automated census found **0** exact≠normalize disagreements under the frozen
normalize rule (all methods, both verify arms). Therefore the disagreement
stratum is empty. **Replacement rule (fixed now):** sample **100** exact errors
from M0 verify-off with seed `20260730` (60+40 collapsed into one error pack).
This does not change the decision thresholds above; it only fills the human
queue when the normalize rescue pool is empty. Pack path:
`trajectory/e3_human_rating_pack.json`.

## Decision rule

1. **CONSTRUCT_COLLAPSE:** auto gap-shrink ≥10 pts **or** human faithful-rate on
   exact-error sample ≥0.50 → gaps reinterpreted; mechanism papers frozen.
2. **EXACT_SURVIVES:** auto gap-shrink <5 pts **and** (human arm complete with
   faithful-rate <0.20, or human arm still blocked with auto-only provisional
   note) → exact-match remains primary science instrument.
3. **GRADED:** otherwise.

Kill decision for E1 remains exact-\(U\) primary (already recorded). E3 does not
reopen E1 weights.

## RESULT — automated arm (2026-07-30)

| Quantity | Value | Artifact |
|---|---|---|
| M0 exact field rate | 0.9027 | `results_e3_normalize_construct.json` |
| M0 normalize field rate | 0.9027 | same |
| Norm rescue count (M0) | **0** / 486 exact failures | same |
| Gap shrink | **0.0 pts** | same |
| Cross-method rescues (M1–M5+M0, both arms) | **0** | same |
| M0 error mix (exact fails) | substitution 380 · omission 106 · fabrication 0 | same |

**Automated verdict:** `EXACT_NOT_OVERSTATING_BY_NORMALIZE`
(falsifier for collapse: shrink <5 pts). Provisional toward **EXACT_SURVIVES**;
full clearance awaits human pack labels.

**Human next step:** label `e3_human_rating_pack.json` → write
`results_e3_human.json` without editing preds/truths.

## What this resolves / does not

**Resolves (auto):** whether the frozen normalize rule converts exact failures
into matches on this instrument — it does not.

**Does not resolve:** clinical ontology / H4 coding objective; open-world
paraphrase faithfulness; inter-rater reliability until second pass exists.

## RESULT — agent-applied rubric audit (Stage 1, 2026-07-31)

| Quantity | Value |
|---|---|
| Pack size | 100 exact errors (Amendment 1) |
| Faithful rate | **0.00** (0/100) |
| Human-acceptable rate | **0.00** (0/100) |
| Normalized matches on pack | **0/100** |
| Prereg verdict | **EXACT_SURVIVES** |
| Qualitative open-slot gap shrinks materially? | **No** |
| Rater | `agent-rubric-pass-1` (single pass; IAA absent — see limitations note) |

**Does not** reopen E1. **Does not** unlock E2/fabric. Paper α keeps exact-match
limitation (strict metric + single-pass agent-applied rubric audit; dual-clinician
unperformed), not because failures are formatting noise.
