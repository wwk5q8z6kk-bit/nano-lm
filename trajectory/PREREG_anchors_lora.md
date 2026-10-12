# PREREG — LoRA at the small anchors (3.15M / 10M): is the escape capacity-gated?

**Pre-registered 2026-08-23, before any anchor-LoRA cell is run.** Design, instrument,
base-matching preconditions, and decision bands are fixed here so a later execution
cannot be accused of post-hoc tuning.

## Status and scope (read before treating this as a program commitment)

This document pre-registers a **design**; it does **not** authorize or promote the run.

- `papers/paper2_draft.md:343` lists "LoRA at the anchors" as **OPTIONAL** — a capacity
  question at 3–10M, **not mechanism**.
- LoRA's causal mechanism is **E2 GATED/STOP** (`PREREG_E2_lora_universes.md`, no
  RESULT, post-KILL). Nothing here reopens E2, and no result from this cell may be
  read as mechanism. "Geometry preservation" remains banned as a product path.
- Per `docs/ACTIVE_NOW.md`, execution still requires the applicable compute
  authorization. A prereg is not an authorization.

## Question

`papers/paper2_draft.md` §"Not established": *does LoRA rescue 3–10M models, or does
escape require 160M-scale capacity?*

The factor isolations established an **interaction** at 159M: the large diluted gap
needs *both* an under-trained base *and* full-parameter adaptation; removing either
factor recovers ~10 points, and removing both floors at 4.2. Every one of those cells
sits at 159M. Whether the method escape exists at all at 3.15M/10M is untested.

## Held-fixed instrument (identical to `PREREG_anchors.md` Stage T-v2)

Reusing the Stage T-v2 instrument is what makes this a **paired** contrast — the
full-FT baselines below were measured on exactly this instrument, not inherited from
the single-instance inst0 readings (~22 / ~23).

- Eval instances: `trajectory/scribe_eval_m{0..4}.json` (v1 distribution, 100 held +
  100 seen each; seeds 20260720–20260724).
- Decoding: the anchors' **native** ChatML + greedy-argmax pipeline (verbatim
  `scribe/gate_scribe.py`), stop on `<|im_end|>`, `max_new` 64. NOT the Pythia
  HF-generate path.
- Field parser: `^CC: … | DUR: … | SEV: … | MED: … | ALG: …$`.
- Tokenizer: `sft/tokenizer.json` (vocab 4098).
- Metrics per cell: `gap_mean` = mean over m0–m4, `gap_sd` = SD (ddof=1), reported
  **diluted and clean**, both as co-primary (see decision rule).
- Device: MPS primary, matching the anchors' native gate device.

## Frozen full-FT baselines (already measured; do not re-run)

| Anchor | Params | Artifact | Diluted gap_mean ± SD | Clean gap_mean ± SD |
|--------|--------|----------|-----------------------|---------------------|
| nano  | 3.15M | `results_anchors_v2_nano.json`  | **18.30 ± 1.34** | **87.25 ± 2.71** |
| scale | 10M   | `results_anchors_v2_scale.json` | **18.72 ± 1.51** | **79.50 ± 2.06** |

159M reference cells (`PREREG_ownstack_160m.md`): full-FT 16.9 ± 1.7 diluted →
LoRA 7.1 ± 1.2. The 159M LoRA effect is therefore **Δ ≈ 9.8 diluted points**.

## Base-matching precondition (MANDATORY — a cell that fails this is uninterpretable)

`checkpoints/anchors/` holds only **already-scribe-finetuned** weights
(`nano_v01_scribe.pt`, `scale10m_scribe.pt`). LoRA-ing on top of those would adapt an
already-adapted model and the resulting JSON would not reveal the error. Each anchor's
LoRA cell MUST start from the **same base its full-FT baseline started from**:

| Anchor | Required LoRA base | Lineage | Release asset | Size (bytes) | sha256 |
|--------|--------------------|---------|---------------|--------------|--------|
| nano  | `dpo.pt`                | SFT+DPO (chat) | v0.1 | 12,609,731  | **UNRECORDED — must be captured on first fetch** |
| scale | `scale10m_pretrain.pt`  | raw pretrain   | v0.1 | 120,096,379 | `892180f02d09cacd2d129ba041dcbcca7635594bc98d8402203af64afc2fc88d` |

Both verified retrievable 2026-08-23 (HTTP 200) from
`https://github.com/wwk5q8z6kk-bit/nano-lm/releases/download/v0.1/<asset>`.

The `scale` hash is the one already pinned by C-3 (`results_c3_10m.json`,
`artifacts/durable_raw/MANIFEST.json:225`). **`dpo.pt` has no recorded hash anywhere in
the repo** — closing that gap is a required output of the first run, not an optional
extra (cf. the `REPRODUCIBILITY_LIMITATION` note at `MANIFEST.json:24`).

Preconditions, all of which must appear in the results JSON:
1. `base_checkpoint_sha256` recorded per cell, matching the table (nano: recorded and
   frozen on first fetch).
2. `base_lineage` field asserting `dpo` / `raw_pretrain`.
3. A **base-is-not-finetuned assertion**: the untouched base scored on inst0 must NOT
   reproduce the scribe fingerprint (nano scribe.pt reads parse ≈98%, recall ≈81%).
   A base scoring near that is the wrong checkpoint — abort the cell.

## LoRA configuration (fixed)

Matching the 159M LoRA cell so the contrast is method-at-scale, not method-variant:
r=16, α=32, dropout 0.0, targets = all attention + MLP projections in the own-stack
family, FT_SEED=0, otherwise the anchors' own scribe finetune recipe and data
(seed-11 v2) unchanged.

## Decision rule (fixed before any result)

Evaluated **per anchor, against that anchor's own baseline above** — deliberately not
against the 159M effect size. Define Δ_diluted = baseline_diluted − lora_diluted.

The anchors' clean gaps (87.25 / 79.50) are ~3× the 159M clean gap (~29.6), so the
anchors and the 159M cell are not the same object despite adjacent diluted numbers.
Diluted gap is a ratio-like quantity over **parsed** fields; near clean saturation it
can move for parse reasons alone. Clean is therefore **co-primary**, not a caveat:

- **METHOD-GENERAL** — Δ_diluted ≥ 7.0 **and** clean gap drops ≥ 15 points.
  LoRA rescue is not capacity-gated; the escape exists at 3–10M.
- **CAPACITY-GATED** — Δ_diluted ≤ 3.0. The 159M method escape does not extend down;
  escape requires ≥159M-scale capacity.
- **GRADED / ARTIFACT-SUSPECT** — 3.0 < Δ_diluted < 7.0, **or** Δ_diluted ≥ 7.0 with
  clean approximately unchanged. Report the band; force nothing. A diluted drop with
  clean still ~85 is a parse/dilution artifact, **not** a rescue, and must be reported
  as such.

Threshold justification (small-n honest): with gap_sd ≈ 1.34–1.51 over n=5 instances,
SE ≈ sd/√5 ≈ 0.60–0.67. The 3.0 boundary is ≈4.5 SE and the 7.0 boundary ≈10 SE — both
resolvable at this n. 7.0 ≈ 70% and 3.0 ≈ 30% of the measured 159M effect (9.8), so the
bands ask "does most / little of the 159M effect survive down-scale" without assuming
the anchors have the same headroom.

**Seed discipline.** Cells are single-seed (FT_SEED=0) as run. Per the §5.2
seed-bound lesson and the corner's Q2 precedent, if **either** anchor lands in
METHOD-GENERAL, a FT_SEED=1 duplicate of that cell is required **before** the claim
stands; |Δseed| > 4 demotes the cell to an interval.

## Compute estimate

LoRA finetunes on 3.15M/10M bases: ~1.5 h total for both cells plus eval (progress-log
estimate), the dominant cost being the 120 MB scale base fetch. Small enough to be
routine within an active experiment budget — but see Status above: this prereg does not
authorize the spend.

## What this does and does NOT resolve

- **Resolves:** whether the LoRA escape measured at 159M exists at 3–10M, i.e. whether
  the interaction account is capacity-gated.
- **Does NOT resolve:** LoRA's mechanism (E2 GATED/STOP — no result here may be read as
  mechanism); the breadth/tokenizer residual bundle; the shared clean residual (~15–18
  at 159M, ~80–87 at the anchors), which no cell in this design addresses.
- **Does NOT license** any claim about the anchors' clean-metric behaviour beyond the
  co-primary readout defined above.

## Status

**Not executed.** No artifacts exist. On execution, write
`trajectory/results_anchors_lora_{nano,scale}_seed0.json` and add a RESULT section
below this line without rewriting the protocol above.
