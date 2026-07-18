# Stage T-v2 — re-scoring the nano/scale anchors on the multi-instance instrument

**Pre-registered 2026-07-17, before any m0–m4 anchor number was observed.**

## Motivation

In the Stage T ladder (paper §6.1) the two own-stack anchors (nano 3.15M, scale
10M) are reported **single-instance**, while the Pythia rungs (160M/410M/1B) are
reported on the **powered multi-instance instrument** (5 fresh instances × 100 held
+ 100 seen, `scribe_eval_m0..m4.json`; gap_mean ± across-instance SD). The two ends
of the ladder are therefore not measured with the same instrument, so the headline
comparison (~22 → 3.5) mixes instruments at the two ends.

Worse, provenance analysis of the anchor number shows a *second* inconsistency. The
paper's nano "~22" traces to `scribe/gate_scribe_v2.log` (released `scribe.pt` = v2
model, recall 81 / halluc 11.5, held 72 / seen 94 → gap 22) scored on the v1 eval
(`scribe_eval.json` = instance 0; `build_scribe_data_v2.py` line 5: "THE EVAL SET IS
NOT REGENERATED — v1 reused byte-identical"). The stale `gate_scribe.log` (gap 17,
recall 74 / halluc 14) is the **earlier v1 checkpoint** before `scribe.pt` was
overwritten — not the released model. So the released anchors have a clean
single-instance reading on inst0, but no multi-instance reading. This stage adds it.

This is a **re-scoring, not a re-training** (cf. PREREG_Tv2 for the Pythia rungs):
the anchor checkpoints are frozen v0.1 release assets; only the eval instrument
changes (inst0 single → m0–m4 multi-instance), using the anchors' own native scoring
pipeline. The `git`-tracked v1 single-instance results (Stage S's pre-registered
gap 23; the nano gate logs) are **preserved unchanged** as the original record; this
stage adds a consistent-instrument reading, it does not overwrite them.

## Frozen models (v0.1 release assets)

- nano `scribe.pt` — scribe **v2** finetune, 3.15M params (d=192 L=6 H=6 KV=2 hd=32
  ff=512 V=4098 S=512). sha256 `0e4f348eea00c660236cfd9e5bc2d9a71274adfc4d738db6f664664c9a06725b`.
- scale `scale10m_scribe.pt` — scribe finetune, ~10M params (d=320 L=8 H=8 KV=2
  hd=40 ff=864 V=4098 S=512). sha256 `f5aca5f04bd1045cc158d46a27b84024bb94baa349ed330933631c8b8d5acf0d`
  (verified against scale/AUDIT.md).

Tokenizer: `sft/tokenizer.json` (vocab 4098, `<|im_start|>`=4096, `<|im_end|>`=4097).

## Instrument (fixed; identical distribution to the Pythia multi-instance rungs)

- Eval instances: inst0 (`scribe/scribe_eval.json`, v1, 40 dlg, cross-check) plus the
  five fresh instances `trajectory/scribe_eval_m{0..4}.json` (v1 distribution, 100
  held + 100 seen each; seeds 20260720–20260724 per PREREG_Tv2).
- Decoding: the anchors' **native** ChatML + greedy-argmax pipeline (verbatim
  `scribe/gate_scribe.py`): prompt = `<|im_start|>user\n{content}<|im_end|>`
  `<|im_start|>assistant\n`; generate token-by-token, argmax, stop on `<|im_end|>`,
  max_new 64. This is NOT the Pythia HF-generate path — the own-stack models require
  their ChatML format. Same field parser `^CC: … | DUR: … | SEV: … | MED: … | ALG: …$`.
- Gap: field-level `seen_value_recall − held_value_recall` (×100), counting fields of
  **parsed** dialogues only (identical convention to gate_scribe.py and the v2 Pythia
  scorer). Per-model: `gap_mean` = mean over m0–m4, `gap_sd` = SD (ddof=1).
- Device: MPS primary (the nano gate's native device). A CPU cross-run is triggered
  **only** if the nano inst0 determinism check below misses its target by > 2 pts.

## Cross-check targets (fixed before m0–m4 observed)

Determinism / pipeline validation — must pass before the m0–m4 means are trusted:

1. **nano `scribe.pt` on inst0** → reproduce `gate_scribe_v2.log`: parse 39/40 ≈ 98%,
   recall ≈ 81%, halluc ≈ 11.5%, held 68/95 ≈ 72%, seen 94/100 = 94%, **gap ≈ 22**.
   Same MPS device → expect exact or ±1 field. This validates arch + checkpoint +
   pipeline for the own-stack path.
2. **scale `scale10m_scribe.pt` on inst0** → reproduce Stage S (scale/AUDIT.md):
   parse 100%, recall ≈ 88%, halluc ≈ 7.5%, held ≈ 77%, seen = 100%, **gap ≈ 23**.
   Original device was CUDA T4; local is MPS → expect close, **flag device drift** if
   the fingerprint (parse/recall) diverges materially.

Gate: if inst0 checks reproduce (nano within ±2 pts of 22; scale parse 100% and gap
within a few pts of 23), proceed to report the m0–m4 means. Otherwise stop and
diagnose (do not tune).

## Contamination check (direction-aware, unchanged from PREREG_Tv2)

Flag ONLY if inst0 gap is LOWER than the fresh-instance mean by > 2 SD (memorization
would make the public instance EASIER). A higher inst0 gap is a hard draw, not
contamination.

## What is NOT re-opened

- No re-training; the anchor checkpoints are frozen v0.1 artifacts.
- The nano-stack vs Pythia-stack confound is unchanged; re-scoring does not separate
  scale from stack (paper §7).
- Stage S's pre-registered single-instance gap 23 and the nano single-instance logs
  are retained as the original record. This stage reports the anchors on the
  consistent multi-instance instrument alongside them.

## Outputs

`trajectory/results_anchors_v2_nano.json`, `trajectory/results_anchors_v2_scale10m.json`
(per-instance held/seen recall + gap, gap_mean, gap_sd, 2SD band, inst0 cross-check,
contamination flag, device, library versions).

## Status

Pre-registered. One pass, no tuning. Then update paper §6.1 and FINDINGS.md.
