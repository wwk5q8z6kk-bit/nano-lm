# PREREG — E2 LoRA universe discrimination (U1–U4)

**Pre-registered 2026-07-30, AFTER E1 kill and E3 automated arm, BEFORE any
LoRA-mechanism intervention runs.** Ban on “geometry preservation” / pathway
language remains until this gate returns a non-VOID verdict.

## Status

**GATED / STOP (post-E1 KILL).** Design remains frozen. No RESULT artifact.
No `trajectory/results_e2_*.json` exists. Do **not** claim RUNNING publicly.

Historical note (operational, superseded): a U3 early-stop pod
(`e2-u3-earlystop` / `kuq4gy63yuaeke`) was started then **terminated 2026-07-31**
under the post-KILL freeze (`papers/EMPIRICAL_FOUNDATION.md` §E2 status). Script
`trajectory/e2/run_u3_earlystop.py` is present for a future written re-scope only —
**not authorized to execute** without owner re-scope that names which ledger row
the run could change.

## Why this gate still matters after E1 KILL

E1 demoted the generative-LM *product* frame. Measurement Paper α still cites
adaptation×data interactions (LoRA / Chinchilla / corner). Those behavioral
facts stay; mechanistic gloss (geometry vs early-stop vs module) does **not**
until U1–U4 are separated.

## Question → Predictions

**Q:** When LoRA (or matched low-rank adaptation) closes or shrinks the held-out
gap relative to full-FT on the same base, which competing universe explains the
delta?

| ID | Universe | Intervention sketch | Distinctive prediction |
|---|---|---|---|
| U1 | Geometry / subspace preservation | Match effective rank & trainability; ablate random vs pretrained subspace | Only pretrained-aligned adapters recover copy |
| U2 | Optimization ease | Match wall-clock / step budget; full-FT with LoRA-matched LR schedule | Full-FT catches LoRA when optimized equally carefully |
| U3 | Early-stop / implicit regularization | Early-stop full-FT on held-gap plateau; continue LoRA past | Gap delta vanishes under matched early-stop |
| U4 | Module / site of adaptation | LoRA on attn-only vs MLP-only vs all (frozen targets list) | Site, not rank, drives recovery |

**H-geometry (U1):** U1 intervention unique; U2–U4 fail to erase LoRA advantage.
**H-early-stop (U3):** matched early-stop erases LoRA vs full-FT gap delta.
**H-ease (U2):** matched optimization erases delta without early-stop story.
**H-module (U4):** site ablation dominates rank.
**H-inconclusive:** ≥2 universes survive → keep behavioral claim only; no mechanism
sentence in Paper α.

## Minimum runnable cell (when unblocked)

Venue: CUDA T4/A10 or better. Artifacts required before start:

1. Own-stack 160M pretrained base (`checkpoints/chinchilla-160m/ownstack160m_pretrain.pt`
   present locally) **or** Pythia-160M HF base.
2. Frozen scribe finetune recipe from `kaggle_ownstack_160m_lora.py` /
   `kaggle_arm1.py` (seeds, steps, LR committed).
3. Eval: multi-instance m0–m4 exact scorer (same as E1/anchors).

**Not sufficient:** re-reading existing `results_*_lora.json` gaps — those are
behavioral, not universe-discriminating.

## Decision rule (fixed now)

On diluted + clean held gap (mean±SD over m0–m4):

1. **SUPPORT universe U\*:** only that universe’s intervention moves gap by
   ≥5 pts in the predicted direction while the other three move <2 pts.
2. **REFUTE U1 geometry language:** if U2 or U3 alone erases the LoRA advantage.
3. **UNRESOLVED:** power fail, OOM, or multi-universe survival → report as such;
   Paper α keeps “adaptation regime matters” without mechanism noun.

## Blockers (exact)

| Blocker | Detail |
|---|---|
| ~~Missing official M0 adapters~~ | E1 RunPod produced Pythia + ownstack LoRA scores; ownstack pretrain mounted for U3 |
| ~~No CUDA~~ | RunPod RTX 3090 (pod `kuq4gy63yuaeke`) |
| ~~peft~~ | Installed on pod (`peft==0.12.0`) |
| Remaining | Full U1–U4 grid still sequential; U3 first per prereg |

**GPU step:** none. U3 early-stop pod was terminated; no RESULT. Do not resume without written re-scope.

## Honest-reporting rule

Single primary pass per universe cell. No post-hoc target-module fishing.
Failures to fit (OOM) = VOID for that cell with reason, not silent drop.
