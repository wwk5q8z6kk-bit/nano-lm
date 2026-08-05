# Scale program preregistration — research tier (DRAFT, blocked on owner gates)

**Status:** DRAFT preregistration. Nothing in this document authorizes compute.
Training is blocked until (a) the fineweb-edu license/commercial question in
`papers/DATA_LICENSES.md` §4 is answered, and (b) the owner approves the
budget named below. Authorized 2026-08-05 as a *draft* by the owner's
"research scale: 160M–1B" decision during the strategy review.

## 1. Question

Does genuine capability breadth — a compute-optimal pretrained base on real
web text — improve Nano's scribe-task robustness *beyond* what the cheap
fixes (adequate data volume or LoRA, which the program showed are
substitutes) already achieve at the same parameter count?

This is a capability question, not a copying-gap question. The copying gap is
already explained (training-stack-dominant; data & method substitutes;
`papers/paper2_draft.md`). The scale program tests what remains: robustness
on messier inputs, abstention quality, and instrument transfer.

## 2. Design — rung 1 (160M), gate before rung 2 (1B)

**Rung 1 model:** ~160M params in the own-stack family (RoPE, GQA, SwiGLU,
RMSNorm pre-norm, tied embeddings), matching the existing within-stack 160M
control's architecture so the comparison is clean against
`C_ADAPT_DATA_CELLS` evidence.

**Pretraining data:** `HuggingFaceFW/fineweb-edu` `sample-10BT` at pinned
revision `87f09149…` ONLY, acquired through `nano_ai/pretraining/prepare.py`
(manifest-verified, per-shard sha256, dedup + contamination digests computed
in the authorized run against `native-state-span-dev-v0`, `fresh_v1`, and all
H-cycle values). Tokenizer: retrain byte-level BPE at V sized by the
embedding-budget formula, OR reuse `pretrain/tokenizer.json` (`7a302eae…`) —
decided and frozen before launch, recorded in the manifest.

**Budget:** Chinchilla-ish D≈20N → ~3.2B tokens. Prior evidence: the 160M ×
3.2B-token control cost **$37 on an H100** (2026-07-19, peer session). Cap
rung 1 at **$150** total (pretrain + finetune + eval + margin).

**Finetune + evaluation:** the scribe finetune protocol and the held-out-value
instruments (diluted AND clean, multi-instance) are reused verbatim as the
comparison spine, plus fabric grounding.v1/v2 regression. Per
`papers/NANO_V2_AMBITION.md` (owner-set 2026-08-05), the finetune target
broadens to **general structured contract v0** — field extraction +
hierarchical notes + markdown tables (clinical as one profile) — with every
structured training pair machine-validated at generation time. New-instrument
additions are additive; the frozen clinical instruments still run unchanged
so rung 1 stays comparable to all prior evidence.

**Tokenizer (updated per ambition doc):** trained fresh for rung 1 with
reserved special tokens (`<think>`, `</think>`, structural symbols) registered
from day one, so later reasoning/structure rungs never force a retrain.

**Runtime pins (lesson of 2026-08-04/05):** provider-controllable identities
only — python/torch/CUDA/tokenizers versions, image digest, GPU model
*disclosed*; kernel and host recorded as observations, never gating.
Same-pod or checkpoint-resumable design; aggressive checkpointing; results
downloaded and dual-domain verified before any pod termination.

## 3. Preregistered decision gates (rung 1)

- **G-scale-1 (capability):** held-out-value clean aggregate improves by
  ≥10 points over the existing within-stack 160M control (66.6 ± 5.0
  value-level) under the identical finetune protocol. Below that: scale rung
  REJECTED for capability; retain as negative result.
- **G-scale-2 (no regression):** fabric grounding.v2 presented-error stays
  0.0% with zero lost-correct; abstention/absence metrics do not regress vs
  the control.
- **G-scale-3 (cost honesty):** if rung-1 spend exceeds its cap, the run
  stops at the last verified checkpoint and reports partial-budget results;
  no silent overrun.
- **Rung 2 (1B, ≈$500–700)** is authorized ONLY if G-scale-1 and G-scale-2
  both pass and the owner re-approves after seeing rung-1 numbers.

## 4. What this program may NOT claim

No clinical validity, no open-world deployment readiness, no product claims
(`STRATEGIC_RESET.md` scope stands). Improvements are claims about the
synthetic instruments and frozen evaluations named above, with limits stated.

## 5. Blocking gates (all must clear before any pod)

1. fineweb-edu license/commercial answer (`DATA_LICENSES.md` §4) — **owner**.
2. Budget approval: $150 rung-1 cap — **owner**.
3. Bounded local preparation run completes and its manifest verifies
   (`prepare.py`; fineweb-edu advanced from proposal only after gate 1).
4. Tokenizer decision frozen and recorded.
5. Provider boot-health precheck passes (see
   `artifacts/nano_h6/runops/CAMPAIGN_FINDINGS_20260805.md` resume
   conditions) — no training on a fleet that cannot boot pods.
6. H6 decision recorded first (Strategy S3: decide-then-spend).
