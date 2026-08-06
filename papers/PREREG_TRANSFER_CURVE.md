# Preregistration — P0, the transfer curve

**Frozen 2026-08-06, before any base smaller than 3B has been trained or
measured.** Thresholds below are fixed now and will not move after results.

## Question

What is the **smallest pretrained base** that recovers most of the transfer the
LoRA control demonstrated at 3B?

## Why it decides something

`RESULT_LORA_CONTROL.md`: same `fit.jsonl`, same task, same arms —
Nano-13M-from-scratch reaches **60.0%** held-out; Llama-3.2-3B + LoRA reaches
**90.3%**. Everything between is unmeasured. If the curve saturates small, the
local-first thesis survives and Nano changes only its initialisation. If it needs
3B, the device target must be renegotiated against the privacy claim.

## Design

**Held constant:** `artifacts/nano_h5/data` fit partition (leak-guarded);
the 3-way label task and prompt; LoRA config (8 layers, batch 4, 400 iters,
max_seq 768, lr 1e-5, seed 20260806); the 17 denial arms over the identical
sealed development documents; 15 documents per arm; the balanced control block.

**Varied:** the pretrained base only.

**Bases** (all Apache-2.0 except where noted, all 4-bit MLX, all local):
- `SmolLM2-135M-Instruct` (Apache-2.0)
- `SmolLM2-360M-Instruct` (Apache-2.0)
- `Qwen2.5-0.5B-Instruct` (Apache-2.0)
- `SmolLM2-1.7B-Instruct` (Apache-2.0)
- `Llama-3.2-3B-Instruct` (Llama 3.2 Community License) — **already measured**
- Nano-13M from scratch — **already measured**, the floor

## Primary metric and preregistered bar

`nano_held_out_arms.mean` — mean over the 12 external denial arms, each a mean
over its documents.

> **BAR: a base "recovers most of the transfer" iff held-out mean ≥ 75.0%.**

Derivation, fixed now: the observed gap is 60.0% (Nano) → 90.3% (3B) = 30.3
points. Half that gap above the floor is 60.0 + 15.15 = **75.15%**, rounded down
to **75.0%**. Chosen before seeing any intermediate point, and deliberately not
set at the 3B's own number — the question is *sufficiency*, not parity.

## Interpretation gate (non-negotiable)

For each base, the **balanced control block runs first**. If worst per-class
recall < 0.20, that base's arm accuracies are recorded as **uninterpretable**
and excluded from the curve. A model that learns "always answer DENIED" would
otherwise post a perfect held-out score.

## Status: SCREENING, not confirmatory

**One seed per base.** `RESULT_SURFACE_HARNESS_RUN1.md` §5 measured 31.5 points
of per-arm seed instability at Kendall τ = 0.00, so single-seed *per-arm*
numbers are noise. The aggregate over 12 arms is more stable, but no
multi-seed evidence supports that for these bases either.

Therefore this experiment may support only **coarse** conclusions — "the curve
has saturated by size X", "base Y clears the bar" — and may **not** support fine
rankings between adjacent bases. The chosen base gets a **confirmatory
multi-seed rerun** before P3, and that rerun is where any claim hardens.

## Falsification

- If **no** base below 3B clears 75.0%, the local-first thesis is in tension with
  the accuracy requirement and the product's device target becomes an owner
  decision rather than an engineering one.
- If the **135M** base clears 75.0%, the finding is that very little pretraining
  suffices, and Nano's redesign is cheap.
- If control blocks fail broadly at small sizes, the 3-way task itself is beyond
  those models and the curve says nothing about transfer.

## What acceptance does not license

Not a claim that any base can carry Nano's **span** contract. Every point here
is **state-only**; the tuned model emits a label and no span. The span port is
`FORWARD_PLAN_20260806.md` H-2 / P2 and is sequenced before P3 for that reason.
