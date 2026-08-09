# Preregistration — P2 falsifier, route (b) span port

**Frozen 2026-08-09, before any span-bearing LoRA is trained or scored on
Qwen2.5-1.5B.** Thresholds below will not move after results.

Authority: `papers/DECISION_SPAN_PORT.md`, `papers/RESULT_TRANSFER_CURVE_BALANCED.md` §5,
`papers/REVIEW_20260809_TIP_CHANGES.md`.

## Question

After LoRA on a **span-bearing** target, can Qwen2.5-1.5B emit span text that
`_locate_unique_patient_span` relocates uniquely in the source transcript?

## Why it decides something

State-only transfer is established at ≥1.5B. Nano's product is state **plus
grounded span**. Route (b) is cheap iff generated span text is exact (or nearly
exact) source text. High `no_match_rate` falsifies route (b) and forces route
(a) or constrained decoding.

## Design

**Held constant**

- Base: `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0) — the balanced-curve clear
  from a family that also cleared at 1.7B (SmolLM2); single-seed screening, not
  a ranking.
- Fit partition only (`artifacts/nano_h5/data`), leak guard unchanged.
- LoRA recipe identical to the transfer curve: 8 layers, batch 4, 400 iters,
  max_seq 768, lr default from mlx_lm, seed `20260806`.
- Balanced label mix (downsample to minority count) — without this the curve
  taught us the run is uninterpretable.
- Development docs and denial arms identical to the cross-model / curve probes.
- Relocation function: unmodified
  `nano_ai.adapters.state_span._locate_unique_patient_span`.
- Control block: balanced supported/missing, worst-class recall ≥ 0.20 or the
  run is uninterpretable.

**Varied / new**

- Training target format: `LABEL: "span text"` (and bare `NOT_MENTIONED` when
  there is no gold span).
- Prompt must request that format (fixing the v2 builder defect where the
  prompt still said "exactly one word").

**Not in this run**

- Multi-seed confirmation.
- SmolLM2-1.7B span port (deferred unless Qwen fails).
- CUAD adapter (P4).
- Pointer-head route (a).

## Primary metrics (reported separately)

Denominators are fixed **before** scoring:

| metric | denominator | definition |
|---|---|---|
| `no_match_rate` | count of emitted non-empty span strings | relocate raises "not exact text inside a Patient turn" |
| `ambiguous_rate` | same | relocate raises "ambiguous across Patient turns" |
| `relocated_rate` | same | relocate succeeds |
| `missing_span_rate` | count of fields whose gold has ≥1 span | model emitted a bare label with no span text |
| `state_held_out_mean` | denial-arm field questions | label matches gold `DENIED` (same as curve) |

A span string that fails to parse out of the generation does not enter the
relocation denominator; it increments `unparsed_span_fields` instead.

## Preregistered bars

> **ROUTE (b) SUFFICIENT** iff the run is interpretable **and**
> `no_match_rate ≤ 0.10`.

> **ROUTE (b) FALSIFIED** iff the run is interpretable **and**
> `no_match_rate > 0.25`.

Between 0.10 and 0.25 is a gray zone: report rates, do not flip the decision;
cheapest follow-up is constrained decoding or quote-copy instructions, not an
immediate route-(a) rewrite.

`ambiguous_rate` is reported but does not alone falsify route (b) — both routes
hit ambiguity, at different layers. `missing_span_rate` is diagnostic: if the
model learned labels but not spans, the data/prompt fix failed.

State held-out mean is secondary here; the curve already cleared 75% for this
base on state-only data. A collapse on state after adding spans means the joint
target broke the state skill and the run does not answer the span question.

## Status: SCREENING

One seed. Coarse accept / falsify / gray only. No family ranking.

## Asks (owner / process — defaults used if unanswered)

1. **Base choice.** Default: Qwen2.5-1.5B. Alternative: SmolLM2-1.7B. Both
   cleared the state bar; evidence does not rank them.
2. **Product boundary at ~1.5B.** Engineering default: proceed; device-target
   renegotiation remains an owner product decision (`FORWARD_PLAN` §6).
3. **Gray-zone follow-up.** Default: try quote-forcing / constrained decode
   before route (a).
4. **CUAD timing.** Default: after this falsifier returns ACCEPT or gray with
   a cheap mitigation; do not block the span port on the real-corpus adapter.

## Falsification summary

| outcome | consequence |
|---|---|
| uninterpretable control | no decision; repair data/prompt and rerun |
| `no_match_rate ≤ 0.10` | route (b) proceeds to P3 design |
| `0.10 < no_match_rate ≤ 0.25` | gray; cheap mitigation before architecture change |
| `no_match_rate > 0.25` | flip toward route (a) or grammar-constrained generation |

## What acceptance does not license

Not multi-seed confirmation, not CUAD/real-document transfer, not promotion of
any LoRA adapter into Nano's frozen inference surface, and not a claim that
joint state+span training preserves the 99.7% state held-out score.
