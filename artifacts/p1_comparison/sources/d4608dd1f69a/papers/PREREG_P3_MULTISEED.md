# Preregistration — P3: multi-seed confirmation of the pretrained direction

**Frozen 2026-08-10, before any P3 seed is trained.** Floors below will not
move after results.

Authority: `papers/FORWARD_PLAN_20260806.md` §5 P3 (requirements: ≥3 seeds,
`surface_robust_accuracy` = min over arm means as primary gate, absolute
floors over fixed gold denominators, train/eval lexicon split hashed before
use). Design inputs: `papers/PLAN_20260809_P3_SCOPING.md` (Wilson interval,
register clustering, gate structure, gold-agreement baseline).

## Question

Does the accepted configuration — Qwen2.5-1.5B-Instruct + balanced
span-bearing LoRA + normalization snap at the proposal layer — survive
confirmatory multi-seed evaluation under worst-case (min-over-arms,
min-over-seeds) gates?

## Why it decides something

Everything in the span-port series is single-seed screening, and
`PLAN_20260805_SURFACE_ROBUSTNESS.md` measured **Kendall τ = 0.00 between
seeds** on which novel phrasings a tuned model handles — arm-level results
are expected to move across seeds. P3 converts "accepted (screening)" into
either a seed-robust claim or a documented fragility, before any P4
real-document work builds on it.

## Design

**Held constant** — everything in `PREREG_SPAN_PORT_B_SNAP.md`: base
(Qwen/Qwen2.5-1.5B-Instruct), data (`data_spans_balanced`, provenance
checks enforced by the runner), LoRA recipe (8 layers, batch 4, 400 iters,
max_seq 768), baseline prompt, snap-fallback pipeline, unmodified binder,
balanced control gate, greedy decode, dev bundle, denial arms.

**Varied** — LoRA training seed only:

- Seed A: **20260806** — the accepted P2 adapter. Its evaluation artifacts
  are already on disk (`probe_Qwen2.5-1.5B_snap.json`,
  `gold_agreement.json`) and are consumed as-is; they are restated here as
  known priors, not re-derived: min-arm state accuracy 0.7667, post-snap
  pooled no_match 0.0913, disjoint_rate 0.0134.
- Seed B: **20260810** (named here, not yet trained).
- Seed C: **20260811** (named here, not yet trained).
- Replacement policy: if a seed's balanced control is uninterpretable
  (worst-class recall < 0.20), train **20260812** as a one-time
  replacement and report the substitution; a second uninterpretable seed
  ends P3 as "uninterpretable" with no gate verdicts.

**Lexicon split** — frozen as currently shipped (4 train phrasings, 12
external), option (a) of the scoping doc: telegraphic register stays
external-only this cycle; zero-shot register transfer is what EXTERNAL
measures. `sha256(repr((_TRAIN_DENIALS, _EXTERNAL_DENIALS))) =
bb801c1f02008a067c61ea819770e83ac1944854193d2ab63df0b1ed3f9b401d`.
Any lexicon change voids this prereg.

**Per-seed evaluation** — the two existing, committed evaluators run
unchanged per adapter: `run_span_port_probe_snap.py` (state per arm,
control, snap relocation; baseline-identity check applies to seed A only —
for seeds B/C the raw counts legitimately differ) and
`evaluate_span_gold_agreement.py` (gold agreement + disjoint). Their
independent regeneration of the same greedy outputs must cross-match
within each seed (relocated / not-relocated counts), as they did for
seed A (448/45); a mismatch voids that seed as a wiring defect.

## Preregistered gates (all floors fixed now)

Denominators: per-arm n = 30 fields (fixed pool × arm grid); relocation
rates over emitted spans; agreement over relocated spans; gold spans as
frozen in the dev bundle.

| gate | metric | floor | justification |
|---|---|---|---|
| G1 interpretability | balanced control worst-class recall, per seed | ≥ 0.20 | unchanged house gate; runs first |
| G2 state (primary) | `surface_robust_accuracy` = min over all 17 arm means, min over seeds | **≥ 0.75** | continuity with the frozen transfer-curve bar, applied at worst-case granularity; seed A passes at 0.7667 |
| G3 grounding contract | `disjoint_rate` (wrong-evidence assertions), max over seeds | **≤ 0.03** | the one class that asserts instead of abstaining; seed A baseline 0.0134, ~2.2× headroom for seed variation |
| G4 span replication | post-snap pooled `no_match_rate`, max over seeds | **≤ 0.10** | the accepted B-series bar, now demanded of every seed |

Reported without floors (explicitly deferred to P4 gate design, where the
product coverage requirement lives): per-arm relocated/abstention rates,
per-arm disjoint counts, exact-vs-overlapping agreement composition.

## Preregistered outcomes

| outcome | condition | consequence |
|---|---|---|
| **state-confirmed** | G1 + G2 pass | the pretrained direction carries surface-robust state at 1.5B; FORWARD_PLAN confidence row "small pretrained base suffices" moves from *unknown* toward *supported (dev-synthetic)* |
| **grounding-clean** | G3 passes | route (b) + snap upholds the contract across seeds; P4 planning may proceed |
| **span-replicated** | G4 passes | screening acceptance replicates; no register work required before P4 |
| state-fragile | G2 fails | stop; per-arm × per-seed diagnosis before any P4 work |
| grounding-dirty | G3 fails | stop; the cross-field attribution class is load-bearing — measure and fix before P4 |
| span-fragile | G4 fails (others pass) | register-coverage experiment (scoping §5 option (b), re-hashed split) or constrained decode, before P4 dogfood |

Overall: **proceed to P4 planning iff state-confirmed AND grounding-clean.**
Span-fragile alone does not block P4 *planning* but blocks P4 *execution*.

## Status: CONFIRMATORY (3 seeds, screening-depth per seed)

Three seeds bound seed variance coarsely; they do not support distributional
claims. Eval decode is greedy — all variance is training-seed variance.

## What acceptance does not license

No claim beyond the synthetic dev bundle and this lexicon (P4 exists for
that). No promotion of any adapter into Nano's frozen inference surface.
No claim about registers absent from both lexicon halves. No implication
that 1.5B is the *smallest* sufficient base (the balanced curve's open
question below 1.5B stays open).
