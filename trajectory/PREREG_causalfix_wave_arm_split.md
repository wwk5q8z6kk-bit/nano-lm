# PREREG — is the DENIED/ASSERTED arm split an objective effect or seed noise?

**Pre-registered 2026-08-25, 02:00 into the causalfix wave, with the information
barrier intact.** At time of writing exactly two of nine runs are observed:
`reval30_decoder_control_s0` and `reval30_evidence_bottleneck_s0`.
`reval30_span_port_s0` and all six seed-1/seed-2 runs are **unobserved**. This
document is written before they land so the decision rule cannot be fitted to
them.

## What has been observed (the whole basis for this prereg)

| run | coverage | assertion_state_correct | emitted on covered atoms |
|---|---|---|---|
| `decoder_control_s0` (constrained) | 6/150 | **0/6** | `DENIED: "migraine"` |
| `evidence_bottleneck_s0` (constrained) | 6/150 | **6/6** | `ASSERTED: "migraine"` |

Both unconstrained cells: coverage 0/150, degenerate repeated-character output.

Nothing else has been looked at. `span_port` has no registered prediction here
because no `span_port` result has been seen; it is evaluated by the same rule.

## Question

Does the objective factor (now genuinely distinct per `c98e4ad`) produce the
assertion-label difference, or is a single-seed 0/6-vs-6/6 split seed noise?

## Primary metric

`assertion_state_correct`, pooled per arm across seeds {0,1,2}, **conditioned on
covered atoms only** (an abstained atom is not an assertion decision).

Reported as pooled k/n with a Wilson 95% interval per arm.

## Decision rule (fixed before seeds 1-2 are observed)

**GENUINE OBJECTIVE EFFECT** requires *all three*:

1. **Coverage precondition.** Every seed of both compared arms has
   `coverage_count >= 1`. Any cell with coverage 0 is `INVALID_NO_SIGNAL`
   (per `scripts/analyze_revalidation.py`) and the comparison is not made.
2. **Interval separation.** The higher arm's Wilson-95 lower bound exceeds the
   lower arm's Wilson-95 upper bound (disjoint and ordered).
3. **Direction consistency.** The same arm is higher in **3 of 3 seeds**. One
   reversal is disqualifying.

**SEED NOISE** if any of: a seed reversal, overlapping Wilson intervals, or a
coverage-0 cell.

**Power, stated honestly.** At ~4% coverage the pooled n per arm is ~18. A
perfect split (18/18 vs 0/18) gives Wilson (0.815, 1.0) vs (0.0, 0.185) —
comfortably disjoint. A moderate split (14/18 vs 4/18) gives (0.529, 0.902) vs
(0.090, 0.470) — disjoint only barely. **This design can detect a large effect
and nothing else.** A negative result is therefore uninformative about small
effects and must not be reported as "no objective effect", only as "no large
effect detectable at this coverage".

## Excluded from the decision (pre-declared)

- **`exact_gold_span` in constrained mode is banned as a decision input.** The
  result files carry `span_metrics_are_tautological: true` — the span comes from
  candidate selection, not the model. A 6/6 there measures the candidate set.
  It is being removed from constrained-mode output rather than footnoted.
- Training `final_loss`. It is not a task metric.
- Any unconstrained-mode span metric while unconstrained coverage is 0.

## Capability-floor clause (fixed now)

If, across all nine runs, pooled constrained coverage is **< 10%** of eligible
atoms **and** unconstrained coverage is 0 in **>= 8/9** runs, then the registered
conclusion is:

> 30M parameters at 1800 steps on a character-level hash tokenizer is below the
> capability floor for `p1_screening_eval_v1`. The arm comparison rests on a
> small covered subset and is not used to rank architectures.

That is a real result and is cheaper to establish than another wave. It is
recorded as such, not as a failed experiment.

## Reporting order (fixed now)

1. **Coverage first**, as the headline, with the abstention count stated in
   absolute terms (e.g. "abstains on 144 of 150").
2. **Arm contrast second**, explicitly conditioned on the covered subset and its
   size.

Reporting the arm split ahead of coverage would repeat the propagation failure
that carried `dc3b310`'s "0% → 83%" past its own conditions.

## Recheck

```bash
.venv/bin/python scripts/analyze_revalidation.py \
  --results-dir artifacts/campaign/reval_results_causalfix \
  --out artifacts/campaign/native30_revalidation_summary_causalfix.json
```

## Status

**Registered, not resolved.** Seeds 1-2 unobserved at registration. Append a
RESULT section below without editing the rule above.
