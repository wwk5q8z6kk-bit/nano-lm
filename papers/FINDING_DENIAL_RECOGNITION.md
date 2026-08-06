# Finding — the absent-state failure is denial recognition, and the detector already exists

**2026-08-05. EXPLORATORY.** This analysis reads H6's development results,
which are already spent. It characterizes a mechanism; it is **not** a
confirmatory result and no gate may be claimed from it. Confirmation路 is
specified in §5.

---

## 1. What was found

H6's largest failure was `absent`: 199/413 correct (48.2%). Decomposing the
confusion:

| gold `absent` predicted as | count | share |
|---|---|---|
| **absent** (correct) | 199 | 48.2% |
| **supported** | **177** | **42.9%** |
| conflicting | 15 | 3.6% |
| missing | 14 | 3.4% |
| uncertain | 8 | 1.9% |

**Of the 177 mislabeled `supported`, 176 had the span exactly right.**

A representative case, verbatim from the diagnostics:

```
field       : medication
gold_state  : absent      raw_state : supported
span_exact  : True
gold_spans  : ["Nothing at all."]
raw_spans   : ["Nothing at all."]
```

The model located the denial perfectly and read it as an assertion of a value.
This is not retrieval, not span selection, and not capacity. It is a failure to
distinguish *a span that denies* from *a span that asserts* — a local, lexical
judgement.

## 2. The detector already exists

`nano_ai/contract.py:93` defines `_is_field_denial(field, text)` against
per-field `_DENIAL_PATTERNS`. It is used at `contract.py:313` to **validate**
that an ABSENT claim carries denial evidence — but it is never used to **decide**
the state. The information needed to fix the model's largest failure is already
in the codebase, on the validation side of the boundary.

Applying it to the fields the model labeled `supported`:

| | |
|---|---|
| absent-gold mislabeled `supported` | 177 |
| of those, denial pattern matches | **176** (99.4%) |
| denial pattern does not match | 1 |
| correct `supported` that would be wrongly flipped | **0 of 3,833** |

Zero false flips is the load-bearing number. The rule is not a coverage-for-risk
trade; on this data it is free.

## 3. What it would and would not achieve

| gate | H6 actual | with rule | required | |
|---|---|---|---|---|
| absent | 199/413 | **375/413** | ≥383 | **still fails by 8** |
| overall | 3,901/5,000 | 4,077/5,000 | ≥3,041 | passes either way |
| supported | 3,120/3,837 | unchanged | — | no regression |

Absent moves 48.2% → 90.8% against a 92.7% requirement. **The mechanism is
identified and 99.4% recoverable, and that alone still does not clear H6's
gate.** The residual 38 failures are a different mixture — 15 predicted
`conflicting`, 14 `missing`, 8 `uncertain`, 1 unmatched denial — and would need
their own diagnosis.

Stating this plainly matters: it would be easy to present "48% → 91%" as a fix
and quietly omit that the gate still fails.

## 4. Why this supports the decomposition thesis

`papers/ENHANCED_PLAN_20260805.md` predicted that composite epistemic states
fail because a single forward pass is naturally an *existential* operator, while
`ABSENT` requires `¬(∃ asserting span) ∧ (∃ denying span)`. The evidence is
sharper than the prediction: the model performs the existential part correctly
(the right span, 99.4% of the time) and fails only the *polarity* judgement on
the span it already found.

That is the decomposition thesis in its strongest form — and it means the fix is
smaller than proposed. Not a new head, not retraining: a rule applied to a span
the model already returns.

It also re-explains E1. A deterministic solver scored 0.999 against a generative
0.925 partly because denial recognition is exactly the kind of local lexical
judgement rules do reliably and a small generative model does not.

## 5. Confirmation path (required before any claim)

1. **Preregister** the rule and its gate before measuring anywhere else.
2. **Confirm on the calibration partition** — 800 examples / 4,000 fields,
   unspent for this purpose — using `nano_ai/training/run_threshold_sweep.py`'s
   loader path to produce fresh inference.
3. **Report coverage beside accuracy**, per `papers/SELECTIVE_VOCABULARY.md`.
4. Only then consider whether the rule belongs in the decision path, and under
   which preregistration.

## 6. Honest limits

- Development data is spent; this is characterization, not evidence for a gate.
- `_DENIAL_PATTERNS` are v0 and domain-specific to synthetic clinic text. Real
  documents deny in far more ways, and the zero-false-flip result will not
  survive unchanged outside this corpus.
- The residual 38 absent failures are unexplained.
- `conflicting` still has a genuine span problem (0.572 span accuracy) that no
  polarity rule addresses.
