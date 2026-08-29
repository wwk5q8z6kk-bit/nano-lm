# Readiness review — E3 dual-clinician arm

**Date:** 2026-08-26
**Reviews:** `trajectory/PREREG_E3_dual_clinician_arm.md` — **cross-branch**,
read at `e6d2fc4` on `work/leakage-power-analysis`; not on this branch
**Against:** the eighteen-field standard (`NANO_VNEXT_MASTER_SPEC.md` §20) and
the readiness gate (§25)
**Verdict:** **READY on the standard's terms.** No blocking gap. Three minor
additions recommended. Still requires owner authorization and clinician
recruitment, neither of which is a readiness question.

This is the document that gates the construct question, which in turn gates both
ranked candidates. It is the strongest preregistration in the program.

---

## The finding that matters is about the standard, not this document

**Field 7 (manipulation) does not apply here, and the standard does not say
what to do about that.**

This is a *measurement* arm. Nothing is manipulated: two clinicians rate frozen
items under a fixed rubric. There is no intervention, no arm contrast, no
condition the model is run under. The eighteen-field chain was derived from
intervention experiments — E-DELIMIT, the tokenizer swap — and it silently
assumes a manipulation exists.

Forcing a manipulation field here would be worse than leaving it empty: it would
invite someone to invent a contrast to satisfy a checklist, which is the
"interesting architecture → implementation → post-hoc explanation" direction the
governing rule forbids.

**Resolved in the standard rather than in this review** — §20 now states that a
field may be marked `N/A — <reason>` for non-interventional studies
(measurement, construct validity, instrument calibration, power analysis), that
an unjustified `N/A` is an unanswered field and therefore a stop, and that
fields 8 and 12 are **never** `N/A`, because a measurement study can still fail
its own preconditions. `check_prereg.py` accepts a justified `N/A` and rejects a
bare one.

Two of this program's five open hypotheses are instrument questions rather than
mechanism questions. A standard that only fits interventions would have pushed
both toward being framed as interventions.

## What this protocol does better than anything else in the program

Recorded because these are the patterns worth copying, not politeness.

- **Sample size derived, not asserted.** `nanoscribe/iaa_power.py`, asymptotic
  SE of Cohen's κ inverted for n, with the table shown. Pool B n=150 clears a
  0.60 lower bound even if true κ is 0.70. The binding constraint is named —
  precision on κ, not on the rate — with the reason: an unreliable pair makes
  the rate uninterpretable however tight its CI.
- **The κ prevalence paradox is pre-registered, with the statistic chosen in
  advance.** If κ < 0.60 while p_o ≥ 0.85 and marginals skew past 80/20, the
  reading is fixed now and the gate moves to PABAK. *"Decided now, not after
  seeing which statistic is kinder"* is the whole discipline in one line.
- **Anchors are a rater-side manipulation check.** 10% pre-labelled, >20%
  failure excludes a rater, recorded not silently dropped — explicitly the R5
  analogue. Without it an inattentive rater yields low κ that reads as *"the
  construct is not rateable"*. This is field 8 done properly on a study that
  has no manipulation to check.
- **Interim is futility-only**, never stop-for-success, with the type-I
  inflation named as the reason.
- **Two estimands kept separate.** Pool A (scribe, value equivalence) and Pool B
  (span-port, evidence unit) share a rating session but not a decision rule —
  and the document says merging them *"would repeat the error this program has
  already made once"*, transporting a conclusion between instruments. That is
  the retrieval-vs-delimitation rule applied before anyone had to be told.
- **The owner is excluded as a primary rater**, with the reason stated: the
  owner is a practising physician, which is what makes the arm runnable, and has
  seen outputs, hypotheses and the predicted direction, so is not blind. Owner
  labels, if any, are a declared non-blind sensitivity analysis outside primary
  κ. Blinding is enumerated exhaustively.
- **§8 is the best interpretation boundary in the program.** It names that the
  arm does not resolve H-delimit vs H-retrieve, is synthetic-world only, and
  that *two raters give reliability, not validity* — a shared professional
  convention can be reliably held and still be the wrong target.

## Field-by-field

Satisfied: 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 18. Field 12 is
genuinely separate from 13 — the κ gate and the n=40 futility stop say *the
construct is not reliably rateable*, which is not the same as either Pool-B
verdict. That separation is what the D33 prereg lacked.

`N/A` with reason: **7 (manipulation)** — non-interventional.

Three recommended additions, none blocking:

1. **Field 16 — clinician time is unpriced.** 250 items × ~30 s ≈ 2 hours per
   clinician, stated; the *cost* of two practising clinicians is not. That is
   the scarcest resource this program has, and the proportionality test in field
   16 cannot be applied without it. It plausibly exceeds every compute cost in
   the ledger to date.
2. **Field 17 — reproducibility is within-pair, not across pairs.** One rater
   pair yields one κ. Nothing states whether a second pair must reproduce the
   verdict before it is believed, which matters most for the ≥70% / ≤30%
   branches, since both license large downstream moves.
3. **Field 1 — the product question is implicit.** The chain to physician-facing
   record review and evidence-grounded documentation (§2) is real but unstated.
   One line prevents a later reader treating this as methodology for its own
   sake.

## Consequence for the two ranked candidates

Neither candidate's readiness changes, but the ordering is now explicit.

Pool B's decision table is what settles candidate 2. `accept as-is` ≥ 70% ⇒ the
minimal-span convention is **refuted**, ~80% of "failures" are not failures, and
**E-DELIMIT round 2 should not run as scoped** — the span-port line would need a
new primary metric first. ≤ 30% ⇒ the convention is validated, "delimitation
failure" becomes licensed wording, and round 2 becomes the right experiment.
30–70% ⇒ unresolved, the wording ban stands, and neither move is licensed.

This confirms from the protocol's own side what the readiness gate implied
independently: **candidate 2 is blocked behind E3, not merely unauthorized.**
Candidate 1 is untouched by it — its blocker is its own field-8 invariance gap.

## Interpretation boundary of this review

Establishes that the E3 protocol has no blocking gap against the eighteen-field
standard, and that the standard needed a rule for non-interventional studies.
Does **not** evaluate whether the protocol's thresholds are the right ones, does
not authorize collection, and makes no clinical claim. No number here is new;
all are read from the protocol at `e6d2fc4`.
