# PREREG — <short experiment name>

**Registered <YYYY-MM-DD>, before any arm ran.**
**Standard:** question-before-architecture-v1
**Rule source:** <hypothesis file / decision record this rule comes from>
**Base commit:** <sha the experiment will run from>

> Fill every section. `scripts/check_prereg.py` enforces presence, not quality —
> an unanswered field is a stop, not a caveat to note in the write-up (§25).
> Delete the parenthetical prompts as you go; leaving `TODO`/`TBD` fails the
> check.

---

## 1. Product question

(What real Nano capability or product requirement is ultimately being improved?
Name the product surface from §2, not a benchmark.)

## 2. Scientific question

(What specific uncertainty about the system is being resolved? This is the
question, not the proposed answer.)

## 3. Instrument

(Which benchmark, task, workload, or capability line actually measures that
question? Name the eval, its size, and its analyzer.)

## 4. Measured bottleneck

(What failure mode has *already been observed* on that instrument? Cite the run
id or artifact. A suspected bottleneck is not a measured one.)

## 5. Hypothesis

(What mechanism addresses that bottleneck, and what causal prediction does it
make?)

## 6. Baseline / control

(What is the comparison, and what must remain unchanged? Include an adversarial
baseline on the same instrument — parrot / majority-class / constant.)

## 7. Manipulation

(What exactly changes between conditions? One thing — see contrast hygiene
R1–R5.)

## 8. Invariance requirements

(What must remain statistically or operationally equivalent for the experiment
to answer its question? **Bound it numerically.** Breach ⇒ VOID. Checked
*before* the primary endpoint is read — see R8.)

## 9. Confound analysis

(What alternative explanations could produce the observed result, and how does
the design distinguish them? An experiment that yields the same artifact
whichever explanation is true is not worth its compute.)

## 10. Outcome measures

(What quantities are measured — capability, quality, compute, latency, memory,
cost?)

## 11. Decision rule

(What constitutes SUPPORTED, REFUTED, NULL/INCONCLUSIVE, and methodological
failure? Fix the thresholds here; post-hoc bar movement is bar-chasing.)

## 12. Kill condition

(What observation makes the experiment *incapable of answering its question*,
requiring VOID rather than interpretation? Distinct from the falsifier.)

## 13. Falsifier

(What result would genuinely count against the proposed mechanism?)

## 14. Authorization

(Exploratory or confirmatory? Who authorized it, when, and at what scope?
Cheapness is not evidential validity — a \$0 run can still be confirmatory.)

## 15. Provenance

(Exactly which code, model, data, configuration, seed, analyzer, and evaluation
rules produce the result. Another researcher must be able to reconstruct it.)

## 16. Resource accounting

(Compute, latency, memory, energy, monetary cost — and is the experiment
proportionate to the uncertainty it resolves?)

## 17. Reproducibility

(What must be repeated across seeds, instances, machines, or analyzers before
the result is considered reliable? An effect smaller than the seed spread is not
an architecture effect.)

## 18. Interpretation boundary

(What does this establish, what does it *not* establish, and what claims remain
untested?)

---

## Status

**Registered, not resolved.** Append a RESULT section below without editing
anything above it. The result closes in exactly one verdict: SUPPORTED /
REFUTED / NULL-INCONCLUSIVE / VOID / PENDING / NOT-AUTHORIZED.
