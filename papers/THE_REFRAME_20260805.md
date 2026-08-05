# The reframe — what two review swarms and one falsified experiment established

**Date:** 2026-08-05. **Status:** synthesis of three independent findings, each
verified against code or recorded evidence by hand after the agents reported.
Nothing here authorizes a run. It changes what the next run should be.

---

## 1. Three corrections to the program's own self-description

**(a) The trunk was never frozen.** H2–H6 are described in places as heads
bolted onto a frozen trunk. They are not: `nano_ai/training/train_*.py` hand
**all** parameters to AdamW, no `requires_grad_(False)` exists outside test
fixtures and the DPO/GRPO reference policies, and seven scripts record
`"full_trunk_trainable": True` (verified: `train_evidence_query_h4.py:970`,
`train_evidence_query_h6.py:610`, `evaluate_evidence_query.py:339`,
`evaluate_pointer.py:426`, and three more). Every one of the five rejected
interventions therefore ran inside **full-FT on a weak base** — which
`papers/paper2_draft.md` independently measures as the *worst cell of the 2×2*.
Five hypotheses were tested in the one regime the project's own data says is
handicapped.

**(b) Held-out value copying is not at the floor.** `papers/DECISION_GATES.md:109`
records H5 held-value at **2,220/2,987 = 74.3%**, against a required 2,167 —
**Pass** (verified verbatim). H5 was rejected on **absence (280/413), conflict
(149/250), uncertainty (162/250)**. On the H-line instrument the binding
constraint was never copying. It was **abstention calibration**.

**(c) The residual copying gap is one slot, and its cause is five words.**
Per-slot fingerprints in `papers/paper2_draft.md`: at the best measured corner
(160M / 3.2B tokens / LoRA) complaint and medication go to **0.0** clean gap —
solved — while allergy reads **100.0**, and allergy reads 100.0 in *all five*
own-stack configurations ever run. The cause is one line:
`scribe/build_scribe_data_v2.py:30` —
`ALG_TRAIN = ["penicillin","peanuts","pollen","latex","shellfish"]`
(verified) — a **five-value** training pool, present in only ~half of examples.

And the one intervention that ever moved it, `trajectory/PREREG_slot_diversity.md`:
held-type recall **0.00 → 24.53 → 66.67** for pool sizes D5 → D20 → D80 at
*fixed* scale, position-controlled, H-slot **SUPPORTED**. That lever has
**never been crossed** with the data-quantity and adaptation-method levers that
solved the other slots. The most promising remaining experiment in the entire
program costs a finetune, not a pretrain.

---

## 2. The through-line: over-abstention is the adversary, in all three code bodies

The session's independent findings converge on one failure mode:

| Where | Evidence | Shape |
|---|---|---|
| **Model** (H-line) | H5 rejected on absence/conflict/uncertain while *passing* held-value | miscalibrated withholding |
| **Product** (wedge_v1) | `.studies/…-r4`: 3/10 useful, `FIX_REPEATED_FAILURE`, `OVER_ABSTENTION` ×3 | withholds answers that are verbatim present |
| **Instrument** (fabric) | gate is `presented_err / max(1, presented)` — abstain-on-everything scores 0.0% and PASSES | **latent hazard only — see correction below** |

> **CORRECTION (same day, after computing the numbers).** The fabric row above
> originally claimed uncounted withholding. That is withdrawn. Across all 24
> cells of `fabric/results_slice_v1.json`: **withheld = 2,642, caught_err =
> 2,642, lost_correct = 0** — every withheld field was an error, and both
> quantities are already recorded per cell. Fabric's *gate* is degenerate (a
> mute verifier would pass it), which is a real structural flaw worth fixing;
> fabric's *verifier* is not over-abstaining. The realised over-abstention is in
> two places, not three: the model and wedge.

The project optimized hard for "never assert what you cannot ground," and got
it — that discipline is real and is the product's differentiator. The cost was
paid on the other side of the ledger, and **the instruments were built so that
this particular cost does not appear in the headline number.** That is the
single most important structural fact discovered today.

---

## 3. What I falsified myself (method note)

The 13-agent architecture swarm ranked "fix the retrieval margin" as backlog
item B1 — *days* of work — citing the study's own recorded failure reason
("low-margin retrieval suppressed all literal evidence"). A sweep
(`wedge_v1/eval/margin_sweep.py`) took it from τ=0.5 to **τ=0.0**: recovery
**0/3**, with a passing positive control.

Tracing further: for `Nano Runtime smallest sufficient solver`, retrieval
returns the correct paragraph at **promote=True, margin=3.728** with the answer
verbatim, and the system still abstains with `EMPTY_EVIDENCE_REJECTED`.
Retrieval is innocent; **evidence-atom binding** is where claims die. (A third
card, `E1 KILL M1_template`, is a task-authoring artifact — that literal appears
**0** times; the corpus says `M1 template`.)

So a recorded diagnosis, and a top-ranked backlog item built on it, were both
wrong — and cost ~10 minutes to disprove because the experiment was cheap and
had a control. **Cheap falsification before expensive building** is the lesson
worth keeping.

---

## 4. What this makes the next moves

1. **Finish H6** — training is complete (`TRAINING_FREEZE_KAGGLE.json`), backed
   up on two domains, development still sealed. One evaluation kernel, once.
   Note the standing hazard: H6's frozen threshold policy takes seed-20260805
   epoch-2 from **0.952 uncalibrated to 0.281 calibrated**, so a raw pass can
   still die at the calibrated stage — and *that* is the abstention-calibration
   problem again, in the gate itself.
2. **Fix the binder, not the margin** (product gate) — diagnose why
   `EMPTY_EVIDENCE_REJECTED` fires on promoted, on-target, verbatim-matching
   paragraphs in `wedge_v1`. This is the measured product blocker.
3. **Publish the denominator** (B3) — coverage and yield beside every fabric
   presented-error number; constant-free gate guard. Cheap, and it makes the
   over-abstention visible where it is currently invisible.
4. **Cross slot-diversity with the winning corner** (the highest-leverage
   *model* experiment now known) — D80-style value pools inside the
   LoRA/Chinchilla cell, targeting the one unsolved slot. Preregister with the
   +30 pt threshold the original sweep used.
5. **Rung-1 reconsidered** — before spending $150 on pretraining scale, note
   that the project's own evidence says the remaining gap is a *finetune-data*
   problem, not a scale problem. Rung 1 may be the wrong next purchase.

Full detail: `papers/ARCHITECTURE_RESEARCH_2026.md` (literature, graded) and
`papers/ARCHITECTURE_ENHANCEMENT_PLAN.md` (subsystem backlog).
