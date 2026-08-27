# Readiness review — D3.3 context/tokenizer wave

**Date:** 2026-08-26
**Reviews:** `artifacts/PREREG-D33-context-tokenizer-wave.md`, read at
`de188a2` on `frontier/accelerated-research-campaign-v2`
**Against:** the eighteen-field standard, `NANO_VNEXT_MASTER_SPEC.md` §20, and
the readiness gate, §25
**Verdict:** **NOT READY** — one blocking gap (field 8). Not an authorization
question; it would still be NOT READY if authorized.

---

## Why this is a review and not a second preregistration

A rival preregistration for this wave was drafted on
`work/question-before-architecture` and **withdrawn before commit**. It proposed
a different primary endpoint — *does the capability floor survive a fitting
tokenizer* — for the same single run whose registered endpoint is *pooled
unconstrained coverage*.

**Two preregistrations for one run is two primary endpoints, which is a garden
of forking paths.** Whichever moved could have been reported as the result. The
D33 prereg is canonical for this wave; this document registers no endpoint,
proposes no arm, and does not compete with it.

The withdrawn draft was better on exactly one axis — it was written to all
eighteen fields — so what it found is preserved here as review findings.

## What the D33 prereg does better than the withdrawn draft

Recorded so the withdrawal is not read as a demotion:

- **A pre-committed numeric endpoint.** CONFIRMED ≥ 45/450 (10%), REFUTED
  ≤ 4/450, UNRESOLVED in between and *reported as such*. The withdrawn draft
  routed its decision rule through `seed_spread` without absolute bounds, which
  is weaker — it could not have fired without a second measurement.
- **An explicit non-comparability table** across all four provenance classes
  (ARCHIVED, KAGGLE, CAUSALFIX, GATED), each `char-level, 512`, each **not
  comparable** to this wave. The withdrawn draft had nothing equivalent. This is
  the difference between swapping an instrument and swapping a hyperparameter,
  and it is the single most important thing in the document.
- **A separate namespace** (`reval30_*_bpe_*`), so results cannot be silently
  pooled with the frozen four.
- **A stated predicted direction** — CONFIRMED — recorded up front *so a null is
  informative rather than deniable*.

## Field-by-field

Satisfied by the D33 prereg as written: 2 (scientific question), 3 (instrument),
4 (measured bottleneck — truncation, 82.7%, correctly distinguished from the
causal claim), 5 (hypothesis, H-context), 7 (manipulation, and only it), 11
(decision rule, with numeric bounds), 13 (falsifier, via REFUTED), 14
(authorization — states outright that nothing has launched), 15 (provenance —
namespace, branch, asset, run_ids, steps, seeds).

Carried implicitly and worth stating explicitly: 6 (baseline — the frozen GATED
wave is the comparison, but §22 also requires an **adversarial** baseline on the
*same* instrument; a majority-class or constant-output floor under BPE is not
named, and the native30 line has returned `NOT_SEPARATED` at effect +0.0000
before, where a null without a floor is unreadable), 9 (confound analysis — the
non-comparability table does most of this work), 16 (resource accounting), 17
(reproducibility — nine run_ids and seeds are fixed, but no statement of what
must replicate before the result is believed), 18 (interpretation boundary — §2
constrains it well; worth saying plainly that a CONFIRMED result licenses *"a
fitting tokenizer restores coverage"*, not *"context length was the mechanism"*,
since compression and fit move together).

Not separable from field 13 as written: **12 (kill condition)**. REFUTED and
"the instrument never tested it" are different outcomes and currently share a
branch. Concretely, `INVALID_NO_SIGNAL` cells, a failed integrity block, or
malformed output under the new tokenizer would all produce a low coverage number
that reads as REFUTED. Recommend a named kill condition so a broken run cannot
be banked as a negative one — this is the arm-B error one level up.

## The blocking gap — field 8, invariance requirements

The prereg's §3 says *"Nothing else changes: same corpus, same nine run_ids, same
1800 steps, same arms, same seeds, same integrity gate."*

That is an **argument that the manipulation is clean, not a check that it was**.
The distinction is R8, and it was paid for: E-DELIMIT arm B's format-feasibility
gate passed at ~96% while the task collapsed to chance. Format compliance and
task preservation are different properties; so are *token-length* preservation
and *task* preservation.

What is bounded: prompt length (median 530 → 204), overflow (82.7% → 0/150),
compression (2.60×), vocabulary (4098, unchanged, so embedding shapes are
untouched). All measured.

What is not bounded: that the BPE encoding preserves *what the eval is asking*.
A round-trip `decode(encode(p)) == p` over all 150 prompts is cheap and worth
running, but it bounds lossless-ness of the tokenizer, **not** equivalence of
the task — it would pass even if the retokenized prompt were harder for the
model to use, which is precisely the arm-B failure mode.

**This is the same class of defect as D7, one level up.** D7 found that the *BPB
measurement* silently desensitises ~2.6× under this exact swap. The open
question is whether the *task* does too. D7 checked the measurement; nothing yet
checks the task.

Under §25 an unanswered gate question is a stop, so the wave is NOT READY until
a bound is stated — or until it is argued, on the record, that no bound is
achievable and the wave proceeds as explicitly exploratory with that limitation
named in its interpretation boundary. Either resolution is acceptable. Silence
is not.

## Recommendations

1. State a field-8 invariance bound, or record on the prereg that none is
   achievable and reclassify the wave as exploratory.
2. Separate field 12 from field 13 so a broken instrument cannot be read as
   REFUTED.
3. Name an adversarial floor baseline under BPE, per §22.
4. Keep everything else as written.

## Interpretation boundary of this review

Establishes that the D33 prereg has one blocking and two non-blocking gaps
against the eighteen-field standard. Does **not** evaluate whether H-context is
true, does not authorize anything, and does not touch the frozen four
populations. No number here is new: all are read from the D33 prereg and
`de188a2:artifacts/campaign/TOKENIZER_CONTEXT_CONFOUND.md`.
