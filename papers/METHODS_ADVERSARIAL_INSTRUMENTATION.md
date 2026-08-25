# Methods — adversarial instrumentation as a primary control

**Status:** methods contribution, candidate for Paper α §measurement-reliability
(the paper's stated second contribution). Currently scattered across commit
messages and defect records; this file consolidates it.

## Claim

In this program, **primary metrics have never caught a measurement defect.**
Every defect found so far was caught by a test whose explicit job was to prove
the instrument could detect a failure it was supposed to detect. That is not a
run of bad luck across unrelated bugs — it is a structural property of
optimistic instrumentation, and it implies adversarial self-tests should be
treated as a primary control, not as hygiene.

## The pattern

An optimism-inflating defect is one that makes the reported number better while
making reality worse. Such defects are invisible to the reported number *by
construction* — that is what makes them optimism-inflating. A metric cannot
detect a defect whose effect is to improve the metric.

The detector must therefore be a test that asserts a **property the instrument
must have**, independent of the number it produces:

- *the manipulation check* — the instrument must provably detect a model that
  merely echoes;
- *region response* — each objective must move when, and only when, its own
  supervision region changes;
- *causality* — a future token must not be able to change an earlier logit.

Each is falsifiable, cheap, and orthogonal to the headline result.

## Verified instances

### 1. Pointer-head manipulation check → `C_POINTER_P1` VOID

`scribe/pointer/PREREG_pointer_head.md` pre-registered a **blocking**
manipulation check: before the gap rule could be read, the copy pathway had to
be shown to have actually engaged. It had not (M=0.18, p_gen=0.83 — the copy
channel never activated). The arm was declared **VOID rather than REFUTED**,
because a non-engaged mechanism cannot falsify a hypothesis about that
mechanism.

Recorded in `papers/EVIDENCE_LEDGER.md` as `C_POINTER_P1 | VOID | VOID |
manipulation check failed`. Without the check this would have been reported as
a clean refutation of the copy-head hypothesis — an optimistic, wrong
conclusion in the confident direction.

### 2. Non-causal attention in the native track → `D_NATIVE_CAUSAL_MASK`

Full detail: `papers/DEFECT_NATIVE_CAUSAL_MASK.md`.

`nanoscribe/native/model.py` ran full bidirectional attention in a next-token
decoder. Training loss read 0.017-0.084 and the analyzer reported six clean
`NOT_SEPARATED` nulls — both perfectly consistent with a healthy experiment.
Neither number could have revealed the defect, because leakage *is* what drove
the loss down.

What surfaced it: the requirement that each objective respond only to its own
region. Editing the span moved the label loss, which is impossible under causal
attention. The contradiction forced a direct causality probe, which measured a
20.15 logit delta at positions the future should not reach.

Note the dependency — this was found only while fixing a *different* defect
(objectives being scalar multiples of one another). Had the arms not needed
separating, the leak would still be in place.

### 3. Owner-reported: `unbound_assertion` laundered into `correct_abstention`

**Not verifiable from the current tree.** `unbound_assertion` has zero
occurrences in the repository; `correct_abstention` exists at
`nanoscribe/harness.py:169,190` and `nanoscribe/benchmark.py:41`. This instance
is recorded here as reported by the owner and **needs its own defect record
with the same enumerated scope** before it can be cited as evidence. Do not
cite it as a third verified instance until that record exists.

## Why this belongs in Paper α

Paper α's second stated contribution is measurement reliability. The
substantive finding is not "we fixed some bugs" — it is:

> Across this program, defects that inflate a reported result have been
> undetectable by the reported result, and have been detected only by
> pre-registered checks asserting instrument properties. Where such a check was
> blocking, it converted a would-be false conclusion into a VOID.

That is a falsifiable methodological claim with worked instances, one of which
(`C_POINTER_P1`) is already in the frozen evidence and cost a full arm.

## Standing rule this implies

Any instrument that produces a headline number should ship with at least one
blocking check that:

1. asserts a property the instrument must have, not a value it should produce;
2. is verified to **fail** against a deliberately broken version — an
   unverified check is not a control (both causality pins in
   `nanoscribe/test_native_loss_target_budget.py` were run against the pre-fix
   model and confirmed to fail);
3. blocks the reading of the primary result when it fires, rather than being
   reported alongside it.

Point 2 is the one most easily skipped, and it is the one that makes the
difference between a control and a decoration.

## Open

- Record defect #3 properly, or withdraw it from this file.
- Decide whether the three-defect pattern is stated in Paper α as a
  methodological finding or held for a separate methods note. It is currently
  neither.
