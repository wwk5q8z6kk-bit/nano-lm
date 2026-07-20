# PREREG — Stage V: compositional verification (NanoScribe Phase D, draft only)

**Drafted 2026-07-20, after the fabric/ Phase 1 audit. NOT pre-registered until an owner
commits fixed thresholds — every number below is a candidate, explicitly marked
owner-decision-pending. Nothing in this document authorizes a run.** Sequencing:
this stage does not execute before C-1b is interpreted (owner instruction), and it
reuses fabric/'s typed-claim/ledger instrumentation rather than building new plumbing.

## Question

Can a verifier improve presented coverage beyond `fabric/`'s v1 (literal substring,
role-aware) and v2 (closed-world template-anchored) without regressing the
zero-presented-error standard v2 already measures on held-out templates and values?

## Why v1/v2 don't already answer this

- v1 is a real lexical-grounding baseline; its own residual (nano 3/158 presented,
  scale 2/179 presented) is exactly the class of binding failure — cross-slot capture,
  template-word capture, partial copy — that a smarter verifier should catch.
- v2 catches all of it, but by design: it re-derives ground truth from the source
  dialogue via the same templates the generator used, then compares. It is a correct,
  honestly-labeled reference extractor for this closed world — not evidence that a
  *general* semantic verifier would do the same on paraphrase, negation, or
  distractor cases the current generator doesn't produce.
- **The current scribe generator has no paraphrase, distractor, or ambiguity axis.**
  Testing "semantic" verification for real needs a new held-out generator extension
  (paraphrased patient replies, wrong-speaker distractor lines, contradictory turns).
  That generator does not exist yet — building it is a Stage V *prerequisite*, not
  this document.

## Candidate arms (unchanged from the earlier sketch, restated for the record)

1. fabric v1 (literal substring, role-aware) — baseline.
2. Normalized lexical (casing/punctuation/morphology/controlled aliases).
3. Span-bound semantic (candidate must map to a specific patient span; semantic
   equivalence allowed; source span mandatory).
4. Unconstrained semantic — included as an adversarial comparison, expected to
   over-accept plausible-but-unsupported claims.

## Candidate evaluation slices

exact-copy · paraphrased · negated · wrong-speaker distractor · wrong-field distractor ·
unseen value types · absence claims · contradictory dialogue · ambiguous dialogue.
(Paraphrase/negation/distractor/contradiction/ambiguity slices require the generator
extension noted above — not yet built.)

## Candidate metrics (definitions only, not thresholds)

- Presented risk = incorrect presented / presented.
- Review load = flagged / all fields.
- Coverage = 1 − review load.

## Candidate thresholds — OWNER-DECISION-PENDING

Grounded in measured baselines from the fabric audit (not invented):

| Gate | Candidate value | Basis |
|---|---|---|
| Presented error ceiling | ≤ 0.0% (match v2's measured floor) | fabric v1/v2 measured: v1 1.1–1.9%, v2 0.0% on nano/scale × inst0 |
| Review load improvement target | reduce below fabric v1's measured 10.5–18.4% on the paraphrase/normalized slice specifically | fabric `results_slice_v1.json` (this run) |
| Coverage improvement | ≥5-pt reduction in review load vs. v1, on the new slices only, with presented error not worse | conservative margin above the ~1.3–2pt seed-noise floor already established (Q2 duplicate, corner seed check) |
| Wrong-role acceptance | 0 tolerated (hard gate, not graded) | matches fabric's existing role-aware invariant; no measured baseline needed — this is a correctness invariant, not a rate |
| Wrong-field acceptance | 0 tolerated (hard gate) | same |
| Seed/run variability | any single-seed cell reporting a candidate arm's presented-error or review-load delta must be bounded the same way as the empirical program: report ±, don't single-run a verdict | matches the project's own seed-duplicate discipline (§Q2) |

These are starting proposals only. An owner must fix the exact numbers (the ≥5-pt
margin above is illustrative, not derived from a power calculation) before this
becomes a real PREREG. Until then, treat this file as a drafted proposal, not a gate.

## Explicit prerequisite gap

Before any run can be designed against fixed thresholds: build the paraphrase/
distractor/contradiction generator extension (does not exist), decide whether it
counts as in-distribution or a new held-out axis, and confirm it doesn't leak into
the frozen anchors/sweep instruments used by the empirical program (regression-suite
invariant in `EMPIRICAL_FOUNDATION.md`).

## Status

Draft only. Not committed as an executable PREREG. No run authorized. Sequenced after
C-1b interpretation per owner instruction.
