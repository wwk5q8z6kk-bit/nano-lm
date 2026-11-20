# H7-BM — local boundary ranking

**Status:** selected bounded intervention; not yet trained or scored
**Capability:** P1 faithful evidence transport
**Changes:** one training objective term only

## Observation

Terminal H6 remains rejected and its one-shot evaluation remains consumed. A
hash-authenticated post-hoc diagnosis of those frozen rows found 858 fields with
an evidence-binding error. Only 12 were purely disjoint. The largest pure
failure shapes were a raw span containing the gold span (467), a gold span
containing the raw span (192), and partial overlap (48); 86 had the wrong span
cardinality. These counts diagnose where H6 failed. They are not a new H6 gate
and do not authorize rescoring H6.

H6 already optimized global start/end cross-entropy. It did not explicitly
rank the exact boundary above adjacent Patient tokens. The evidence therefore
supports a local boundary-precision test before another architecture change.

## Hypothesis

> Adding a local hard-negative margin term around each gold start and end token
> will reduce exact evidence-boundary errors on a new, disjoint synthetic screen
> without materially changing state selection.

The mechanism is intentionally narrow. Nano's trunk, H6 state-conditioned
queries, state objective, global pointer objective, data exposure, optimizer,
verifier, decoder, and inference contract remain fixed. The treatment adds only
`nano_ai.training.boundary_margin.local_boundary_margin_loss` to the training
loss.

## Why this is Nano work

Nano's defining claim is faithful representation: a state is useful only when
it is grounded in the exact source evidence that supports it. This experiment
trains Nano's own evidence-binding mechanism. It is not a product feature and
does not turn Nano into a research application or a wrapper around another
model.

## Falsifier

The intervention is rejected if the treatment fails its preregistered paired
boundary-improvement bar, regresses state selection beyond the non-inferiority
limit, or trades containment errors for disjoint evidence. A failed result
returns the program to architecture choice between span refinement and a
generate-then-relocate path; it does not trigger threshold tuning on H6.

Execution bars and data boundaries are frozen in
[`../preregistrations/H7_BOUNDARY_MARGIN_SCREEN.md`](../preregistrations/H7_BOUNDARY_MARGIN_SCREEN.md).
