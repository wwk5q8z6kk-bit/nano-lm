# Independent review — margin results and conditional durability

**Verdict:** the reviewed chapter and module accurately separate descriptive source evidence, constructed counterexamples and conditional preservation results. No material sign error, missing interaction term, unsupported necessity theorem or unqualified floating-point certificate remains in the reviewed text. Three source/assumption clarifications requested during review were incorporated.

## Reviewed identities

| Item | Identity |
|---|---|
| Research evidence cutoff | `9d88a2c20233f42b4a49c4430264b20f842895ad` |
| Original margin implementation Git blob | `4e14e8f25a22fa743dd6ffcc51a812df11ecad16` |
| `analysis/margin_durability.py` SHA-256 | `00c37e665770370d06123ec1a97d9f35cfed2248defa5a752e7bbec02aa6bf3d` |
| `checks/test_margin_durability.py` SHA-256 | `7a63d38175ad02ce60999f77f1848aad1563f2467ed80cc7e071836ba4f8cfc3` |
| Chapter 11 SHA-256 at factual/math review | `32f7a4b9506c29b0af88f7ece261d4a2bf241c028e32b2a8b512cc43cf817c49` |

The review used the exact frozen source copies and [manifest](../evidence/source_manifest.json), the [independent source audit](margin_evidence_review.md), the committed rounded result fields and the [reanalysis](../results/margin_reanalysis_v4.json). Later integration-status edits do not expand the scope of this hash-specific mathematical review.

## Source and endpoint checks

The five-arm means, margins 1/3/6 trend, approximately 11.00-point old-regular improvement and 9.76-point old-correction decrease agree with the source summaries. The M1 and finite M4 conditions hold in all recorded seeds; the compound M2 and M3 success criteria fail. Seed ranges are descriptive extrema, not confidence intervals. Matching old and new rounded fields on reused seeds is correctly narrower than an independent replication claim.

The chapter correctly distinguishes solver return status, immediate acquisition, phase-end survival and later retention. Its 624-event stream reconstructions, repeated-entity counts, same-value counts and final cohort denominators match the conditional source reconstruction. They are not falsely labeled saved run metadata. Historical per-write predictions and residuals remain absent; repeated-entity collisions can accommodate the observed errors but do not establish their cause. The constructed two-write LP example demonstrates possibility, not historical attribution.

| Requested clarification | Resolution in reviewed chapter |
|---|---|
| The arm named `half` uses `min(1, 0.5 × current top-two gap)` | The cap and the difference from uncapped 90% preservation are explicit. A gap larger than 2 is not protected at half its value. |
| `edit_l1` excludes unsuccessful LP writes | The successful-write denominator and per-seed rounding are explicit. |
| An orthogonal address update is harmless only under the stated readout condition | The counterexample table explicitly fixes the decoder and describes an address-only update. |

The same first two clarifications were added to this reviewer's source-audit document. Raw upstream files were preserved without modification.

## Mathematical checks

Proposition 11.1 correctly expands a pairwise bilinear margin change into the state term, decoder term and their interaction. Applying the triangle inequality and Cauchy–Schwarz to the three terms gives the stated norm bound. Keeping the interaction term matters when both components change.

Proposition 11.2 follows by telescoping the one-step lower bounds. Nonnegative budgets make a strict full-horizon bound sufficient for every supplied prefix. Positivity against **every** rival gives a unique target argmax. The semantic target must remain unchanged; a revision requires a new trajectory or the declared censoring policy.

The converse is not claimed. A failed sufficient test is inconclusive. Fixed-decoder orthogonal state changes and coherent invertible transport can preserve logits despite a loose or negative norm lower bound. Thus the source pilot does not prove that every future update must reassert a constraint, nor that an L1-minimal parameter write is intrinsically the most harmful functional change.

The native-model extension is conditional on a valid Hessian operator-norm bound along the whole actual step segment for a fixed prefix and target. A sampled Hessian or stale Jacobian does not meet that premise. The chapter correctly refrains from importing the bilinear result as a complete transformer certificate or a learned protection algorithm.

## Numerical verification and limits

An independent execution of the [margin suite](../checks/test_margin_durability.py) passes **9 of 9 tests**. These cover known-answer joint drift, finite trajectories, random finite examples, large-margin reversal, harmless orthogonal changes, coherent transport, ties, malformed inputs and output-file preservation. The displayed decomposition and drift calculations agree with the reviewed equations.

The implementation computes float64 diagnostics. Its positive lower bound is not an outward-rounded interval certificate; residual and rounding enclosures would be required for that claim. The theorem is sufficient in real arithmetic and limited to the supplied states, relevant rivals and unchanged semantic target. It makes no automatic statement about unsampled prefixes, paraphrases, distributional coverage, training stability or language quality.

Reproduction from the architecture directory:

```bash
python3 -B checks/test_margin_durability.py
```

No upstream neural experiment was rerun for this source/math review. The original margin-pilot precision and event-history gaps remain empirical work, with explicit successor measurement requirements rather than invented closure.
