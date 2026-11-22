# Independent review: causal n-gram cache comparator

**Reviewer:** /root/v5_native_comparison  
**Reviewed implementation:** prototype/causal_ngram_cache.py  
**Reviewed fixture suite:** prototype/test_causal_ngram_cache.py  
**Final review status:** causal/counting mechanics pass; both mixture API findings and a subsequent subnormal interpolation defect were corrected by the owner and independently rechecked.

This review executed only small integer cache fixtures. It did not execute the new native comparison, language data, a corpus benchmark, GPU work, network work or paid compute.

## Source identity at initial review

| File | SHA256 |
|---|---|
| causal_ngram_cache.py | abc131024f163e3a798ae9089e8bf632268e78328b615ec720d7abfc0f86b715 |
| test_causal_ngram_cache.py | f9fe1645215bed26152184e2f9f1793f1d55af354f1b3a3256a2d0922b10662f |

These sources include the owner's correction that normalizes a valid Real-valued smoothing input to a finite, positive Python float.

## Findings

### CC-R1: mixture weight needs equivalent numeric normalization

The mixture function accepts Real-valued weights but does not normalize the accepted value to Python float. With a valid matched cache prediction and native vector (0.1, 0.2, 0.3, 0.4), a weight of numpy.longdouble("0.5") returns numpy.longdouble entries. The numeric mixture is meaningful, but the advertised tuple of floats is not JSON-serializable in this environment. A Fraction equal to 10 raised to the 1000th power raises OverflowError during the finiteness check instead of the intended invalid-input error.

**Requested correction:** safely convert an accepted Real to a Python float, catch conversion overflow, then check finiteness and the closed interval [0, 1]. Add focused nonbuiltin-Real and extreme-input regression coverage. This follows the same policy already applied to smoothing.

### CC-R2: public cache prediction can bypass proper-distribution checks

The function validates its native probability input, but assumes a supplied CachePrediction is valid. CachePrediction is publicly constructible and dataclass replacement is allowed. Replacing a real prediction's probabilities with (-1.0, 2.0, 0.0, 0.0), setting matches to 1, and mixing at weight 1.0 returns that invalid vector unchanged.

**Requested correction:** validate the supplied cache prediction's type and positive normalized probability vector; ensure the matches/counts/matched-order metadata do not contradict the supported prediction contract. The function should reject malformed public inputs before returning a probability vector. An alternative opaque validated result type would need an explicit API change, which is unnecessary here.

Neither finding changes the observed count oracle or permits the cache's own predictor to see a future target. They are API-boundary correctness gaps for the optional native mixture.

## Causal and statistical contract review

| Area | Evidence and conclusion |
|---|---|
| Score before observation | score_then_observe validates the target and mask, obtains a prediction from the preceding state, scores that stored distribution, and only then observes the target. A masked target advances observed context without adding loss. Invalid token/mask inputs preserve state. |
| Longest context | predict searches suffix orders from the largest permitted order down to one. For a chosen order, only completed context/follower transitions contained in the retained prefix count. If no suffix has a completed follower, the cache distribution is uniform. |
| Unknown target exclusion | A candidate follower index is strictly below the retained prefix length. The unknown next token is absent from the table. Prediction accepts context identity but no target or future suffix. |
| Left window boundary | Rebuilding transitions from retained observed tokens excludes transitions whose source context lies before the left boundary. An independent probe with full history (0, 1, 2, 0), window 3, retained prefix (1, 2, 0) returned zero matches: the discarded 0-to-1 transition was not counted. |
| Overlap | Overlapping completed transitions are counted consistently. The independent grouped-table oracle covers all 1,093 prefixes of length zero through six over a three-token fixture alphabet. |
| Crop/document boundaries | reset clears both prefix and observed count. Identity mismatch rejects an operation until reset. The caller must perform reset for every crop as well as every document; the implementation does not infer boundaries from token IDs. |
| Same-prefix native mixing | The cache documents a caller obligation that native probabilities use exactly the same observed crop prefix. No native state or native input identity is passed to the mixture function, so this property is not mechanically established by the API or these fixtures. |
| No-match control | No matched transition causes native mixing to return the supplied native vector exactly, independent of the requested weight. Weight zero also preserves the native vector. This avoids making uniform cache probabilities an incidental no-match change to the native baseline. |
| Smoothing | The formula uses total smoothing mass alpha distributed as alpha/V across vocabulary entries. Positive finite values are normalized; an unrepresentable state raises instead of emitting a nonpositive or unnormalized distribution. This is not per-token add-alpha smoothing. |
| Snapshot identity | The hash includes configuration, context identity, observed count and retained prefix. Snapshot data are immutable. The hash identifies retained causal state, not discarded history, a corpus, model state, native-prefix agreement or authenticated document provenance. |

## Executed checks

The owner's fixture suite was executed independently at the hashes above. All **eight grouped checks** passed:

1. Independent count oracle and positive normalization: 1,093 prefixes.
2. Sequential prefix recomputation and score-before-observe.
3. Full-vector target/future-suffix noninterference: 40 suffix variants.
4. Explicit crop/document reset and removal of pre-crop context.
5. Masked loss still advances observed context.
6. Immutable snapshots and atomic invalid-input rejection.
7. Configuration/context validation, including Fraction smoothing conversion.
8. Native mixture endpoints, no-match identity and native-input validation.

Five additional reviewer assertions passed: histories with equal observed count and equal retained suffix have identical snapshots despite differing discarded history; differing observed counts change the hash; differing crop identity changes the hash; differing configuration changes the hash; and an independently reconstructed canonical hash equals the emitted state hash.

The two explicit malformed-mixture probes reproduced CC-R1 and CC-R2. The complete native comparison remains unexecuted by this reviewer.

## Limits and follow-up

This establishes small-fixture behavior of a causal comparator. It does not establish language cross-entropy improvement, fair end-to-end native integration, measured memory/latency, optimized complexity or selection of an architecture branch. The reference re-enumerates transitions rather than maintaining a production count index.

The parent and owner have been notified. Final disposition and replacement hashes will be appended after the owner's correction and a focused recheck.

## First closure after API correction

The owner normalized mixture weights safely to finite Python floats and added validation for public prediction records: positive normalized vocabulary-sized numeric probabilities, nonnegative integer counts, consistent match totals and orders, and agreement with the declared smoothed counts.

At the final hashes below, the reviewer reran all eight grouped cache checks and five repair-focused assertions. The numpy.longdouble weight now returns JSON-serializable Python floats. Huge Fraction weight, negative probabilities, inconsistent match totals and inconsistent zero-order metadata each raise ValueError. **CC-R1 and CC-R2 are closed.** No remaining blocker was found within the specified integer-fixture comparator scope.

| File at first closure | SHA256 |
|---|---|
| causal_ngram_cache.py | ce8f9e2c77166b0bdb7e066e19525d4000a1470f9ae0198085aef585a487b566 |
| test_causal_ngram_cache.py | 828531c4d8b2616be890164602f4e1cf506c7a589d6085ecab4c63d540cd3a50 |

Prediction validation establishes internal numeric coherence; it does not authenticate snapshot provenance or verify that a native model consumed the same prefix. Those caller obligations remain explicit.

## Numerical amendment and final closure

During its Chapter 13 proof audit, the owner identified an additional floating-point boundary and the coordinator authorized a narrow reopen. With the smallest positive Python float t, both native and cache vectors can equal (t, 1.0). The earlier two-product interpolation at weight one half evaluates the first coordinate as 0.5t + 0.5t = 0.0, violating positivity despite valid inputs. The cache vector is naturally attainable with vocabulary two, smoothing 2t and observed tokens (0, 1, 0); this is not an impossible fabricated state.

The final implementation anchors interpolation at the smaller positive operand and validates positivity and normalization of the returned mixture. **CC-R3, subnormal interpolation underflow, is closed.** The reviewer independently reproduced the old expression's zero, confirmed the repaired equal-vector case, ran 21 combinations of positive native vectors and endpoint/interior weights, and reran the expanded eight-group suite. Every repaired mixture in those checks contained finite positive Python floats and normalized within the declared tolerance. No model training was performed.

| Final reviewed file | SHA256 |
|---|---|
| causal_ngram_cache.py | 8836cf8ab645628499a6b438963d798c6c064cf832bb67df8a42a8ac32325732 |
| test_causal_ngram_cache.py | b22f37dc4fc9a9355683eec6d30f7b5a9471bc4b2a89a0178e2e3174c0abd23f |

The new result receipt is results/causal_cache_checks_v5_final.json; the earlier receipt remains unchanged. Its grouped JSON summary is byte-identical because group names and counts did not change. The final source/test hashes, rather than that summary alone, identify the expanded assertion scope. These final hashes supersede the first-closure hashes for this review.
