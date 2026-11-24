# Independent second review of the V6 full architecture

**Reviewer scope:** sections 3.2, 3.3, 4.1, and 7 of 00_FULL_ARCHITECTURE.md. No edits were made to that file.

**Inspected document SHA-256:** 53a2a9aae403d04113b0083852be75493412548b468af72b509f2a9ac58bebbe.

**Result:** no blocking mathematical or scope error found in these sections. The numerical inventory checks and qualifications below pass. This is a review of source shapes, algebra, and claims; it is not checkpoint verification, training replication, or measured serving-memory validation.

## Numerical inventory

The historical initializer inspected is:

research/frontier/ontogenesis/tpu/train_jax.py at commit 2791a216281a29aeaa3ec7720a7520c3a9952892.

Its SHA-256 is f01aea2d45e5d7e123f56dd99bd7e08738a97ed8b387bbbf38f8ef4f7d6e5d41, matching the prefix recorded for the upstream run. The initializer assigns qn and kn one vector of length dh each; they are broadcast across heads. Thus adding 2dh, rather than 2d or 2Hdh, is correct.

| Component | Scalars per block |
|---|---:|
| Q | 1,638,400 |
| Output projection | 1,638,400 |
| K | 327,680 |
| V | 327,680 |
| Fused gate/up projection | 8,847,360 |
| Down projection | 4,423,680 |
| Residual RMS gains | 2,560 |
| Query/key RMS gains | 128 |

Embedding contributes 41,943,040, and the final RMS gain contributes 1,280. The total is exactly **386,062,080**.

The GQA KV shape is two arrays, 20 logical layers, batch 1, 4 KV heads, 512 positions, and 64 coordinates per head. At two bytes per element this is **10,485,760 bytes = 10 MiB**. The document correctly labels it a hypothetical cache tensor allocation, not measured memory of the full-prefix evaluator. Shared weights do not identify the logical cache state of separate layers.

For the NumPy reference, the separate full-head formula gives **15,208,704** parameters with eight unique blocks and **11,798,784** with four unique blocks reused over eight logical layers. The document correctly avoids calling this a parameter-matched efficacy comparison.

The token exposure arithmetic is also exact: 234,950 × 64 × 512 = **7,698,841,600** scored token presentations, or approximately **19.94198 presentations per parameter**. Microbatching partitions the declared batch; multiplying by the microbatch count again would double count.

## Formal statements and boundaries

### Actual updates and fixed representations

Section 4.1 correctly identifies the installed optimizer displacement as the object requiring protection. It also distinguishes an unenforced old-address invariant from demonstrated effect magnitude. The 200-zero-gradient-step attenuation .9^200≈7.0551×10^-10 is correct. This does not by itself bound normalized Adam displacement, but the document does not claim that it does.

### Causal execution

The induction in section 7.1 is valid given fixed parameters, prefix-only layer inputs, prefix-derived state, and correct reset/cache ownership. It does not certify an adapter with extra side channels; the document states the necessary scope.

### Representation transport

For an invertible map T on the relevant representation space, D'=D∘T^-1 and E'=T∘E yield D'(E'(x))=D(E(x)) by substitution. Whole-network tying, normalization, positional transformations, and cached-state compatibility are properly separated from this elementary identity.

The winner-margin statement is correct: if all logits move by at most ε in infinity norm and the original margin exceeds 2ε, the winner remains unchanged.

### Nonlinear and finite-horizon protection

Taylor's decomposition in section 7.3 is correct with the stated local Hessian condition. For a vector-valued output, a quantitative implementation must specify the norm and corresponding uniform component/operator remainder bound. No numerical certificate is asserted in the document, so this is an implementation requirement rather than a mathematical defect.

The finite-horizon statement is sufficient when the protected meaning is fixed and the actual margin losses are bounded at each step. Failure of that sufficient bound cannot establish forgetting. The document states both qualifications.

### Finite-α attenuation

The spectrum α/(σ_i²+α) is correct. Finite α>0 gives full rank and preserves exact null directions; supported positive-eigenvalue directions are attenuated rather than removed. trace(P)/m is correctly interpreted as average attenuation. The proposed exact-projector alternatives require a separate numerical definition, as the document states.

### Counterfactual context dependence

The bound .5[p(y_A)+p(y_B)]≤.5 follows because y_A and y_B are distinct and probabilities sum to at most one. A deterministic predictor that returns the same answer under both prompts has joint correctness zero. The premise is identical output behavior when the context is ignored, including reset/run-order conditions. The document correctly declines to assert an independence-based stochastic joint bound.

A high joint score only establishes the sensitivity needed for those pairs; it does not prove broad reasoning. This scope is correct.

### Storage and edit costs

The finite-state counting argument b≥N log2K is valid for lossless encoding of every arbitrary assignment, under the stated reliable-bit assumptions.

The edit example is exact. Combining y−x≥3 with x−(2/5)y≥μ implies x≥2+(5/3)μ and hence y≥5+(5/3)μ. For μ≥0 both coordinates exceed their starting values, so L1 movement is at least 4+(10/3)μ. Equality is achieved at those lower bounds. At μ=0, (x,y)=(2,5) gives the protected margin zero, a tie rather than a positive-margin guarantee.

## Reproducible receipt

Run:

~~~bash
python v6_review/correction_theory/architecture_math_audit.py
~~~

The script records the document and historical-source hashes, counts parameters from the initializer's named tensor shapes, computes cache bytes independently from the full tensor shape, checks the reference counts and token exposure, and verifies selected edit solutions with exact rational arithmetic. Its output is architecture_math_audit_results.json.

The formal reasoning above is a human-readable mathematical review, not a claim that enumerating a few numeric examples proves a universal theorem.

