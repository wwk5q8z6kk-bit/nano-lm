# Independent review — tiny trainable native reference

**Verdict:** the reviewed implementation is suitable as a tiny engineering reference within its declared scope. Two reproduced storage-alias defects were corrected before this verdict. No unresolved material derivative, optimizer, snapshot or checkpoint-contract defect was found in the reviewed paths. This is a source review with small executable checks, not a proof for arbitrary floating-point inputs or a language-model result.

## Reviewed source identities

| File | SHA-256 at final review |
|---|---|
| `prototype/trainable_native.py` | `34f65e06d9251066363d3969fc47cfdb17a0fa5b1877e6da20087fb6651852ed` |
| `prototype/test_trainable_native.py` | `ec15d6a167102362ed256c8f118ca047f97a52e6ad0fbdb5596833bd54b4e3bf` |

The review used NumPy 2.3.5. The [training contract](../10_TRAINING_AND_CONTROLLED_UPDATE_CONTRACT.md) explains the implementation's intended boundary. Final release receipts belong to the separate [validation record](../06_VALIDATION_AND_UNRESOLVED_WORK.md).

## Findings and repairs

| Reproduced condition | Before the repair | Final behavior |
|---|---|---|
| Two parameter names share an array or overlapping views | Assigning the q array to k passed validation. Sequential candidate commits changed q again through k; the maximum actual-versus-reported displacement error was `0.005999999321177674`. Serialization removed the alias while preserving serialized values and digest; a subsequent identical step could diverge. | Pairwise storage overlap is rejected before operations use the parameters. Logical sharing remains one named tensor used multiple times. |
| One parameter array overlaps itself through writable strides | A zero-stride 8 × 8 q array passed the first, pairwise-only repair. Alternating signed gradients gave an actual-versus-reported displacement error of `0.005999999520000035`. | Every parameter must be C- or F-contiguous, which excludes these internally overlapping views, in addition to being disjoint from every other named parameter. Unsupported noncontiguous layouts are rejected rather than silently copied. |
| Finite candidate arrays have an overflowing diagnostic norm | The author separately reproduced a very large finite learning rate for which receipt construction could fail after mutation. | Candidate arrays and all receipt values are validated before committing parameters, moments or step. The independent final suite rerun includes the atomic-rejection regression. |

The independent storage check covers six forms: identical arrays, transposed overlap, contiguous overlapping slices, an embedding-row view, zero strides and overlapping nonzero strides. Each was rejected by eight entry points: forward, loss/backward, direct optimizer step, training step, snapshot, checkpoint save, state digest and new optimizer construction. All **48 rejections** preserved raw parameters, moments, step number and prior displacement arrays and created no checkpoint file. The model had already taken a valid step before alias injection, so the check covered nonempty optimizer history.

The published suite contains six forms across seven operations, or **42 rejection checks**. These counts describe different check sets; neither is a sample-size claim about model behavior.

## Derivatives and optimizer semantics

The analytical paths agree with the stated equations: RMSNorm differentiates through its epsilon-containing denominator; SiLU and the gated feed-forward path use the product rule; attention softmax uses its row Jacobian; masked future edges have zero probability and derivative; and RoPE's reverse map uses the transposed rotation. Shared-block gradients accumulate over logical uses. The tied table receives both output-head and input-lookup contributions, including repeated-token accumulation.

The final suite passes **17 of 17 groups**. Central finite differences cover 62 directions over 31 named tensors across the dense and shared configurations. The largest absolute discrepancy is `6.803216656692257e-09`. A separately untied equivalent graph verifies that shared gradients equal the sum of its per-use gradients. These are independent numerical checks of sampled directions, not an exhaustive derivative proof.

AdamW clips the global raw-gradient norm, computes bias-corrected first and second moments, and adds decoupled decay to the complete proposed displacement. The explicit reference policy decays every distinct tensor, including norm gains and embeddings. The stored displacement is computed as candidate minus old stored parameter, so its inventory includes moment history, preconditioning, decay and addition rounding. It is not presented as a raw-gradient or nullspace guarantee. The first-step known-answer test and direct array comparisons support that interpretation.

Input, state, candidate and receipt checks precede parameter commits. The atomicity claim covers rejected in-process operations in the stated API. It does not cover concurrent mutation, process termination, hardware failure or arbitrary invalid memory objects.

## Snapshots and checkpoint boundary

Inference snapshots copy and freeze the learned arrays and use the original reference's configuration/parameter digest. Old snapshots retain their own semantics; an updated snapshot rejects a cache tagged with a different model version. Training does not mutate existing inference snapshots.

Checkpoints retain model configuration, tensor names, float64 values, Adam moments, step and optimizer hyperparameters. Loading validates metadata and tensor contracts without enabling pickle and verifies the internal digest. Exclusive creation protects an existing checkpoint path. The code does not promise crash-atomic publication or authenticated provenance.

The suite establishes bitwise parameter, moment and receipt equality for its default **C-contiguous** five-step resume fixture in the recorded environment, with identical subsequent inputs. An additional independent F-contiguous probe confirmed exact realized-displacement arrays, equal state digests at reload and equal state digests after one identical next step. Some diagnostic norms differed by a few ulps after reload canonicalized parameters to C order. This is a declared precision boundary: the review does not establish universal bitwise receipt or trajectory invariance across layouts, numerical libraries or environments. No sampler exists; future document order remains the caller's responsibility.

## Limits and reproduction

The deterministic cyclic-token learning fixture exercises the gradients and optimizer on training data. It establishes no held-out language quality, data efficiency, continual-learning benefit, architecture advantage, device throughput or native-data result. The correction protocol and native learning comparison remain distinct integration and scientific questions.

From the architecture directory, the final source and behavioral checks are reproducible with:

```bash
python3 -B prototype/test_trainable_native.py
```

The command prints its report and uses temporary checkpoint locations. This review added no result receipt inside the package and made no upstream research or remote repository change.
