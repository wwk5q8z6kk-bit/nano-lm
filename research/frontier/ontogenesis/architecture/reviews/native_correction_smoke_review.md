# Independent review: native correction integration smoke

**Review date:** 2026-11-22. **Reviewer:** independent trainable-reference agent.

**Conclusion:** no material defect found in the fixed integration path. Actual native next-token learning, causal event delivery, predictions from the complete native output head, and revision-aware measurements are connected consistently. This is an execution and accounting result; it does not establish correction effectiveness, generalization, language quality, or an architecture advantage.

## Frozen sources reviewed

| Source | SHA-256 |
|---|---|
| `experiments/native_correction_smoke.py` | `a664645c79e0cbb4a7bc87784790f38364d7f77788b9fca951fe7eca3707f746` |
| `experiments/test_native_correction_smoke.py` | `94189081eb05dc3ba7300e829f9aef6585338ab44d2ed7cb3f7fb7491d349137` |
| `prototype/trainable_native.py` used by the bridge | `34f65e06d9251066363d3969fc47cfdb17a0fa5b1877e6da20087fb6651852ed` |
| `experiments/correction_protocol.py` used by the bridge | `f7339939059c8337a2257c9064c3073522b63fb5744cb6a464e7423c15b86087` |

The reviewer read the bridge, its complete test file, and the relevant training loss path. The supplied integration suite independently passed **9/9 tests**. No bridge or author-test source was edited during review.

## Executed independent replay

`checks/review_native_bridge_replay.py` preserves the additional reviewer check. It replays the fixed fixture directly through `TrainableNative.loss_and_grad` and `AdamW.step`, bypassing the bridge learner's observation method and `train_step` wrapper. It reconstructs the query and final value target, computes full-vocabulary cross-entropy directly from the logits, compares optimizer receipts and event-boundary state digests, and checks actual tensor differences against the optimizer's displacement inventory. Frozen snapshots use the original `NativeReference` forward implementation to check every recorded prediction.

| Check | Observed result |
|---|---:|
| Replayed optimizer updates | 16 |
| Matching before/after event optimizer-state digests | 16 |
| Actual tensor displacement arrays checked | 176 |
| Original-reference predictions checked | 23 |
| Maximum absolute error in direct cross-entropy calculation | 4.440892098500626e-16 |
| Maximum absolute logit difference from original reference snapshots | 0.0 |

These are observations on Python 3.12.14 / NumPy 2.3.5 with the fixed model seed 20261122. Equality of replayed update receipts and state hashes was checked in this runtime; cross-platform bitwise invariance is not claimed.

Run from the package root:

```bash
python -B experiments/test_native_correction_smoke.py
python -B checks/review_native_bridge_replay.py
```

The reviewer replay prints a JSON result with implementation and reviewer-script hashes. It does not save a checkpoint or benchmark result by default.

## Wiring and denominator assessment

| Contract | Review finding |
|---|---|
| Causal delivery | `observe` receives exactly the next delivered prefix. The coordinator owns the full manifest. A changed final-event target does not alter earlier recorded states, event receipts, or phase endpoints. |
| Current-event exposure | Only the current assignment is trained, twice. Historical delivered events are retained as an identity record and are not replayed. Initial observations and same-value revisions also consume their declared two updates. |
| Native loss | The input is the three-token query. Only the final shifted next-token value target is scored. Its value token is absent from model input, and loss normalization covers one scored target per update. |
| Native prediction | Argmax covers all eight output tokens of the tied native head. Non-value tokens become explicit abstentions. The reviewer checked the recorded logits through the original immutable forward reference. |
| State receipts | Event and phase receipts identify the corresponding actual optimizer state and delivered prefix. Inference leaves parameter/optimizer identity unchanged. Hashes identify state; they are not saved checkpoint payloads. |
| Endpoint semantics | Immediate acquisition follows each individual write. Horizon zero follows every write in the phase. Subsequent same-key revisions censor older events under `new_revision`. |
| Denominators | All five assignment events remain represented as eligible, censored, or not yet due for every horizon. Missing acquisition is distinct from failure; failed acquisition remains in primary eligible accuracy. |
| Execution scope | The fixed test consumes 16 documents, 48 input tokens, 16 scored target presentations, and 32 masked target positions. There are 23 inference queries totaling 69 query tokens. No test requires an accuracy threshold. |

The independently verified all-assignment partitions are:

| Phase-end horizon | Eligible | Censored | Not yet due | Total |
|---|---:|---:|---:|---:|
| 0 | 4 | 1 | 0 | 5 |
| 1 | 3 | 2 | 0 | 5 |
| 5 | 1 | 2 | 2 | 5 |
| 10 | 0 | 2 | 3 | 5 |
| 20 | 0 | 2 | 3 | 5 |

## Scope retained

The bridge has one handwritten integer-token sequence, eight events, one seed, and sixteen updates. It has no replay method, LP correction, checkpoint payload output, external dataset, held-out task, or architecture comparison. Its output does not constitute a G03b pilot result. Causal access separation is an audited interface and call-path property, not a Python security boundary. A broader native training or continual-learning experiment still needs its separately registered data, budgets, controls, and evaluation.
