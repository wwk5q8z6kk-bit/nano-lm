# Conditional causal copy: small, reversible research candidate

**Status:** experimental implementation and CPU correctness tests, **not** an adopted architecture or a measured language-model improvement. **Project:** Nano LM (unchanged).

## Why test this?

Nano LM's M21 result on held-out code showed useful improvement for copyable tokens but a substantial regression elsewhere; on equal training time a dense GQA baseline won overall (PR #194). More recently, the 14M Muon tournament (draft PR #203) found a seven-scalar value-residual change outperformed a costly depth-routing alternative at comparable throughput. These findings favor cheap, conditionally activated mechanisms over replacing a well-performing backbone.

**Hypothesis:** a small source-copy head, consulted only when the observed prefix contains relevant candidates, might preserve the dense model's general language performance while improving exact repeated-symbol prediction. This may fail because the extra head, training, and memory traffic could cost more than the achieved improvement. Pointer/copy models are prior art; no novelty is implied.

## Formal contract

Given tokens x[0:t+1] and *causal* hidden states h[0:t+1], the model predicts x[t+1] from the backbone's vocabulary logits. It optionally copies source token x[j] for 1 <= j <= t, keyed by its preceding context h[j-1]. Only the observed prefix is eligible—no targets or future tokens enter pointer selection.

Let alpha(t,j) = softmax_j(q(h[t]) · k(h[j-1])/sqrt(d_ptr)) over the last window eligible positions; g(t) = sigmoid(w_g h[t]+b_g) (forced zero with no eligible candidates). The combined distribution is:

p(y|x[:t+1]) = (1-g(t)) p_dense(y|x[:t+1]) + g(t) sum_j alpha(t,j) 1[x[j]==y].

The sparse loss computes only the true target's pointer probability and does **not** allocate another dense (B,T,V) tensor. Full-vocabulary pointer probabilities are available only for inference/debugging. Complexity is O(B*T*window*d_ptr) for pointers on top of the backbone; this is not free and has poor scaling if applied indiscriminately. The small initial gate bias avoids changing a pretrained backbone too strongly before the head learns; it is not a substitute for evidence-based calibration.

## Run local correctness tests

From this directory:

    python -m pytest -q test_causal_copy.py

Current local result: **8 passing CPU tests** for prefix-only access, copy alignment, distribution normalization, matching target-only/full losses, no-candidate fallback, gradient flow, shape guards, and high-gate numerical stability. No model-training or energy claims are made.

## Next experiment (not launched here)

Use an allowed/licensed real held-out text/code corpus and pretrained/checkpointed dense baseline **only after confirming data lineage**. Compare (1) dense alone, (2) dense plus parameter-matched generic head, (3) dense with always-on copy, (4) gated copy with calibrated gate, and (5) original M21 on equal-token and equal-wall-clock budgets. Test unseen strings/identifiers, repeated tokens, non-copyable tokens, OOD repositories, general BPB, peak VRAM, wall-time, source-copy precision, and p50/p95 latency; isolate the head's parameter and memory overhead. Use validation seeds for hyperparameters and fresh confirmation seeds. Do not silently alter any running preregistered experiment. Do not launch paid compute from this research slice.

## Integration choices

The exported CausalCopyMixture operates on base logits, input token IDs, and an independently computed causal hidden representation; it does not modify the dense backbone. If the source backbone cannot efficiently expose hidden states, count that additional cost or reject the candidate. A larger window costs linear work per token and should be evaluated against attention-based copying already available to the model.
