# Design note: Surprise-Gated Recurrent Depth (SGRD). Track A original hypothesis, PROPOSED, untested

## 1. Derivation
The gated delta rule keeps a per-head associative memory $S_t \in \mathbb{R}^{d_k\times d_v}$ and updates it by

$$S_t = a_t\,(I - \beta_t k_t k_t^\top)\,S_{t-1} + \beta_t k_t v_t^\top .$$

This is exactly one step of online gradient descent, with step size $\beta_t$ and decay $a_t$, on the in-context regression loss
$\mathcal{L}_t(S) = \tfrac12\lVert S^\top k_t - v_t\rVert^2$, because $\nabla_S \mathcal{L}_t = k_t (S^\top k_t - v_t)^\top$.
The residual $e_t = v_t - a_t S_{t-1}^\top k_t$ (equal to $u_t/\beta_t$ in the WY form already computed in `LinearMixer`) is the memory's **prediction error** for token $t$. It is the information-theoretic *surprise* of the new association under the current memory.

Applying a weight-shared core $r$ times (recurrent depth) means each token's representation is refined $r$ times, and each pass re-queries and re-writes the memory. Seen as optimization (test-time-training view), looping amounts to taking more solver steps on the in-context objective.

**Hypothesis.** Extra passes are only useful for tokens whose association is *not yet explained* by memory. So the halting decision should depend on surprise $\lVert e_t\rVert$, which the delta rule computes for free, rather than only on the hidden state:

$$p^{halt}_{t,i} = \sigma\!\big(w^\top h_{t,i} + \gamma \log \lVert e_{t,i}\rVert + b\big), \qquad
y_t = \sum_{i=1}^{r} \Big(p^{halt}_{t,i}\prod_{j<i}(1-p^{halt}_{t,j})\Big)\, h_{t,i}$$

The model is trained PonderNet-style, with expected-output mixing and a KL to a geometric prior that sets compute pressure. At inference a token exits once its cumulative halting mass exceeds τ.

## 2. Predictions (falsifiable)
1. At matched **expected inference FLOPs**, SGRD has lower BPB than fixed-loop R2 and than hidden-state-only halting (ablation γ=0).
2. Mean loops per token correlate with token difficulty: higher on code and math, and on the first occurrence versus repeats of a span (copy probe).
3. If $\gamma \to 0$ after training (the model ignores surprise), the hypothesis is rejected even if quality improves.

## 3. Cost accounting
Dense training computes all $r$ passes, so training FLOPs equal fixed-$r$ looping. The inference saving is *expected loops / r*. It only turns into wall-clock gains with token-sparse execution, which is measured separately and not assumed.

## 4. Gate
Implemented after LAB-ARCH-02 round 1, and only if a delta hybrid (H2/H3) is within 1% of T0 **and** looping shows any iso-FLOP benefit. If looping shows no iso-FLOP benefit, SGRD is reframed as compute-saving (fewer loops at equal quality) rather than quality-raising.
