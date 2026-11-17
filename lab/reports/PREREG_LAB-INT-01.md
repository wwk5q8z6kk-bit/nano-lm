# PREREG LAB-INT-01 — Track A -> Track B Architecture Surgery & Progressive Hybridization

*Registered 2026-11-17 before running conversion ablations.*

## 1. Scientific Question
Having validated two Track A architectural wins (`X1_gated_vres`: `-4.15%` BPB + `96.1%` OOD induction; `X4_nanolm_v2_muon`: `-7.09%` BPB at `0.9566` + `-80%` KV cache), **how can Track A discoveries be transferred into an already-pretrained Dense GQA Transformer without destroying learned representations?**

## 2. Three Falsifiable Surgery Arms (`lab/integration/surgery.py`)
Starting from an unmodified pretrained Dense GQA checkpoint $M_{\text{pre}}$ (preserved untouched):
1. **`I0_naive_swap` (Naive Component Replacement Control):**
   - Replace non-anchor GQA layers (`DDDADDDADA` / `DDDADDDADDDADDDA`) with fresh `NextGenDeltaMixer(sharp_v=True)` layers and fine-tune with standard next-token CE.
   - **Hypothesis:** Causes catastrophic immediate BPB spike (`> +2.0` BPB at step 0) and fails to recover pretrained BPB under short adaptation budgets (`<= 5M` tokens).
2. **`I1_progressive_rad_kd` (QKV-Warm-Started 2-Stage Layerwise-MSE + Proper Top-K+Tail Logit KD):**
   - Initialize `NextGenDeltaMixer` `q, k, v, o` weights directly from pretrained GQA projections (`repeat_interleave` across KV groups), align mixer outputs via Stage-1 layerwise MSE $\lVert y_{\text{delta}}^{(l)} - y_{\text{attn}}^{(l)} \rVert_2^2$, then run Stage-2 Proper Top-64+Tail Logit KD from frozen $M_{\text{pre}}$.
   - **Hypothesis:** Dramatically outperforms `I0_naive_swap` (`>= 0.15` BPB lower after matched short adaptation), while quantifying any remaining short-budget conversion gap honestly (do NOT merge into Track B release unless lossless `<= +0.5%` BPB).
3. **`I2_zero_init_gated_vres` (Exact Function-Preserving Value-Highway Graft):**
   - Graft per-head input-dependent Layer-0 Value Highway $v^{(l)} \leftarrow v^{(l)} + \tanh(w_{l,h}^\top x) (v^{(0)} - v^{(l)})$ with $w_{l,h} = 0$ so **step-0 output logits are bit-identical (`0.000000` KL / `0.0000` $\Delta$BPB)** to $M_{\text{pre}}$, then adapt with Top-64+Tail KD.
   - **Hypothesis:** Zero step-0 capability loss (`KL = 0` exact) and strict validation BPB improvement or preservation.
