# PREREG LAB-COG-01 — Unified Epistemic Belief-State, Counterfactual World Modeling & Exact Evidence Pointer vs Trajectory Imitation

*Registered 2026-11-17 before running `LAB-COG-01`.*

## 1. Scientific Motivation (Unifying Nano LM `H6`, Open Brief, and Software-Native Cognition)
Across both `nano_ai/` H6 (`artifacts/nano_h6/H6_TERMINAL_DECISION.json`, where row-level audit of 5,000 decisions showed `184/214` absence errors and `53/60` uncertainty errors came from **epistemic state selection**, while `714/718` supported errors came from **exact span boundary binding**) and agentic software engineering (`SWE-EVO` / partially observable repositories), standard `Transcript -> Action/Token` models suffer three structural failures:
1. **Conflating lossy semantic state with lossless exact evidence:** Regenerating unseen identifiers/constants through a fixed vocabulary softmax corrupts unfamiliar values unless backed by exact source-span pointers (`ECBS` hypothesis).
2. **Transcript interference on belief revision:** Conditioning on raw growing transcripts retains attention mass on early misleading symptoms even after a later diagnostic test falsifies that hypothesis.
3. **Imitation blindness to counterfactual risk & ambiguity:** Imitating only successful trajectories never teaches *why* premature patching under intent/state ambiguity fails, nor separates *intent uncertainty*, *state uncertainty*, and *verification sufficiency*.

## 2. Four Matched-Parameter Neural Arms (`~1.6M` Params, `NanoLM-v2` `DDDA` Backbone, `n=3` Seeds)
Evaluated on **in-distribution (`4–5` module)** AND **unseen OOD (`7–9` module, unseen identifier vocabulary & unseen `4`-digit constants `[1000..9999]`)** partially observable software repositories (`lab/cognition/swe_env.py`):
- **`C0_transcript_imitation`:** Standard trajectory SFT `(goal + full_transcript -> action + target + generated_value)`.
- **`C1_obs_world_model`:** `C0` + next-observation token prediction $\mathcal{L}_{\text{obs}}(o_{t+1} \mid h_t, a_t)$ (CWM-style).
- **`C2_latent_loop_only`:** `C0` + Huginn-style randomized-depth latent recurrence (`K in {1,2,3}` passes) without explicit belief/counterfactual/pointer supervision.
- **`C3_ebwm_counterfactual`:** Persistent 8-slot Latent Belief Workspace $Z_t = U_\theta(Z_{t-1}, o_t, a_{t-1})$ + Factored Epistemic Head `(u_intent, u_state, u_verify)` + Decision-Oriented Counterfactual Transition Head $T_\theta(Z_t, a)$ + Exact Source-Span Pointer $P_\theta(Z_t, E_t)$.

## 3. Explicit Success & Falsification Criteria
- **Primary Metric:** **OOD Verified Resolution Rate** (`%` of unseen `7–9` module episodes solved cleanly with post-patch verification and zero unrecovered regressions).
- **Kill Rule:** Reject `C3_ebwm_counterfactual` if it fails to beat both `C0` and `C1` by `>= +8.0 pp` on OOD Verified Resolution across `n=3` seeds, or if `C2_latent_loop_only` matches `C3` without explicit belief/counterfactual/pointer supervision.
