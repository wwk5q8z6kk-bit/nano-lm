# 17 — Corrected memory evidence, temporal scoring and protection theory

**V6 review date:** November 23, 2026. This chapter uses two separately pinned draft studies: G04/PR204 at`e1f56b1c4b36b4eb4f16cef4a9b4f5487b147f7c`, and G06/PR202 at`8b356f938644287eb78177a2299a78ffc02f0f05`. Their study files are absent at the frozen main research cutoff. They are evidence about research candidates, not integrated Nano LM runtime components.

The [independent source/theory review](reviews/v6_correction_theory_review.md) contains exact Git links, per-seed tables, source hashes and the complete audit. This chapter records the consequences for the architecture and decision ledger. The review recomputes 26 registered predicates without neural training, reconstructs world metadata and supplies deterministic counterexamples. No upstream raw result is edited.

## Executive findings

| Finding | What changes in V6 | What remains valid or open |
|---|---|---|
| Corrected narrow router fails N4/N5 | Withdraw the earlier narrow-router success claim | Later cooldown policies have separate favorable registered outcomes |
| Final predictions scored against historical targets | Mark `revise_next5/10` temporally invalid after repeated changes | Registered N/K/Q predicates use other fields and retain their checked outcomes |
| Cooldown has no expiry | Describe lifetime suppression of `(group, relation, value)` | Semantic epoch handling and B→C→B behavior need an explicit policy |
| Narrowing also changes fitting schedule | Attribute results to the composite policy | Mask-only attribution requires a sequential unmasked control |
| Corrected localized protection has strong bilinear retention | Use corrected entity-arm values and denominators | No native-LM or nonlinear retention result follows |
| Finite-alpha protection is full-rank attenuation | Replace exact-nullspace interpretation and “free fraction” wording | Exact preservation requires an exact operator and fixed features, or bounded residuals |
| Old address invariance is unmeasured | Require actual address/decoder drift receipts | This does not show a material error in reported retention percentages |

## 1. G04: repaired results and the remaining trade-off

The original narrow-router code used a union of permitted slots across revised groups. The repaired path applies each group/relation's own mask. Corrected N4 succeeds in only 1/5 seeds and N5 in 0/5. Their registered five-seed-mean predicates fail. Oracle scope restriction still offers a useful trade-off; it does not make the learned router successful by association.

The later cooldown studies have fresh specified seed sets. The code suppresses a group/relation/value after its first flag for the lifetime of the run. It is not time-based expiration. It reduces repeated actions, but can also miss later legitimate changes. The most useful paired results are:

| Comparison | Stable accuracy | Old revision | Interpretation |
|---|---:|---:|---|
| Typed ordinary router |71.16%|82.28%|Reference composite policy |
| Typed narrow + cooldown |77.02%|68.48%|About+5.87 pp stable,−13.80 pp revision versus ordinary router |
| Typed no router |82.44%|15.14%|Narrow+cooldown still has 5.42 pp stable deficit |
| EM no detector |72.07%|16.30%|Separate model/seed family |
| EM narrow + cooldown |71.47%|53.92%|About−0.60 pp stable,+37.62 pp revision; 6.4 of 12 changes missed |

K1–K5 and Q1–Q4 hold under their specified **mean** predicates. Some individual seeds do not meet those same bounds. “Within the 2 pp stable tolerance on these five seeds” is supported for the EM comparison; “zero measurable cost” is not an equivalence result and omits resource costs.

Narrow revision fits each flagged group separately with a fresh AdamW instance, while ordinary revision fits the combined batch. Mask, order, batching and optimizer-reset behavior therefore change together. A schedule-matched unmasked arm is needed before claiming that the mask alone causes the improvement. The measured composite-policy comparisons remain usable.

EM's absence of supplied shared-cluster labels is narrower than complete absence of latent structure: other address construction uses type-conditioned features, and reclustering re-reads stored creation-time facts. A “no replay” claim limited to optimizer training does not mean zero retained observations or reread cost. Report all those states and operations.

## 2. A remaining evaluator defect: prediction time and target time differ

Suppose a relation's truth is A initially, B beginning at phase 10, and C beginning at phase 20. An entity created at phase 11 belongs to the phase 10 creation cohort. The final model at phase 39 should answer C. The inspected `revise_next5` loop still compares that final answer with historical B.

Clipping the creation cohort before the next change does not fix the target, because the prediction is still made at the final phase. A perfect final-truth predictor can be marked wrong. On registered world 7802, that perfect predictor gets 135/161=83.851% under `revise_next5`, and 126/152=82.895% under `revise_next10`. These are scorer diagnostics, not corrected model accuracies.

| Intended question | Correct observation | Correct target |
|---|---|---|
| How quickly was change B learned? | Prediction at an explicit offset after B and before its supersession | Truth valid at that prediction time |
| How well does the final model answer entities born after B? | Final-time prediction for the declared birth cohort | Final truth, including later C |
| Did a particular revision survive? | Same revision followed until a declared horizon or supersession | That revision's value while it remains valid |

The registered narrow, cooldown and EM gates use `revise_old`, stable, exception and noise fields, so this defect does not change their audited pass/fail outcomes. `revise_old` selects the last still-active change and avoids this particular mismatch. Aggregate result files lack the individual predictions needed to reconstruct replacement `revise_next5/10` scores. Do not divide the old scores by the perfect-predictor ceiling or silently rewrite them.

V6's temporal reference uses explicit event identities, validity intervals and prediction timestamps to separate early adaptation from final-truth scoring. Missing eligible predictions remain incorrect; superseded revisions are counted under an explicit censoring policy rather than disappearing. Its repeated-change examples and receipt belong to [the validation chapter](06_VALIDATION_AND_UNRESOLVED_WORK.md). This is a reference repair available for integration, not a rerun or patch of PR204's historical experiments.

### 2.1 Reversions and suppression state

With a lifetime key `(group, relation, value)`, requests B, C, B lead to accept, accept, reject. The second B is a new semantic epoch even though its value matches a previous epoch. Cluster labels can also be reused after split/merge. A future router needs explicit scope and epoch identity, a reopening rule for new evidence, and tests for false initial flags, reversions and reclustering. Adding an epoch field is a proposed design; it needs evidence that the version policy itself does not create spurious repeated work.

## 3. G06: strong scoped retention and precise denominators

The study is a bilinear classifier with logits \(W_r u_e\), 16 latent types, 8 relations, 32 values, 40 phases of 64 new entities and three shown relations/entity. It fits new facts with 200 Adam steps and then performs address corrections. The localized mechanism filters the **actual decoder displacement after the optimizer step**, which is more relevant than projecting only a raw gradient.

The entity-protection repair keeps previously corrected entities protected throughout fitting even if they receive another correction later in the phase. Corrected files supersede the older entity-arm rows. Selected final means over seeds 8001–8003 are:

| Protection | Width | Old corrected facts | Old regular facts on corrected entities | New unshown-relation inference |
|---|---:|---:|---:|---:|
| None |24|21.470%|83.450%|99.690%|
| All old entities, alpha 1 |24|98.537%|99.957%|66.250%|
| Corrected entities, fixed |24|98.780%|100.000%|91.147%|
| Corrected relation |24|93.917%|92.097%|99.293%|
| None |96|21.597%|91.947%|99.823%|
| Corrected entities, fixed |96|99.877%|100.000%|98.917%|
| Corrected relation |96|99.637%|92.957%|99.857%|

The supported conclusion is that protection scope changes the stability/inference trade-off in this bilinear setup. No pass/fail reversal follows from substituting the corrected entity arms into the registered local predicates. Unlimited retention, nonlinear capability and native-language efficiency remain untested.

“624 corrections” denotes 624 target assignments. There are 16–19 same-value assignments and 583–589 distinct fact keys per seed. The final old-correction denominators are 272,277 and 270 keys whose **latest assignment** is at least 20 phases old. This is a final age-selected cohort, not survival over all 624 revision events. Repeated and no-op assignments affect age, so every future summary must state whether it tracks assignment events, semantic changes or current keys.

The replay arm retains 97.44% of old corrections and 98.35% new inference, but only 40.35% of old regular facts on corrected entities. Its growing replay set and data presentations are not compute-matched to no replay. A favorable correction endpoint cannot erase the side-effect or resource account.

## 4. Exact protection theorem and its assumptions

Let \(W\) be a relation decoder, \(u\) a protected fixed address and \(D\) the **committed** decoder displacement. Then

$$
(W+D)u-Wu=Du.
$$

If columns of \(A\) contain all protected addresses and \(P_0\) is the orthogonal projector onto \(\ker(A^\top)\), then a displacement \(D=BP_0\) satisfies \(DA=0\). This gives exact finite-step preservation of the specified bilinear logits in exact arithmetic. The optimizer producing `B` can be arbitrary. Fixed addresses, exact projection and complete protected constraints are premises.

The actual finite-alpha implementation uses

$$
P_\alpha=\alpha(AA^\top+\alpha I)^{-1},\qquad\alpha>0.
$$

In an eigenbasis of \(AA^\top\), its eigenvalues are

$$
\lambda_i=\frac{\alpha}{\sigma_i^2+\alpha}\in(0,1].
$$

Therefore it is full rank and generally non-idempotent. It preserves exact null directions and attenuates other directions; it does not project exactly onto a nullspace. With \(A=I_2\), alpha 1 gives \(P=.5I_2\). Its normalized trace is 0.5 while the nullspace dimension is 0. A proposed row displacement(1, 0) changes the protected logit at(1, 0) by 0.5 after filtering. The trace is a mean attenuation factor.

For a fixed address \(u=Ac\), a useful conditional bound is

$$
\|BP_\alpha u\|_2\le\|B\|_2
\max_{\sigma\ge0}\frac{\alpha\sigma}{\sigma^2+\alpha}\|c\|_2
\le\frac{\sqrt\alpha}{2}\|B\|_2\|c\|_2.
$$

The maximum occurs at \(\sigma=\sqrt\alpha\). Conditioning and the coefficient norm matter. Finite precision, repeated updates and moving addresses add further terms. This bound does not turn a small alpha into an accuracy guarantee.

## 5. Address motion and actual-state instrumentation

If the address changes by \(\delta u\), the exact identity is

$$
(W+D)(u+\delta u)-Wu=Du+(W+D)\delta u.
$$

A decoder filter controls the first term; it does not automatically control the second. The audited base model keeps one Adam optimizer over full address and decoder tensors. An inactive row can have zero current gradient while retained momentum still moves it. A scalar diagnostic with gradients(1, 0), rate 0.003, betas(0.9, 0.999) and epsilon 1e-8 gives second-step displacement−0.0020101747 despite the zero second gradient.

That is a counterexample to a general invariant, **not an estimate of drift in the G06 runs**. Previously corrected rows have 200 zero-gradient fit steps before writing, reducing the first moment by \(0.9^{200}\approx 7.055\times 10^{-10}\). Their later motion may be numerically negligible. Recently old clean rows used by a broader guard are a different case. The code neither enforces nor reports the required old-address invariant, so decoder-only causal attribution remains unverified without invalidating the observed 98–99% retention figures.

The next instrumentation must log actual protected-row displacement; frozen-address decoder-only logit change; representation-only change under the new decoder; total logit change; initial and remaining margins; and how optimizer state is frozen or reset. An enforced frozen-row control then provides a causal comparison.

For a nonlinear model, a Jacobian constraint removes only the first-order term. For example, \(f(\theta)=\theta^2\) at 0 has zero Jacobian, yet a 0.1 update changes the output by 0.01. Smoothness assumptions can bound the remainder; crossing nonsmooth boundaries needs separate analysis. The inspected nonlinear note has development mechanics checks, not a registered G06 run on its specified seeds. Optimizer failure does not certify global infeasibility.

## 6. Architecture actions and reopen conditions

| Action | Status in V6 | Next evidence |
|---|---|---|
| Preserve corrected-file precedence and draft-branch identity | Recorded with exact copies and 26-predicate reanalysis | New integrated commit if adopted |
| Replace stale narrow-router success interpretation | Corrected in current ledger | New schedule-matched router study |
| Separate historical targets and final prediction time | Reference temporal repair and adversarial examples | Integration into a future G04 evaluator plus per-query outputs |
| Version semantic scope and suppression epochs | Specified as candidate policy | Reversion, split/merge and false-first-flag tests in real router |
| Rename finite-alpha “projection” to soft attenuation | Mathematical scope corrected | Actual residuals, conditioning and cumulative margins |
| Require fixed-address evidence or a joint drift bound | Added to update contract | Measured old-row trajectories and frozen-row control |
| Count every stored observation, replay presentation and fit | Required reporting boundary | Wall-time/memory/cost measurements on a declared workload |
| Establish native-model benefit | Open scientific gate | Tuned native baseline, legitimate held-out tasks and long horizons |

The corrections make the research claims narrower and more actionable. They do not replace missing model experiments with proof language. The full raw and reviewed record remains available in the source manifest and audit bundle.
