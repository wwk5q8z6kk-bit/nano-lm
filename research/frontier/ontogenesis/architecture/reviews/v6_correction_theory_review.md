# G04/G06 corrected-evidence and theory audit for the V6 continuation


**V6 package relocation:** exact upstream source bytes are unchanged. Packaged read-only audit runners are in `analysis/reconcile_upstream_v6.py`, `analysis/reconcile_correction_v6.py` and `analysis/reconcile_context_v6.py`; their stdout is compared with saved audit results by the V6 release gate. Original review workspace commands below describe the audit session.

**Date:** 2026-11-23. **Classification:** source audit, independent arithmetic, deterministic world reconstruction, and synthetic mathematical diagnostics. **No neural training or upstream modification was performed.**

## 1. Findings that should change the architecture record

The corrected results support continued investigation of scoped revision and localized protection. They do not establish a general continual-learning guarantee, a working language-model mechanism, or a cost-free implementation.

| Finding | Evidence status | Architecture consequence |
|---|---|---|
| G04's originally successful narrow-router predictions N4/N5 are withdrawn after the revision-mask repair. | Corrected raw data and independent gate recomputation agree: N4 fails, 1/5 seeds; N5 fails, 0/5. | Preserve the withdrawal explicitly; do not inherit the original “under 3 points” claim. |
| The later cooldown studies have separate favorable results. | K1–K5 and Q1–Q4 hold on their specified five-seed means. | Record these as separate composite-policy experiments with their revision losses and remaining misses. |
| A remaining G04 repeated-update scoring defect affects revise_next5/10. | Exact source inspection plus perfect-final-truth diagnostic on the registered worlds. | Require temporal target validity and prediction-time recording in the next evaluator. This defect does not change the audited N/K/Q gates, which use other fields. |
| The “narrowing increases false alarms” explanation was itself withdrawn by a later diagnostic. | Corrected normal versus corrected narrow: 34.6 versus 32.8 false flags per seed. | Do not preserve that causal story merely because it appears earlier in the correction notice. |
| G06's corrected entity protection retains a strong bilinear result. | Corrected m=96 entity arm: 99.8767% old corrections, 100% old regular facts on corrected entities, 98.9167% new unshown-relation inference. | Replace the earlier entity-arm numbers and retain the synthetic/bilinear scope. |
| Finite-α protection is soft attenuation. | Direct spectral derivation and a two-dimensional counterexample. | Rename trace(P)/m as mean update attenuation; exact invariance needs explicit additional conditions. |
| Fixed protected addresses are not enforced or measured in the G06 fit loops. | Persistent Adam state, zero-gradient rows, no old-row restoration. | Add actual address-displacement instrumentation. This audit does **not** demonstrate a material effect in the reported runs. |
| Neither draft study is integrated at the frozen research cutoff. | Complete 4,594-entry recursive research tree, not truncated; no relevant study paths. | Cite pinned draft PRs separately from implemented runtime evidence. |

All raw arithmetic in this review is reproduced by [independent_audit.py](../analysis/reconcile_correction_v6.py). Its output is [independent_audit_results.json](../results/correction_audit_v6.json). These files are audit diagnostics, not new experimental model results.

## 2. Immutable provenance and integration status

| Evidence source | Pinned revision | Status observed |
|---|---|---|
| Frozen research cutoff | 3a38b5f1529a718838b5bc83a010574bfa9a4c6e | Complete recursive tree inspected. Relevant G04/G06 study files absent under research/frontier/ontogenesis. |
| G04, PR204 | e1f56b1c4b36b4eb4f16cef4a9b4f5487b147f7c | Draft, open, unmerged; 28 commits at inspection. |
| G06, PR202 | 8b356f938644287eb78177a2299a78ffc02f0f05 | Draft, open, unmerged; 8 commits at inspection. |

The bounded source collection contains **67 exact files, 479,600 bytes**. Every retrieved file matches both its Git blob SHA-1 and the SHA-256 recorded in [source_manifest.json](../evidence/upstream_v6/correction/source_manifest.json). PR metadata, histories, and cutoff-tree findings are in [source_context_complete.json](../evidence/source_manifest_v6.json). A frozen file's presence in a draft branch is evidence of its contents, not evidence that the running project uses it.

Related predecessor evidence **is** already present at the research cutoff: correction_entity.py, correction_interference.py, correction_margins.py, correction_optimizer.py, and their correction-interference result directory. In particular, correction_interference.py has Git blob cb8386a62380499fc94ffa22bf990049acb1419e and constrained.json has blob 8c5e00cc0772842da890e200321744621f3de3b5 at both the cutoff and the pinned G06 branch. The baseline is therefore inherited evidence, while the new nullspace/local-protection and G04 studies remain separate draft additions. [Cutoff predecessor inventory](../evidence/upstream_v6/correction/cutoff_predecessor_paths.json)

The histories corroborate the documented sequence:

| Study | Protocol/scorer commits | Result commit |
|---|---|---|
| G06 durability | 9a6db1d / bc56027 | dd8923f |
| G06 local protection | 5adc2b0 / b916aba | 298e565 |
| G06 entity-protection correction | Existing protocol, same seeds | 8b356f9 |
| G04 narrow | cf11f8e | 42520d6, subsequently superseded by correction 19e4258 |
| G04 cooldown | 071587e | ec55a21 |
| G04 EM cooldown | b65154b | e1f56b1 |

Commit order corroborates source publication order. It cannot by itself establish that no unrecorded exploratory run occurred before a protocol commit. The disclosed development seed is 1 for the G04 cooldown protocols; confirmatory sets are 7701–7705 and 7801–7805 respectively. [G04 correction notice][g04-corrections] [Cooldown protocol][g04-cd-protocol] [EM cooldown protocol][g04-em-protocol]

## 3. G04: what the corrected narrow-router result actually says

The published repair addresses stale repeated-change accounting, phase-free missed-change keys, age-mismatched exception subsets, per-group revision masks, override-store leakage into decoder targets, stale mid-phase predictions, and missing committed runner settings. The exact source retains the original results and places corrected results under results_v2. The runner makes the arm settings inspectable. [Correction notice][g04-corrections] [Runner][g04-runner]

The decisive mask fix is in threeway.py: revision facts are grouped with their **flagged group key and relation**, and each group receives its own allowed-slot mask. The previous union mask admitted slots belonging to another revised pair. The corrected implementation therefore invalidates the old narrow-router success claim; its new raw results are the relevant comparison. [threeway.py][g04-code]

### Corrected narrow study, seeds 7501–7505

Percentages below are seed means, rounded for presentation. “Stable shared” excludes the private-table contribution diagnostically; “Stable full” is the deployed sum of both reads.

| Arm | Stable full | Stable shared | Old entities revised | Exceptions kept | Noise resisted |
|---|---:|---:|---:|---:|---:|
| No router | 82.99% | 83.56% | 19.57% | 37.10% | 92.18% |
| Oracle, ordinary revision | 68.65% | 71.76% | 99.71% | 99.85% | 90.40% |
| Oracle, share ≥0.2 | 76.79% | 80.94% | 92.00% | 99.85% | 95.11% |
| Oracle, share ≥0.35 | 78.01% | 82.46% | 75.88% | 99.85% | 94.40% |
| Three-way router | 72.51% | 75.22% | 81.50% | 85.24% | 91.47% |
| Three-way, share ≥0.2 | 74.12% | 76.76% | 78.50% | 89.94% | 92.53% |
| Three-way, share ≥0.2, private table ×16 | 74.59% | 74.81% | 81.28% | 85.78% | 90.84% |

Source: corrected per-seed files and [narrow verdict][g04-narrow-verdict]; independently recomputed.

| Registered predicate | Corrected verdict | Seeds meeting predicate |
|---|---|---:|
| N1: oracle share .2 improves shared stable accuracy by ≥6 pp | Holds | 5/5 |
| N2: oracle share .2 old revision ≥88% | Holds | 5/5 |
| N3: share .35 decreases old revision and does not decrease shared stable accuracy relative to .2 | Holds | 5/5 |
| N4: router share .2 improves shared stable accuracy by ≥4 pp, with old-revision loss ≤10 pp | **Fails** | **1/5** |
| N5: narrow router with large private table is within 6 pp of no-router stable accuracy and old revision ≥60% | **Fails** | **0/5** |

The N4 stable gain is only **1.544 pp**. The N5 stable deficit is approximately **8.406 pp**. Favorable oracle scope results and failed learned-router predictions must remain separate. [Narrow protocol][g04-narrow-protocol] [Narrow scorer][g04-narrow-scorer]

The correction notice initially suggests that narrowing raises repeat alarms. Its later diagnostic rejects that explanation: corrected ordinary revision has 34.6 false flags per seed, corrected narrow revision 32.8; repeated false flags are 13.8 and 12.6 respectively. The apparent increase came from comparing corrected output against defective old code. Re-firing is common in both settings; this is not evidence that the mask causes it. [Repeat diagnostic][g04-repeat] [Correction notice][g04-corrections]

A small documentary count also needs precision: the notice's “9 registered seeds” is **9 repeated-change events across 8 seeds**, namely 7201, 7203, 7302, 7403, and 7601–7604. Seed 7602 contributes two repeat events. This audit reconstructed the generator draws to check the distinction.

## 4. G04: cooldown is useful under a bounded, explicit trade-off

The code's cooldown is **permanent suppression after one flag for the same (group key, relation, value)**. It has no elapsed-time expiry. The protocol name should not imply a time-based retry policy. [threeway.py][g04-code]

### Reported-policy comparisons

| Study / arm | Stable rule | Old revision | Exceptions kept | Noise resisted | Flags / seed | False flags / seed | Missed changes / 12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Typed: no router | 82.44% | 15.14% | 34.13% | 90.57% | 0 | 0 | 12 |
| Typed: ordinary three-way | 71.16% | 82.28% | 86.79% | 90.03% | 52.6 | 28.2 | 1.2 |
| Typed: ordinary + cooldown | 72.70% | 79.56% | 89.01% | 91.56% | 25.0 | 14.0 | 1.0 |
| Typed: narrow | 72.88% | 82.33% | 86.13% | 90.31% | 63.8 | 39.0 | 1.0 |
| Typed: narrow + cooldown | 77.02% | 68.48% | 94.71% | 92.65% | 22.2 | 11.8 | 1.6 |
| EM: no detector | 72.07% | 16.30% | 99.17% | 88.44% | 0 | 0 | 12 |
| EM: narrow detector | 66.88% | 60.44% | 90.21% | 87.29% | 43.8 | 30.8 | 5.6 |
| EM: narrow + cooldown | 71.47% | 53.92% | 96.27% | 89.43% | 16.6 | 11.0 | 6.4 |

Typed and EM rows use **different seed sets and different baseline policies**. They are not a randomized comparison of typed versus learned addresses. [Typed cooldown result][g04-cd-result] [EM cooldown result][g04-em-result]

The useful paired effects are:

- Typed narrow + cooldown versus ordinary router: stable accuracy improves about **5.866 pp**, while old revision falls **13.798 pp**. Versus no router, the stable deficit is **5.420 pp**, with seed deficits **4.59–6.85 pp**.
- Typed narrow + cooldown versus narrow alone: stable accuracy gains **4.136 pp**, old revision loses **13.850 pp**.
- EM narrow + cooldown versus EM no-detector: old revision gains **37.620 pp**, while stable accuracy decreases **0.602 pp**. Every seed's stable result is lower by **0.07–1.26 pp**.
- EM narrow + cooldown versus EM narrow alone: stable accuracy gains **4.592 pp**, old revision loses **6.518 pp**. One seed loses **19.95 pp** of old revision.

These are arithmetic contrasts from stored, four-decimal per-seed scores. No equivalence test, precision interval, or power analysis establishes zero stable cost. “Within the preregistered 2 pp tolerance on these five seeds” is supportable; “no measurable cost” is too broad. Hardware, memory, optimizer-state, and data-revisit costs were not measured in these files.

### Exact cooldown predicates

| Predicate | Requirement on five-seed mean | Mean verdict / seed count |
|---|---|---|
| K1 | Ordinary cooldown flag count ≤0.6 × ordinary router | Holds; 5/5 |
| K2 | Ordinary cooldown misses ≤0.5 more changes and absolute old-revision difference ≤5 pp | Holds; 3/5 |
| K3 | Narrow cooldown stable accuracy ≥ordinary router +3 pp | Holds; 5/5 |
| K4 | Narrow cooldown exception retention ≥ordinary router +3 pp | Holds; 5/5 |
| K5 | No-router stable accuracy minus narrow cooldown stable accuracy ≤8 pp | Holds; 5/5 |
| Q1 | Absolute EM cooldown versus no-detector stable difference ≤2 pp | Holds; 5/5 |
| Q2 | EM cooldown old revision ≥no-detector +25 pp | Holds; 4/5 |
| Q3 | EM cooldown stable accuracy ≥narrow alone +2 pp | Holds; 5/5 |
| Q4 | EM cooldown exception retention ≥93%, absolute noise difference versus no-detector ≤2 pp | Holds; 3/5 |

The scorer's **per-seed K2 diagnostic** permits one extra miss, whereas the **mean predicate** permits 0.5. The seed counts are descriptive; they are not a separate rule requiring four of five successes. The independent audit reproduces these exact conventions. [K scorer][g04-cd-scorer] [Q scorer][g04-em-scorer]

### Causal and implementation boundaries

1. **Mask and optimization schedule change together.** Ordinary revision trains the concatenated revision batch in one fit. Narrow revision runs one fit per group. Every fit constructs a fresh AdamW optimizer. A sequential-unmasked control is needed to identify the mask's effect separately from order, batching, and optimizer resets. This does not invalidate the full-policy comparisons; it limits the attribution.
2. **Typed scope selection uses the finite query universe.** allowed_slots aggregates read weights over preconstructed queries for all entities, including future entities in typed runs. It is not an online mechanism with unknown future query addresses.
3. **EM learns the shared assignment, not every component of the feature pipeline.** Private addresses are still constructed by addresses("typed",8): all eight slot indices are randomized, but their weights are retained from a type-conditioned address construction. Label matching and all creation-time observations are retained for reclustering. “No supplied shared cluster labels” is narrower and more accurate than “no latent structure anywhere.”
4. **“No replay” refers to optimizer replay.** EM re-reads stored creation-time facts during reclustering every five phases. Those retained observations, assignments, address arrays, table values, and optimizer state belong in the resource ledger.
5. **A stable numerical group label is not a versioned semantic identity.** Cluster split/merge or a later rule epoch can reuse the same group key.

These observations follow directly from [threeway.py][g04-code], [address construction][g04-addresses], [reclustering][g04-recluster], and [label matching][g04-match].

## 5. Remaining G04 evaluator defect: final predictions versus historical targets

### Exact counterexample

Let a type/relation rule be A initially, B from phase 10, and C from phase 20 onward. Consider an entity born at phase 11 with this relation unshown.

At final phase 39, truth_at returns **C** for that entity. A perfect final-truth predictor answers C. The current revise_next5 loop nevertheless includes the phase-11 entity in the cohort of the phase-10 change and compares its final prediction with **B**. It scores the perfect final prediction as wrong.

The repair clips the creation cohort before the next rule change. That does not repair the time mismatch: the prediction is still evaluated at the end of the run. This is visible in threeway.py's final evaluation block:

~~~python
for p, z, r, v in changes:
    nxt = next_change_phase_for_this_pair
    es = entities_born_between(p + 1, min(p + 6, nxt))
    bucket += list(final_model_prediction(es, r) == v)
~~~

This snippet is explanatory pseudocode. The exact pinned implementation is [threeway.py][g04-code], final-evaluation section. Current truth is global by type/relation; it does not preserve an entity's creation-time target after a later change.

### Size of the mismatch in the registered worlds

The following diagnostic uses the **perfect final-truth predictor**, not a trained model. Numerators and denominators come from reconstructing the exact generator draws and the scorer's cohort masks. The generated change identities match all saved detector-stage records for the audited datasets.

| Seed | Perfect final predictor under revise_next5 | Under revise_next10 |
|---|---:|---:|
| 7703 | 143/152 = 94.079% | 128/140 = 91.429% |
| 7704 | 131/144 = 90.972% | 132/140 = 94.286% |
| 7802 | 135/161 = 83.851% | 126/152 = 82.895% |
| 7803 | 153/165 = 92.727% | 132/146 = 90.411% |
| 7804 | 144/155 = 92.903% | 126/141 = 89.362% |
| 7805 | 124/137 = 90.511% | 141/149 = 94.631% |

The other cooldown seeds score 100% in this perfect-predictor diagnostic. Earlier repeated-update seeds also exhibit the defect, and their complete diagnostic values are in the JSON output.

These values are **not corrections to the actual model scores** and should not be used to divide or rescale them. A model may happen to predict a superseded answer and receive credit under the historical target. Aggregate JSON lacks the per-query final predictions needed to reconstruct a valid replacement score.

### What remains valid, and what should change next

- revise_old restricts itself to the final change still in force for a pair. It does not have this particular historical-target mismatch.
- The audited narrow N1–N5, cooldown K1–K5, and EM cooldown Q1–Q4 predicates do not depend on revise_next5/10. Their recomputed pass/fail statuses remain unchanged.
- This finding concerns the **interpretation and validity of revise_next5/10**, not the other registered gate conclusions.
- To measure **early adaptation**, record predictions at explicit times after each change, before a superseding change, and compare with truth valid at that prediction time.
- To measure **final retention by birth cohort**, query at the final time and compare with final truth.
- Store event IDs, valid-from/valid-until times, prediction timestamps, targets, predictions, eligibility masks, and denominators. Test a perfect predictor and a deliberately stale predictor before consuming model results.

A deterministic repeated-update evaluator test belongs in V6 even if no G04 model is rerun.

## 6. G06: corrected localized protection and its denominators

### What the experiment is

The model is a bilinear classifier, logits(e,r)=W[r]u[e], with 16 latent types, 8 relations, 32 values, 3 shown relations per entity, 40 phases of 64 new entities, and 16 randomly assigned correction targets per phase after the first. There are no background exceptions in this G06 world. Each phase fits the new facts for 200 Adam steps; address edits are computed afterward. Localized protection modifies the **actual decoder displacement after Adam's step**, a valuable distinction from merely projecting raw gradients. [Durability implementation][g06-code] [Local implementation][g06-local-code] [Underlying model and edit][g06-base]

The entity-only bug excluded an entity from all protected relations whenever that entity received any new correction in the current phase. The corrected_before mask now preserves prior corrected entities throughout the fit. The rerun changes only the two entity arms and uses the same seeds. The original table in the prose and original verdict table still show the superseded entity values; the fixed files take precedence. [Local result and correction][g06-local-result]

### Corrected results, means of seeds 8001–8003

| Protection scope | Width | Corrections with latest assignment ≥20 phases old | Old regular facts on corrected entities | New entities' unshown relations |
|---|---:|---:|---:|---:|
| None | 24 | 21.470% | 83.450% | 99.690% |
| All old entities, α=1 | 24 | 98.537% | 99.957% | 66.250% |
| All old entities, α=10 | 24 | 91.097% | 99.123% | 91.237% |
| All old entities, α=100 | 24 | 52.503% | 94.890% | 99.560% |
| Corrected entities, **fixed** | 24 | **98.780%** | **100.000%** | **91.147%** |
| Corrected relation | 24 | 93.917% | 92.097% | 99.293% |
| None | 96 | 21.597% | 91.947% | 99.823% |
| All old entities, α=1 | 96 | 100.000% | 100.000% | 80.740% |
| Corrected entities, **fixed** | 96 | **99.877%** | **100.000%** | **98.917%** |
| Corrected relation | 96 | 99.637% | 92.957% | 99.857% |

The entity fix changes m=24 from 98.413/99.913/91.447% to 98.780/100/91.147%; m=96 from 99.640/100/99.053% to 99.877/100/98.917%. [Fixed m24 raw][g06-fixed24] [Fixed m96 raw][g06-fixed96] [Local verdict with original arm values][g06-local-verdict]

The supported finding is that **protection scope changes the measured stability/inference trade-off in this particular bilinear setup**. It is not a proof that localization removes a fundamental trade-off for arbitrary distributions, unlimited corrections, deep models, or fixed memory budgets.

### Exact event and denominator audit

“624 corrections” means 624 random target assignments, not 624 distinct facts whose value necessarily changed.

| Seed | Assignments | Same-value assignments | Actual value changes | Distinct assigned fact keys | Final ≥20-phase correction denominator | Old regular facts on corrected entities | Entity-phase protection omissions fixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8001 | 624 | 16 | 608 | 587 | 272 | 749 | 111 |
| 8002 | 624 | 18 | 606 | 589 | 277 | 782 | 107 |
| 8003 | 624 | 19 | 605 | 583 | 270 | 743 | 118 |

Each seed has 2,560 entities and 7,680 initially shown fact positions; infer_late has 3,200 queries. The ≥20-phase correction score is computed on **current fact keys whose latest assignment is old enough**. It excludes keys whose latest assignment is more recent, including repeatedly updated keys. It is not an average survival probability over all 624 update events and is not a survival analysis that accounts for censoring. The denominator table is world-only reconstruction, independent of model training.

### Registered G06 predicates

| Prediction | Predicate | Recomputed outcome |
|---|---|---|
| N1 | No-protection matches recorded constrained baseline on corrected_old and corrected_now, per seed | Holds |
| N2 | α=10^-4: corrected_old ≥80% and regular facts on corrected entities ≥95% | Fails; latter mean is 93.97% |
| N3 | α=10^-4 acquisition within 5 pp of no protection | Fails; acquisition 94.54% versus 100% |
| N4 | α=10^-4 new unshown inference within 5 pp of no protection | Fails; 56.06% versus 99.69% |
| N5 | Correction replay: corrected_old ≥90%, acquisition within 2 pp of no protection | Holds |
| N6 | α=10^-4 trace(P)/m ≤.05 in every seed, as the scorer implements it | Holds |
| L1 | Reproduce the no-protection width-24 baseline and all-entity α=1 within .01 | Holds |
| L2 | Entity-only width-24 inference ≤80% | Fails in a favorable direction; corrected result 91.15% |
| L3 | Relation-only width24: old correction ≥90%, inference ≥85% | Holds |
| L4 | Relation-only width96: old correction ≥90%, inference within 3 pp of no protection | Holds |
| L5 | All-entity width96 inference ≤85% | Holds |
| L6 | α=1→10→100 reduces corrected_old and increases inference monotonically | Holds |

The audit recomputes all N/L booleans from raw files, with fixed entity files substituted where applicable. No pass/fail reversal follows from that substitution. [G06 scorer][g06-scorer] [Local scorer][g06-local-scorer]

The replay arm retains 97.44% of old corrections and 98.35% of new inference, but only 40.35% of regular old facts on corrected entities. It reuses all successful historical correction records that still match current truth, with possible duplicate/re-activated records. Full-batch fact counts grow and are not compute-matched to the no-replay arms. Its storage and work must be measured before a resource advantage is asserted. The external fallback store is excluded from the model-alone metrics; feasibility/failure and store use remain separate mechanisms. [Durability implementation][g06-code]

## 7. Formal correction: what protection can and cannot guarantee

### 7.1 Exact bilinear invariance requires exact constraints and fixed features

For one relation, let W be the decoder and u the protected address. If the actual update is D and u is fixed, then

\[
(W+D)u-Wu=Du.
\]

Thus Du=0 is sufficient for exact logit preservation. If the columns of A are all protected addresses and P0 is the orthogonal projector onto ker(Aᵀ), then D=BP0 implies DA=0. This is an exact finite-step statement for this bilinear readout, independent of how the proposed displacement B was generated.

The inspected code does a useful thing: it obtains the Adam displacement first and then filters that actual W displacement. A raw-gradient constraint alone would not generally constrain an adaptive optimizer's eventual displacement. OWM and OGD are relevant published prior art, not new inventions of the Nano LM study. [OWM][r-owm] [OGD][r-ogd]

### 7.2 Finite α is not an exact orthogonal projector

The implementation uses

\[
P_\alpha=\alpha(AA^\top+\alpha I)^{-1}.
\]

If σ_i are A's singular values, the eigenvalues of Pα are

\[
\lambda_i(P_\alpha)=\frac{\alpha}{\sigma_i^2+\alpha}.
\]

For finite α>0 all eigenvalues are positive. A nonzero protected direction generally has an attenuation strictly between 0 and 1, so Pα²≠Pα and PαA≠0. In the α→0 limit, eigenvalues vanish on span(A) and remain 1 in its orthogonal complement.

**Synthetic example:** A=I₂ and α=1 give P=.5I₂. Its trace divided by dimension is .5, but the exact free nullspace dimension is zero. With u=(1,0) and proposed decoder displacement B=(1,0), BPu=.5, so the protected logit changes.

Consequently, trace(P)/m is a **mean attenuation factor**, not a count or fraction of exactly free directions. Reporting rounded 0.0000 also does not establish exact zero. P is inverted in float64 and then cast to float32 in the implementation; no numerical-error certificate is recorded. [Durability implementation][g06-code] [Local implementation][g06-local-code]

For u=Ac with fixed u, an informative bound is

\[
\|BP_\alpha u\|_2
\le \|B\|_2\,
\max_i\frac{\alpha\sigma_i}{\sigma_i^2+\alpha}\,
\|c\|_2
\le \|B\|_2\frac{\sqrt{\alpha}}{2}\|c\|_2.
\]

The last inequality follows by maximizing ασ/(σ²+α) over σ≥0. This is conditional: the coefficient norm, actual displacement, conditioning, finite precision, and accumulation across steps still matter. A small α alone is not a universal accuracy certificate.

### 7.3 Address motion: absent invariant, unmeasured effect size

If u also moves by δu, the exact identity becomes

\[
(W+D)(u+\delta u)-Wu=Du+(W+D)\delta u.
\]

Filtering D against the old address controls only the first term. A fixed-address theorem therefore needs enforced fixed addresses, or an explicit bound on the second term.

The base model constructs one persistent Adam optimizer over full u and W tensors. Indexing new rows gives zero gradients to inactive rows inside an otherwise active u tensor; it does not make those rows independent parameter objects with grad=None. Retained first moments can produce a nonzero displacement under a zero current gradient. The base fit method has a frozen-row restore option, but the G06 runs do not supply it and the protection branches do not apply it. [Underlying Model][g06-base] [G06 loops][g06-code] [PyTorch Adam documentation][r-pytorch]

**Synthetic diagnostic:** with learning rate .003, β₁=.9, β₂=.999, ε=10^-8, and scalar gradients (1,0), Adam's second-step displacement is −0.0020101747339743388 despite the zero second gradient. This proves that zero current gradient is insufficient as a general invariant. It is **not an estimate of old-address drift in these trained runs**.

The schedule provides an important mitigating fact: corrections are written after 200 fit steps in which that old entity has zero gradient. Under default β₁, its first moment has already been multiplied by

\[
.9^{200}=7.055079108655367\times10^{-10}.
\]

Previously edited rows may therefore be numerically stationary in practice. Recently old, unedited rows used by the all-entity guard are a different case. Neither actual displacement was logged in the collected results. The justified architecture change is to **measure and enforce the needed invariant**, not to declare the retention findings false or explain their magnitude by Adam.

A future instrumented diagnostic should log max/mean protected-row displacement, decoder-only logit drift at frozen old addresses, representation-only drift under the new decoder, total logit drift, and optimizer-state handling. An explicit frozen-row control can then separate causes.

### 7.4 Classification and nonlinear extensions need additional bounds

If the original winning logit has margin γ>0 and every logit changes by at most ε, the winning class is preserved when 2ε<γ. Protecting a single winning logit alone does not protect the margin if competitor logits move.

For a differentiable nonlinear model, Jδθ=0 removes the first-order term, not the finite-step remainder. The synthetic example f(θ)=θ² at θ=0 has J=0, yet δθ=.1 changes the output by .01. Under an appropriate local smoothness assumption one can bound the remainder by a constant times ||δθ||²; ReLU boundary crossings require separate care.

The G06 nonlinear file explicitly records **no registered run on seeds 8001–8003**. Its development mechanics checks report address-optimization failures in a two-layer ReLU decoder. Those failures motivate a different write mechanism; they do not prove that every address-only optimization problem is infeasible or that every nonlinear architecture must change shared weights. A target that is coordinatewise dominated in a nonnegative hidden representation is a sufficient simple infeasibility example, but actual target-specific certificates were not supplied. [Nonlinear design/mechanics file][g06-nonlinear]

## 8. Concrete V6 revisions and decision-ledger entries

| Proposed decision | Reason | Acceptance evidence needed | Current status |
|---|---|---|---|
| V6-CT-01: Keep G04/G06 evidence pinned and separate from runtime adoption | Draft PR files are absent at the research cutoff | Source hash, corrected-file precedence, immutable PR head, integrated implementation commit if later adopted | Ready for documentation |
| V6-CT-02: Supersede narrow-router N4/N5 success; preserve later K/Q evidence separately | Corrected masks reverse N4/N5; cooldown uses new seeds and a different policy | Existing raw reanalysis and source history | Ready for documentation |
| V6-CT-03: Require time-valid event targets and prediction timestamps | Perfect final predictions fail the current historical-target cohort score | Deterministic A→B→C evaluator test; stale-predictor negative control; per-query records | V6 reference repair implemented and checked; upstream data unchanged |
| V6-CT-04: Version suppression state by semantic scope and rule epoch | Lifetime (group,r,value) suppression blocks B→C→B and can alias changing cluster labels | Reversion, false-first-flag, cluster split/merge, and delayed-new-evidence tests | Proposed policy change |
| V6-CT-05: Attribute scope effects only after schedule-matched controls | Narrowing changes masks and per-group optimizer scheduling | Sequential unmasked versus masked comparison, fixed data presentations and logged optimizer steps | Experiment pending |
| V6-CT-06: Name finite-α mechanism soft protection and report actual residuals | Pα is generally full-rank and non-idempotent | ||DA||, representation drift, margins, conditioning, numerical precision, and time | Ready for theory correction; instrumentation pending |
| V6-CT-07: Protect actual displacements and the feature invariant required by the proof | Optimizer preconditioning and address motion can violate raw-gradient reasoning | Actual-update checks; old-row drift log; frozen-row control where intended | Requirement ready |
| V6-CT-08: Distinguish update events, distinct keys, no-op assignments, and age-censored cohorts | 624 assignments are only 583–589 distinct keys and include 16–19 no-ops | Event log plus explicit eligibility denominator and age distribution | Evaluator/ledger requirement ready |
| V6-CT-09: Compare resource costs at equal accounting boundaries | Replay, reclustering observations, guards, tables, optimizer moments, and rerouting have different costs | Parameter/state bytes, retained evidence bytes, optimizer-step/data-presentation counts, wall time on specified hardware | Unmeasured |
| V6-CT-10: Keep nonlinear feasibility and model-scale transfer open | No registered nonlinear result, no language-model validation | First a valid write rule, then preregistered durability/side-effect/resource comparisons | Open research |

A minimal suppression-state counterexample is already executable in the audit script: requests B, C, B for one group/relation yield accept, accept, reject under the current lifetime key. This is a **synthetic boundary case**, not evidence that such a failure occurred in the current seed sets.

The next work can close evaluator and proof-contract gaps without another expensive model run. Claims about model quality, nonlinear transfer, scalability, and hardware cost require new evidence and remain open.

### Implementation continuation

After this source audit, a reusable temporal-scoring reference was added in work_v6/architecture/experiments/temporal_revision_scoring_v6.py. It reuses the existing immutable event manifest and target-provenance types, separates early-adaptation plans from final-truth cohort plans, binds exact prediction timestamps, and counts missing eligible predictions as wrong. Its 20 adversarial checks pass, including 1,350 tiny-history plans checked against an independent event scan. A second agent reviewed the temporal/provenance/denominator logic and independently reran the checks. The result is a measurement repair; it does not reconstruct missing G04 per-query predictions or rerun its models. Exact source/result hashes and API details are in [the implementation review](v6_temporal_revision_review.md).

## 9. Audit execution and reproducibility

Run from the repository workspace:

~~~bash
python v6_review/correction_theory/independent_audit.py
~~~

The executed audit:

- verifies 67 primary-file Git blob and SHA-256 hashes;
- recomputes all **26** audited registered gates: 5 narrow, 5 cooldown, 4 EM cooldown, 6 durability, and 6 local-protection predicates;
- confirms every recomputed gate agrees with the relevant stored verdict, using corrected entity arms for the local G06 check;
- confirms cooldown raw flag counts, false-flag counts, missed-change counts, and unique lifetime keys;
- reconstructs G04 change identities and confirms they match saved stage records;
- reconstructs G06 assignment counts, no-op counts, correction-age denominators, and the entity-phase protection omissions repaired by the patch;
- demonstrates the repeated-update target mismatch and small theoretical counterexamples without importing a model implementation or fitting parameters.

No scalar reanalysis can recover missing per-query predictions, optimizer-state trajectories, wall-time measurements, or true memory peaks. No upstream result was rewritten. No assertion here promotes a synthetic diagnostic into a trained-model result.

## 10. References and primary source links

### Research literature

- **Kingma and Ba.** *Adam: A Method for Stochastic Optimization.* arXiv:1412.6980; ICLR 2015. [Primary paper][r-adam]. Used for the momentum/update distinction, not as a convergence guarantee for the present nonconvex experiments.
- **Zeng, Chen, Cui, and Yu.** *Continual learning of context-dependent processing in neural networks.* Nature Machine Intelligence 1, 364–372 (2019). DOI 10.1038/s42256-019-0080-x. [Primary article][r-owm]. Relevant prior art for OWM.
- **Farajtabar, Azizan, Mott, and Li.** *Orthogonal Gradient Descent for Continual Learning.* AISTATS 2020, PMLR 108:3762–3773. [Primary paper page][r-ogd]. Relevant prior art for output-sensitive gradient constraints; finite-step nonlinear preservation requires the qualifications above.
- **PyTorch.** Adam documentation, retrieved 2026-11-23. [Official documentation][r-pytorch]. Describes defaults and the distinction between a zero gradient and an absent gradient. The collected G06 output does not pin its original installed PyTorch version.

All other factual tables derive from the pinned repository sources or the explicitly labeled independent audit. No external article establishes Nano LM's own empirical performance.

[g04-corrections]: https://github.com/wwk5q8z6kk-bit/nano-lm/blob/e1f56b1c4b36b4eb4f16cef4a9b4f5487b147f7c/research/frontier/ontogenesis/CORRECTIONS-G04-20261123.md
[g04-code]: https://github.com/wwk5q8z6kk-bit/nano-lm/blob/e1f56b1c4b36b4eb4f16cef4a9b4f5487b147f7c/research/frontier/ontogenesis/threeway.py
[g04-runner]: https://github.com/wwk5q8z6kk-bit/nano-lm/blob/e1f56b1c4b36b4eb4f16cef4a9b4f5487b147f7c/research/frontier/ontogenesis/run_g04_v2.py
[g04-narrow-protocol]: https://github.com/wwk5q8z6kk-bit/nano-lm/blob/e1f56b1c4b36b4eb4f16cef4a9b4f5487b147f7c/research/frontier/ontogenesis/PROTOCOL-THREEWAY-NARROW.md
[g04-narrow-scorer]: https://github.com/wwk5q8z6kk-bit/nano-lm/blob/e1f56b1c4b36b4eb4f16cef4a9b4f5487b147f7c/research/frontier/ontogenesis/threeway_narrow_score.py
[g04-narrow-verdict]: https://github.com/wwk5q8z6kk-bit/nano-lm/blob/e1f56b1c4b36b4eb4f16cef4a9b4f5487b147f7c/research/frontier/ontogenesis/results_v2/threeway_narrow/verdict.json
[g04-repeat]: https://github.com/wwk5q8z6kk-bit/nano-lm/blob/e1f56b1c4b36b4eb4f16cef4a9b4f5487b147f7c/research/frontier/ontogenesis/results_v2/threeway_narrow/diagnostic_repeat_flags.json
[g04-cd-protocol]: https://github.com/wwk5q8z6kk-bit/nano-lm/blob/e1f56b1c4b36b4eb4f16cef4a9b4f5487b147f7c/research/frontier/ontogenesis/PROTOCOL-THREEWAY-COOLDOWN.md
[g04-cd-result]: https://github.com/wwk5q8z6kk-bit/nano-lm/blob/e1f56b1c4b36b4eb4f16cef4a9b4f5487b147f7c/research/frontier/ontogenesis/THREEWAY-COOLDOWN-20261123.md
[g04-cd-scorer]: https://github.com/wwk5q8z6kk-bit/nano-lm/blob/e1f56b1c4b36b4eb4f16cef4a9b4f5487b147f7c/research/frontier/ontogenesis/threeway_cooldown_score.py
[g04-em-protocol]: https://github.com/wwk5q8z6kk-bit/nano-lm/blob/e1f56b1c4b36b4eb4f16cef4a9b4f5487b147f7c/research/frontier/ontogenesis/PROTOCOL-THREEWAY-EM-COOLDOWN.md
[g04-em-result]: https://github.com/wwk5q8z6kk-bit/nano-lm/blob/e1f56b1c4b36b4eb4f16cef4a9b4f5487b147f7c/research/frontier/ontogenesis/THREEWAY-EM-COOLDOWN-20261123.md
[g04-em-scorer]: https://github.com/wwk5q8z6kk-bit/nano-lm/blob/e1f56b1c4b36b4eb4f16cef4a9b4f5487b147f7c/research/frontier/ontogenesis/threeway_em_cooldown_score.py
[g04-addresses]: https://github.com/wwk5q8z6kk-bit/nano-lm/blob/e1f56b1c4b36b4eb4f16cef4a9b4f5487b147f7c/research/frontier/ontogenesis/memory_rule_sharing.py
[g04-recluster]: https://github.com/wwk5q8z6kk-bit/nano-lm/blob/e1f56b1c4b36b4eb4f16cef4a9b4f5487b147f7c/research/frontier/ontogenesis/memory_rule_recluster.py
[g04-match]: https://github.com/wwk5q8z6kk-bit/nano-lm/blob/e1f56b1c4b36b4eb4f16cef4a9b4f5487b147f7c/research/frontier/ontogenesis/memory_rule_transport.py
[g06-code]: https://github.com/wwk5q8z6kk-bit/nano-lm/blob/8b356f938644287eb78177a2299a78ffc02f0f05/research/frontier/ontogenesis/nullspace_durability.py
[g06-local-code]: https://github.com/wwk5q8z6kk-bit/nano-lm/blob/8b356f938644287eb78177a2299a78ffc02f0f05/research/frontier/ontogenesis/nullspace_local.py
[g06-base]: https://github.com/wwk5q8z6kk-bit/nano-lm/blob/8b356f938644287eb78177a2299a78ffc02f0f05/research/frontier/ontogenesis/correction_interference.py
[g06-local-result]: https://github.com/wwk5q8z6kk-bit/nano-lm/blob/8b356f938644287eb78177a2299a78ffc02f0f05/research/frontier/ontogenesis/NULLSPACE-LOCAL-20261122.md
[g06-fixed24]: https://github.com/wwk5q8z6kk-bit/nano-lm/blob/8b356f938644287eb78177a2299a78ffc02f0f05/research/frontier/ontogenesis/results/nullspace_local/fixed_owm_ent/owm_ent_m24.json
[g06-fixed96]: https://github.com/wwk5q8z6kk-bit/nano-lm/blob/8b356f938644287eb78177a2299a78ffc02f0f05/research/frontier/ontogenesis/results/nullspace_local/fixed_owm_ent/owm_ent_m96.json
[g06-local-verdict]: https://github.com/wwk5q8z6kk-bit/nano-lm/blob/8b356f938644287eb78177a2299a78ffc02f0f05/research/frontier/ontogenesis/results/nullspace_local/verdict.json
[g06-scorer]: https://github.com/wwk5q8z6kk-bit/nano-lm/blob/8b356f938644287eb78177a2299a78ffc02f0f05/research/frontier/ontogenesis/nullspace_durability_score.py
[g06-local-scorer]: https://github.com/wwk5q8z6kk-bit/nano-lm/blob/8b356f938644287eb78177a2299a78ffc02f0f05/research/frontier/ontogenesis/nullspace_local_score.py
[g06-nonlinear]: https://github.com/wwk5q8z6kk-bit/nano-lm/blob/8b356f938644287eb78177a2299a78ffc02f0f05/research/frontier/ontogenesis/nullspace_nonlinear.py
[r-adam]: https://arxiv.org/abs/1412.6980
[r-owm]: https://www.nature.com/articles/s42256-019-0080-x
[r-ogd]: https://proceedings.mlr.press/v108/farajtabar20a.html
[r-pytorch]: https://docs.pytorch.org/docs/2.15/generated/torch.optim.adam.Adam_class.html

