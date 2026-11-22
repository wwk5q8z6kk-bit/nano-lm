# V4 independent audit — newly completed correction-margin pilot

**Source cutoff:** `9d88a2c20233f42b4a49c4430264b20f842895ad` on `research/ontogenesis-phase1`. Previous inspected cutoff: `8d3b03e08a5c14d2e9ab330ac8d5006e87e61613`. This audit performs source reads, arithmetic on committed rounded summaries, exogenous-event reconstruction and one constructed LP counterexample. It imports no Torch, runs no model training and makes no remote writes.

## 1. What changed

The compare is two commits ahead. `411a535…` adds per-run seed parsing to the TPU multi-run specification and a second-batch controller. `9d88a2c…` adds five correction-margin result JSONs and an interpretation in `INVESTIGATION.md`. No nested-protocol or native TPU training result is added in this compare. Infrastructure changes are source changes, not evidence of a completed run.

The margin predictions M1–M4 were present at the earlier `8d3b03e…` cutoff. The implementation itself is unchanged at the new result commit: Git blob `4e14e8f25a22fa743dd6ffcc51a812df11ecad16`. This establishes a source-level preregistration/result order; it does not independently establish runtime lineage, environment, checkpoint identity or that no uncommitted change was used.

Exact source copies and identities are in `evidence/source_manifest.json`. The five result files, current investigation, margin implementation and its two local dependencies were fetched at the same immutable commit and checked against Git blob SHA identities. The dependencies retain their earlier blobs.

## 2. Exact descriptive summary

All arms have three seeds: 8001, 8002 and 8003. Percentages below are means of the committed, four-decimal per-seed summaries; L1 values are means of the stored three-decimal summaries. They are not pooled event estimates or confidence intervals.

The source arm name `half` denotes a capped constraint: each other relation's current argmax must retain a gap of at least `min(1.0, 0.5 * current_top_two_gap)`. It does not retain half of a gap larger than 2. The `keep` arm uses `0.9 * current_top_two_gap` without that cap. The recorded `edit_l1` averages successful LP writes only before rounding within each seed; unsuccessful writes supply no edit to that average.

| Arm | Mean reported edit L1 | Solver success | Correction accuracy at phase end | Neighboring-fact accuracy at phase end | Old regular accuracy on corrected entities | Final eligible old-correction accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Minimum L1, margin1, no other constraints |12.6623|100.0000%|99.5200%|35.3200%|34.2100%|23.7967%|
| Capped half-margin preservation, correction margin1 |21.7763|99.8400%|99.8400%|98.0267%|83.4500%|21.4700%|
| Capped half-margin preservation, correction margin3 |55.5033|99.8933%|99.8933%|97.2733%|78.6567%|31.1200%|
| Capped half-margin preservation, correction margin6 |109.9973|99.8933%|99.8933%|96.9100%|75.6667%|35.1500%|
|90% neighboring-margin preservation, correction margin3 |62.4117|99.8400%|99.8400%|98.5333%|89.6533%|21.3600%|

The repeated `half_m1` arm exactly matches the old constrained arm's **four common rounded performance fields and solver-success field for each seed**. Those fields are `corrected_now`, `collateral_now`, `rule_old_corrected`, `corrected_old`, and `feasible`/`lp_feasible`. The new files do not provide weights, full trajectories, residuals, original optimizer states or fresh seeds. Prefer “matches the previously committed rounded summaries” over an unrestricted “reproduced exactly.”

### Registered rules and scope

| Prediction | Raw evidence | Scoped verdict |
|---|---|---|
| M1: minimal unconstrained edit has collateral accuracy at most90% |35.32% mean; every seed below90% | Registered numerical condition holds. Minimum L1 displacement alone was insufficient for locality in this tested setting |
| M2: correction retention rises with margin1/3/6 and reaches60% at6 |21.47% →31.12% →35.15%; monotonic in each seed; each margin6 seed below60% | Direction holds; threshold fails; compound success claim fails |
| M3: keep90% of neighboring margins yields at least93% old-regular accuracy, with solver success at least90% |89.6533% old-regular accuracy; each seed below93%;99.84% solver success | Retention threshold fails; solver-success condition holds; compound success claim fails |
| M4: every tested arm has final eligible old-correction accuracy below90% |All five arms, all seeds below90% | Finite registered condition holds. It is not a theorem about every margin or permanent memory |

Increasing the correction margin in the half-preservation family improves the reported old-correction endpoint while worsening old-regular accuracy on corrected entities. At margin3, increasing neighboring-margin preservation from half to90% improves old-regular accuracy by **10.9967 percentage points**, but lowers old-correction accuracy by **9.76 points**. These are descriptive contrasts on the recorded summaries; they expose competing endpoints rather than establish a general optimum.

## 3. New endpoint distinction: write success versus phase-end survival

The source's `corrected_now` is evaluated **after all16 corrections in the phase**, not immediately after each edit. The minimal arm reports100% solver success but only99.52% mean phase-end correction accuracy. It is therefore incorrect to treat the two fields as interchangeable.

Faithful reconstruction of the source's NumPy world/correction draw order, without model training, gives:

| Seed | Correction events | Phases containing multiple corrected relations of one entity | Earlier same-entity writes followed by another write in that phase | Errors implied by the rounded minimal-arm phase-end score | Same-value assignments |
|---|---:|---:|---:|---:|---:|
|8001|624|2|2|2|16|
|8002|624|4|4|2|18|
|8003|624|5|6|5|19|

Selection is without replacement over **fact pairs**, so distinct relations of the same entity can still be selected. A later unconstrained address edit can undo an earlier correction to another relation of that entity. The available collision counts can accommodate the observed errors. They do **not** show which errors actually occurred or their cause: there are no per-edit predictions, margins or deployed-precision residuals in the raw files. Numerical issues remain a possible explanation, as does later-write interference. Do not attribute the gap to float32 without evidence.

### Constructed counterexample using the actual edit function

An isolated diagnostic extracts the source's `edit` function by AST, imports only NumPy and SciPy, and supplies two binary relations with class1-minus-class0 logit differences:

\[
g_0(u)=u_0,\qquad g_1(u)=-2u_0+u_1.
\]

From `u=(0,0)`, the minimum-L1 margin1 write for relation0 gives `(1,0)`. The following minimum-L1 margin1 write for relation1 gives `(-0.5,0)`. Both solver calls succeed; the second target is correct, and the first target becomes wrong. Both targets are jointly feasible, for example at `(1,3)`. Thus solver success for each local write cannot certify survival through subsequent writes, even in exact representable arithmetic with a jointly feasible solution.

This counterexample supports the measurement distinction. It is **not a reconstruction of the historical learned weights** and does not establish the actual cause of their errors.

## 4. Denominators, revisions and same-value events

The source contains39 correction phases with16 selected keys each, giving624 event observations per seed. Phase-end correction accuracy therefore averages equal-sized phase cohorts. The final old-correction and old-regular cohorts have different eligibility rules.

| Seed | Unique keys ever corrected | Repeat-key correction events | Final old-correction denominator | Final old-regular facts on corrected entities |
|---|---:|---:|---:|---:|
|8001|587|37|272|749|
|8002|589|35|277|782|
|8003|583|41|270|743|

These denominators are **derived from the pinned exogenous generator**, not fields saved in the result JSON. They assume the stored run used the source's default40 phases,64 new entities,16 corrections, and the same constants. The result files alone do not record every runtime argument. The faithful reconstruction records constants, source hashes and stream hashes explicitly.

The16/18/19 same-value assignments update `corrected` and `when` even though the new value equals the prior value. Such events can change cohort membership and apply a margin-enlarging edit without introducing a changed factual label. A successor should record both revision identity and `semantic_change`, with separate treatment of no-op value assignments. Per-revision survival and semantic-value persistence are related but distinct endpoints.

The final `corrected_old` cohort contains keys whose most recent selected update was at least20 phases earlier. It can exclude older versions that were subsequently revised, including same-value revisions. Immediate phase scores and final cohorts cannot be subtracted as a same-event decay estimate.

## 5. Claims requiring narrower wording

| New repository wording | Assessment | Recommended replacement |
|---|---|---|
| “The smallest possible write is the most damaging of all” | Smallest is conditional on L1 coordinates, fixed decoder, selected address, margin and LP constraints; other metrics or parameterizations were not tested. Measured collateral accuracy is not a per-write harm delta | “Among the tested arms, the reported minimum-L1 unconstrained address edit had the lowest phase-end neighboring-fact accuracy.” |
| “What protects … is constraints, not smallness” | The observed counterexample refutes sufficiency of this L1-minimal recipe; it does not prove all small updates need explicit constraints | “The tested L1-minimal recipe was insufficient; the constrained recipe substantially improved the measured locality endpoint.” |
| “Supplying [constraints] … removes [harm]” | Computed-arm neighboring-fact accuracy is about98%, not100%; protected predictions and scored true facts differ; pre-write correctness was not logged | “Explicit preservation or targeted rehearsal substantially improved locality in the tested setting.” |
| “Anything the later data does not re-assert is eroded, whatever the margin” | Only a finite margin set and specific model/training stream were tested; some updates can leave protected outputs invariant without replay | “These tested write-time margin choices did not maintain high final old-correction accuracy through the subsequent training stream.” |
| “A constraint has to be present at every later update that can disturb it” | This is a proposed mechanism principle, not a necessity theorem established by the experiment | “Test mechanisms that preserve the relevant address/readout relationship through subsequent learning, including explicit constraints, replay, isolation and coherent transport.” |
| “Margin1 … reproduced exactly” | Only common rounded summaries are comparable | “The common rounded summaries and solver-success rates match for all three reused seeds.” |

L1 displacement is also coordinate-dependent: an invertible change `u'=Au` preserves predictions with the corresponding decoder map but changes the objective `||A delta||_1` in general. These observations should not be converted into an intrinsic theorem about the size of a function change.

The source protects each other relation's current argmax, which can already be wrong, and converts the update to float32. It collapses nonzero solver statuses into a boolean. These V3 limitations still apply. The raw report's field named `feasible` is best described as **reported solver success** until direct residual validation is available.

## 6. Recommended changes to the architecture package and ledger

1. Change the V3 pending margin-protocol status to **completed three-seed synthetic pilot at9d88a2c**, while preserving the earlier pending event and source cutoff.
2. Add the five-arm table and the exact M1–M4 finite-rule outcomes. Preserve the original source interpretation as historical evidence, with a new scoped review rather than rewriting its raw files.
3. Add a decision distinguishing per-write acquisition, survival through the remaining writes in the phase, and later training survival. Require event-order and revision logs.
4. Extend G03b to log every pre/post write, semantic no-op status, superseding revisions, phase-end predictions and eligible horizon denominators. Hash manifests to confirm common streams.
5. Retain solver return details and post-conversion constraint checks. The new100%-success/99.52%-phase-end discrepancy is a measurement reason to implement these, not a numerical-failure verdict.
6. Treat the margin experiment as evidence against this L1-minimal recipe's sufficiency for locality and against the registered high-retention targets. It selects no native LM backbone and does not establish a universal need for exact external storage.
7. Keep nested/TPU results pending within this compare; new infrastructure code alone does not supply an empirical result.

## 7. Reproduction and audit artifacts

`analysis/reconcile_margin_results.py` runs read-only by default; `--out` exclusively creates a new result file. The source directory must contain the pinned margin implementation, `cab_s.py`, and the five JSON summaries. `--prior-results` optionally compares the old constrained summaries.

```bash
python3 -B analysis/reconcile_margin_results.py --source-dir evidence/upstream_v4/research/frontier/ontogenesis --prior-results /path/to/constrained.json --out /new/path/margin_audit.json
```

The final executed result is `results/margin_reanalysis_v4.json`. It includes raw-summary means/ranges, source identities, the conditional old-summary match, reconstructed stream/cohort diagnostics and the constructed LP example. Earlier scratch diagnostic output is superseded by this consolidated receipt. No neural model was trained or rerun during this audit.
