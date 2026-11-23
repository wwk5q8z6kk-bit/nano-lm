# V6 temporal scoring repair — implementation and review receipt

## Completed files

- experiments/temporal_revision_scoring_v6.py
- experiments/test_temporal_revision_scoring_v6.py
- results/temporal_revision_checks_v6.json

The old correction_protocol.py, G04/G06 branches, and upstream raw results are unchanged.

## Contract and API

The contract was frozen before checks in temporal_revision_contract_frozen.json. Its canonical SHA-256 is **407fc15ceb89cc51662a28990bc7979c4bdb6325a46e4f07234fb5fb73cf45b1**.

The implementation reuses correction_protocol.Manifest, Event semantics, TargetProvenance and the exclusive write_report utility. Existing summarize already handles revision censoring, but returns undefined accuracy for missing eligible measurements and has no separate final-truth-by-cohort endpoint. The V6 layer therefore adds the missing receipt/endpoint semantics without altering that earlier API.

~~~python
early = early_adaptation_plan(
    manifest, ("event_B", "event_C"),
    offset=1, through_phase=39, censor_policy="new_revision",
)
final = final_truth_plan(
    manifest, ("event_B",), through_phase=39,
)
receipt = score(plan, tuple(predictions))
~~~

A Prediction contains a case ID, explicit Time(phase, global_order), value or abstention, and canonical plan hash. Planned queries occur at phase end; order=len(manifest.events) denotes phase end. Plans and derived intervals belong to the coordinator, not to model input.

Intervals are left closed and right open. The caller chooses new_revision or value_change for early adaptation. The latter bridges same-value assignments only until the first subsequent different value; a later return to the old value never revives the original interval. Final cohorts contain one anchor per distinct query key and resolve the latest target event at the final prediction time.

The scorer reports total=eligible+censored+not_yet_due. Missing eligible predictions and explicit abstentions are wrong. Empty eligible denominators yield null, not a fabricated accuracy. It rejects duplicate/unknown receipts, foreign plan hashes, ineligible receipts, out-of-domain values and wrong/invalid timestamps. Every eligible target is derived from the manifest with its exact observed source-event ID.

## Verification

**20/20 tests pass**, including **1,350 independently scanned tiny-history plans** (27 histories × 2 policies × 5 cutoffs × 5 offsets). Checks cover:

- A→B→C: perfect predictions at each scheduled early time score 2/2; a stale B predictor scores 1/2.
- At the final time, correct C scores 1/1 and stale B scores 0/1.
- Missing eligible early receipts remain 0/2.
- B→C→B retains distinct event identities and never reactivates the first B event.
- Same-value assignments have different explicit outcomes under the two policies.
- Same-phase updates, future supersession, exact boundary censoring, missing/duplicate records, malformed times, unknown/future cohorts, abstention, immutable records, and empty cohorts.
- Default CLI creates no artifacts and explicit output refuses to overwrite.

The upstream-evidence reviewer independently inspected the temporal/provenance/denominator logic and reran all 20 tests. No material defect was found under the stated contract. Its reviewed limitation remains explicit: a declared timestamp and matching receipt cannot establish that an untrusted model was actually queried at that time.

## Exact source and result hashes

| Artifact | SHA-256 |
|---|---|
| temporal_revision_scoring_v6.py | 89f56cfc18e95a1cfea91b286efe5eae02d212877fde9bfddf51023e469c9e35 |
| test_temporal_revision_scoring_v6.py | c8e98f9c7bdd5d47545e2bd5b4bca2394cc3f618597e0773b130bceafb23b4a3 |
| Existing correction_protocol.py dependency | f7339939059c8337a2257c9064c3073522b63fb5744cb6a464e7423c15b86087 |
| temporal_revision_checks_v6.json | 7323e568f674987a8b605cf570499ae3f549b4dcd114355d195a01b4854a6780 |

The result receipt is 17,519 bytes and was created exclusively. Default verification is:

~~~bash
python -B experiments/test_temporal_revision_scoring_v6.py
~~~

Explicit creation of a separate new receipt uses --out with a previously nonexistent path. No training was run. These are synthetic measurement checks, not model-quality results or a correction of missing G04 per-query data.

