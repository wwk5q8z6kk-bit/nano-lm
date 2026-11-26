# H7 evaluator diagnosis — 2026-10-22

The captured H7 evaluator completed and recorded **REJECT**. Preserve that result unchanged. Its gold labels are invalid for the H7 screen, so this execution does **not** establish model accuracy or reject the proposed boundary-training mechanism. This is a defect in gold construction, not evidence that the model predicted every field incorrectly.

## Evidence and scope

This diagnosis reads the [preserved result](../../artifacts/p1_comparison/sources/h7-20261022/h7_screen_evaluation.json) and source code only. The result SHA-256 is `9d1b8035b87b8af9419ead92a0dffd66806df1e77720abdf6a1d1deb98951d49`; its bytes match the authenticated capture member `results/h7_screen_evaluation.json`. The five source files in the causal chain below are unchanged from frozen commit `d6ed212c32384c1d50089e50260918028af88bfd`. No screen input was opened, predictions regenerated, or scientific evaluator executed for this diagnosis.

The result embeds the intended counts at `screen.manifest.screen.state_field_quota`, keyed by `field:state`. Summing those counts by state contradicts every arm's `raw.by_gold_state[state].total`:

| Gold state | Embedded annotation quota | Gold used by each arm |
| --- | ---: | ---: |
| supported | 3,837 | 0 |
| absent | 413 | 0 |
| uncertain | 250 | 0 |
| conflicting | 250 | 0 |
| missing | 250 | 5,000 |
| Total | 5,000 | 5,000 |

Each arm has 1,000 stored `raw.item_diagnostics` rows and 5,000 field diagnostics. All 5,000 have `gold_state: missing` and empty `gold_spans`. This mismatch is present in both the stored aggregate and every stored field diagnostic.

| Arm | Recorded state = joint accuracy | Recorded joint-correct fields | Recorded count-mismatch fields |
| --- | ---: | ---: | ---: |
| H6 control, seed 20260805, epoch 2 | 3.20% | 160 | 4,840 |
| H7 treatment, seed 20260805 | 2.78% | 139 | 4,861 |
| H7 treatment, seed 20260806 | 3.48% | 174 | 4,826 |

These percentages equal the arms' missing-prediction frequencies against the erroneous all-missing gold. They are not accuracy estimates against the annotated H7 targets.

## Exact code chain

1. The H7 generator constructs and validates annotated fields, then stores them in `StateSpanExample.target` via `encode_state_span(fields)`: [h7_boundary_screen_data.py, lines 331–353](../../nano_ai/training/h7_boundary_screen_data.py#L331-L353). Its quota explicitly parses that target: [lines 391–395](../../nano_ai/training/h7_boundary_screen_data.py#L391-L395).
2. The H7 evaluator calls the legacy `_fixture_cases(examples)`: [evaluate_evidence_query_h7.py, line 684](../../nano_ai/training/evaluate_evidence_query_h7.py#L684). That helper ignores `example.target` when constructing gold and instead calls `_extract_fields(request)`: [evaluate_state_span.py, lines 697–705](../../nano_ai/training/evaluate_state_span.py#L697-L705).
3. H7 uses new question templates: [h7_boundary_screen_data.py, lines 45–66](../../nano_ai/training/h7_boundary_screen_data.py#L45-L66). None match the legacy question allowlist: [deterministic_v0.py, lines 26–67](../../nano_ai/adapters/deterministic_v0.py#L26-L67). Exact matching therefore returns `None` ([lines 168–173](../../nano_ai/adapters/deterministic_v0.py#L168-L173)); unknown questions are skipped ([lines 231–239](../../nano_ai/adapters/deterministic_v0.py#L231-L239)); fields with no observations resolve to `MISSING` ([lines 255–257](../../nano_ai/adapters/deterministic_v0.py#L255-L257)).
4. Scoring consumes `case.gold`, not the annotated target: [evaluate_pointer.py, lines 1005–1010 and 1052–1067](../../nano_ai/training/evaluate_pointer.py#L1005-L1067). H7 copies these accuracy values into the result: [evaluate_evidence_query_h7.py, lines 503–518](../../nano_ai/training/evaluate_evidence_query_h7.py#L503-L518).

The error taxonomy then compares nonempty predicted spans with empty gold spans and reports `count_mismatch`: [diagnose_h6.py, lines 159–166](../../nano_ai/training/diagnose_h6.py#L159-L166). The shared-state-correct comparison selects fields where both arms agree with this erroneous gold ([evaluate_evidence_query_h7.py, lines 531–546](../../nano_ai/training/evaluate_evidence_query_h7.py#L531-L546)). Its stored denominators of 123 and 113 consist of jointly missing predictions; empty spans match exactly in both arms, yielding 100% span exactness and zero gain. This does not measure evidence-boundary improvement.

## What remains valid

- The captured bytes, checkpoint identities, recorded predictions, execution completion, and recorded decision remain historical evidence. Each arm's zero reported decode failures is a decoding observation, not a correctness or clinical-readiness claim.
- The recorded `REJECT` remains terminal for this execution. No promotion follows from it, and this diagnosis does not relabel it `PASS` or `GRAY`.
- State accuracy, joint accuracy, wrong-presented counts, binding-error changes, shared-state span gains, and gold-dependent safety conclusions cannot support H7's scientific mechanism verdict. The actual treatment effect remains unestablished by this execution.
- This diagnosis is scoped to H7's use of the legacy helper with new templates. It does not invalidate H6 or other experiments without a separate audit of their own target/gold agreement.

## Test gap and future correction

The generator tests check target parsing, exact source slices, and expected challenge states ([test_h7_boundary_screen_data.py, lines 44–58](../../nano_ai/tests/test_h7_boundary_screen_data.py#L44-L58)). The H7 evaluator tests check fabricated comparison rows, decision logic, authentication/order, and frozen byte hashes ([test_evaluate_evidence_query_h7.py, lines 20–135](../../nano_ai/tests/test_evaluate_evidence_query_h7.py#L20-L135)). Neither checks that the gold passed to scoring equals the annotated target or matches the manifest's state quotas.

A future, separately specified experiment needs gold built from the authoritative annotated target, plus an assertion that target-derived and scorer gold states/spans agree before model inference. Tests should cover every state, multiple conflict spans, new question wording, an oracle prediction with exact expected scores, and deliberate quota disagreement that fails closed. These can use small synthetic fixtures without consuming a held-out screen.

Preserve this consumed screen, raw result, and frozen evaluator. Any repaired evaluation belongs to an explicitly separate analysis/experiment with its screen-exposure status and inferential limits stated; it must not replace or silently rerun H7. This note authorizes no new experiment.
