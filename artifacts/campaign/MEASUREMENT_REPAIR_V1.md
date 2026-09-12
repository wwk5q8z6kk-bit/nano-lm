# Measurement Repair v1

Authority: `artifacts/campaign/measurement_repair_v1.json` (machine-readable).

Historical campaign results are **relabeled, never overwritten**. Six measurement defects invalidated native architecture rankings and unlocked the QLoRA gate on incorrect grounds. Each defect now has a regression test.

## Defects

| ID | Summary | Regression test |
|----|---------|-----------------|
| D1 | Agent canary exhausted token budget on reasoning; scored as capability failure | `nanoscribe/test_agent_canary.py::test_parse_action_marks_unrecognised_output_unparsed` |
| D2 | Unparsed teacher output scored as bad policy | same as D1 |
| D3 | Eval injected gold into candidate set (tautological span metrics) | `nanoscribe/test_native_p1_eval.py` |
| D4 | Tokenizer hard-truncated prompts to 64 chars; newlines decoded as `?` | `nanoscribe/test_native_p1_eval.py::test_hash_tokenize_detokenize_roundtrip_chars` |
| D5 | Student gap averaged counts as rates; hardcoded managed ref | `nanoscribe/test_metric_contract.py` + `student_gap_v1` regeneration |
| D6 | 96-row unit fixture used for architecture ranking | `nanoscribe/test_native_corpus.py` + `nanoscribe/test_corpus_launch_guard.py` |

Infrastructure defect **I1** (pip clobbering pod torch) is tracked separately in `artifacts/campaign/infra_defect_torch_cuda_v1.json`.

## Metric contract

All reported rates must carry explicit `numerator`, `denominator`, `rate`, `eligible_unit`, and `aggregation_level`. No rate may exceed 1.0. Implementation: `nanoscribe/eval/metrics.py`.

## Gates after repair

- **QLoRA compatibility canary**: OPEN (`artifacts/campaign/gates/qlora_canary_gate_v1.json`)
- **QLoRA adaptation**: LOCKED (`artifacts/campaign/gates/qlora_adaptation_gate_v1.json`)
- **Native data / Wave 1**: scaffold ready, corpus build required (`artifacts/campaign/gates/native_data_gate_v1.json`)

## Native round reclassification

Round 1/2/extended summaries retain original fields but are **REVALIDATION_CANDIDATES**, not winners. Provisional successive-halving signal (`evidence_bottleneck`, `span_port`) is preserved for Wave 1 revalidation on a real architecture-screen corpus.
