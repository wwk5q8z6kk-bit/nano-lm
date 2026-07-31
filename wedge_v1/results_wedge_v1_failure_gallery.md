# wedge_v1 failure gallery

**Source:** `wedge_v1/results_owner_dogfood.json`
**Accuracy:** 1.0 (5/5)

> Empty fine buckets mean *unobserved on this corpus*, not that the mode is solved.

## Fine buckets

- **evidence_absent** (0): _unobserved_
- **retrieval_miss** (0): _unobserved_
- **wrong_span_retrieval** (0): _unobserved_
- **verifier_rejection** (0): _unobserved_
- **correct_abstention** (1): `O05`
- **over_abstention** (0): _unobserved_
- **entity_type_collision** (0): _unobserved_
- **multi_document_contradiction** (3): `O01`, `O02`, `O03`
- **unsupported_composition** (0): _unobserved_
- **ingestion_layout_failure** (0): _unobserved_
- **ok_supported** (1): `O04`
- **other** (0): _unobserved_

## Representative examples

### multi_document_contradiction
- id: `O01` status=`CONTRADICTED` ok=True
- query: How long before cached entries expire?
- repro: `python -m wedge_v1 compare TERM --corpus /Users/mac/Projects/nano-lm/wedge_v1/data/owner_corpus`

### ok_supported
- id: `O04` status=`SUPPORTED` ok=True
- query: QPS
- repro: `python -m wedge_v1 ask "…" --corpus /Users/mac/Projects/nano-lm/wedge_v1/data/owner_corpus`

### correct_abstention
- id: `O05` status=`ABSTAIN` ok=True
- query: What is the clinical accuracy of NanoScribe in hospitals?
- repro: `python -m wedge_v1 ask "OOS clinical question" --corpus /Users/mac/Projects/nano-lm/wedge_v1/data/owner_corpus`

## Coarse buckets

- **ok_abstain** (1): `O05`
- **ok_contradicted** (3): `O01`, `O02`, `O03`
- **ok_supported** (1): `O04`

_Not Evidence Core._
