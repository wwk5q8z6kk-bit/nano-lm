# Nano capability specification

50 capabilities across 26 domains. Status: IMPLEMENTED 11, PARTIAL 14, PROPOSED 6, ABSENT 19

| id | domain | capability | stage | status | evidence |
|---|---|---|---|---|---|
| `PER-TEXT` | Perception | Ingest text sources (notes, letters, summaries) | A | **IMPLEMENTED** | nano/pipeline.py::segment; nano/test_nano_clin_001.py |
| `PER-AUDIO` | Perception | Ingest encounter audio with speaker diarisation | A | **PROPOSED** | — |
| `PER-TABLE` | Perception | Ingest tabular data (labs, medication lists) | B | **PROPOSED** | — |
| `PER-IMAGE` | Perception | Ingest imaging and its report as linked evidence | E | **PROPOSED** | — |
| `PER-SIGNAL` | Perception | Ingest waveforms and continuous device data | E | **PROPOSED** | — |
| `SRC-PROV` | Source intelligence | Record authorship, source type, and bitemporal times | A | **IMPLEMENTED** | nano/contracts.py::SourceArtifact, EvidenceSpanV2 |
| `SRC-EPIST` | Source intelligence | Distinguish measured / observed / reported / inferred | A | **IMPLEMENTED** | nano/contracts.py::EpistemicStatus; test_patient_report_is_not_promoted |
| `EXT-ASSERT` | Extraction | Extract clinical assertions preserving original wording | A | **PARTIAL** | nano/pipeline.py; two defects recorded in artifacts/nano_clin_001/decision.md |
| `EXT-EVENT` | Extraction | Derive clinical events from assertions | A | **PARTIAL** | nano/contracts.py::ClinicalEvent (1:1 with assertions only) |
| `EXT-NEG` | Extraction | Preserve negation as first-class, not absence | A | **PARTIAL** | nano/test_nano_clin_001.py::test_negation_is_preserved |
| `NRM-CONCEPT` | Normalization | Map surface terms to concepts without losing the original | B | **PARTIAL** | nano/contracts.py::ClinicalAssertion.normalized_concept (field only) |
| `NRM-UNITS` | Normalization | Retain original and normalised units together | B | **PARTIAL** | nano/contracts.py::ClinicalAssertion original_units/normalized_units |
| `TMP-BITEMP` | Temporal | Separate event / documentation / discovery / system time | A | **IMPLEMENTED** | nano/contracts.py::TemporalExtent; test_approximate_time_cannot_carry_an_exact_date |
| `TMP-PRECISION` | Temporal | Preserve time precision and refuse to invent it | A | **IMPLEMENTED** | nano/contracts.py::TemporalExtent.__post_init__ raises |
| `LNG-EPISODE` | Longitudinal | Group related events into episodes | B | **ABSENT** | — |
| `LNG-TRAJ` | Longitudinal | Build per-problem trajectories across years | B | **ABSENT** | — |
| `LNG-ZOOM` | Longitudinal | Descend from trajectory to exact source passage | B | **PARTIAL** | nano/contracts.py id chain; artifacts/nano_clin_001/*.jsonl |
| `STA-PROJ` | State | Project patient state from the evidence ledger | A | **IMPLEMENTED** | nano/test_nano_clin_001.py::test_state_is_a_rebuildable_projection |
| `STA-VERSION` | State | Append new evidence without mutating history | A | **PARTIAL** | test_new_evidence_appends_and_does_not_overwrite (append yes; downstream invalidation not implemented) |
| `STA-DIFF` | State | Diff two state versions into a change report | B | **IMPLEMENTED** | nano/contracts.py::StateDelta; nano/test_state_delta.py |
| `CFL-DETECT` | Conflict | Detect contradictions and never resolve them silently | A | **PARTIAL** | nano/pipeline.py::_detect_conflicts; concept-scoped, 5 concepts only |
| `GAP-DETECT` | Conflict | Distinguish not-found / unavailable / never-performed | A | **PARTIAL** | nano/test_nano_clin_001.py::test_not_found_is_distinguished_from_absent |
| `MEM-HIER` | Memory | Working / encounter / episodic / longitudinal / semantic memory | CORE | **ABSENT** | — |
| `MEM-SELECT` | Memory | Decide what is worth remembering | CORE | **ABSENT** | — |
| `KNW-SEP` | Knowledge | Keep patient / medical / institutional / model knowledge apart | E | **PROPOSED** | — |
| `RET-ROUTER` | Retrieval | Route between semantic / exact / temporal / graph / episode retrieval | D | **ABSENT** | — |
| `RSN-TEMPORAL` | Reasoning | Reason over before/after/during/overlap/recurrence | D | **ABSENT** | — |
| `RSN-CAUSAL` | Reasoning | Separate documented rationale / temporal association / inferred cause | E | **ABSENT** | — |
| `RSN-METACOG` | Reasoning | Represent what is known / unknown / conflicting / needed | D | **PARTIAL** | nano/contracts.py::PatientStateSnapshot.uncertainties, unresolved_questions |
| `PLN-DECOMP` | Planning | Decompose a question into an evidence and tool plan | D | **ABSENT** | — |
| `PLN-ADAPTIVE` | Planning | Spend computation in proportion to difficulty | CORE | **ABSENT** | — |
| `TOL-DELEGATE` | Tools | Delegate arithmetic and lookup rather than estimating | D | **ABSENT** | — |
| `VIS-FROMSTATE` | Visualization | Compile charts and timelines from state, not from prose | C | **PARTIAL** | scripts/run_nano_clin_001.py emits timeline.json from events |
| `GEN-COMPILE` | Generation | Compile many artifact types from one patient state | C | **PARTIAL** | nano/contracts.py::DerivedArtifact; one renderer implemented |
| `GEN-AUDIENCE` | Generation | Same facts, different audience (clinician / patient / specialist) | C | **ABSENT** | — |
| `VRF-CLAIM` | Verification | Verify every factual claim against evidence | A | **IMPLEMENTED** | nano/contracts.py::VerificationReceipt; artifacts/nano_clin_001/*/verification_receipt.json |
| `VRF-ADVERSARIAL` | Verification | Attack own output before emitting it | C | **ABSENT** | — |
| `UNC-ABSTAIN` | Uncertainty | Abstain as a first-class outcome | A | **IMPLEMENTED** | fabric/schemas.py::DecisionAction; ledger claim C_FABRIC_SLICE |
| `LRN-CORRECTION` | Learning | Absorb a human correction and recompute what depends on it | D | **ABSENT** | — |
| `LRN-LOOPS` | Learning | Fast loop touches state; slow loop touches weights, offline | CORE | **PROPOSED** | — |
| `SAF-ISOLATION` | Safety | Patient isolation enforced in the type system | A | **IMPLEMENTED** | nano/test_nano_clin_001.py::test_no_cross_patient_contamination |
| `SAF-ACTION` | Safety | Human review is the action boundary | A | **PARTIAL** | nano/contracts.py::ConflictRecord requires human_disposition |
| `EVD-LOCATE` | Evidence | Locate a claim in its source, modality-independently | A | **IMPLEMENTED** | nano/contracts.py::EvidenceSpanV2 |
| `CNV-STATE` | Conversation | Track dialogue state separately from world state | C | **ABSENT** | — |
| `CNV-ACT` | Conversation | Choose to answer, clarify, retrieve, show, cite, or abstain | C | **ABSENT** | — |
| `MTA-EPISTEMIC` | Metacognition | Maintain typed machine state for known/unknown/conflicting/needed | D | **PARTIAL** | nano/contracts.py::PatientStateSnapshot |
| `MTA-WOULDCHANGE` | Metacognition | Name what evidence would change the conclusion | D | **ABSENT** | — |
| `PRD-SEPARATE` | Prediction | Keep observed / inferred / predicted / simulated strictly apart | E | **ABSENT** | — |
| `ART-PLAN` | Artifact compilation | Plan an artifact before writing it (audience, scope, evidence, format) | C | **ABSENT** | — |
| `HUM-AUDIENCE` | Human interaction | Model the recipient without changing the underlying facts | C | **ABSENT** | — |
