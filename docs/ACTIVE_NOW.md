# Active Now — NanoLM Foundation Program

**Updated:** 2026-11-16
**Integration base:** `origin/master` @ `acc9b7e`

## Status (canonical — mirrored in `ACTIVE_NOW.json`)

| Field | Value |
|-------|-------|
| `program_execution_status` | `NANOLM_FOUNDATION_ACTIVE` |
| `capability_frontier` | `NANOLM_2B_4B_GENERAL_FOUNDATION` |
| `current_gate` | `NANOLM_ARCHITECTURE_AND_BASELINE_VALIDATION` |
| `evidence_core` | `FROZEN_UNTOUCHED_BY_DOC_PR` |
| `training_backend` | `KAGGLE_AND_RUNPOD` |
| `training_status` | `ACTIVE` |
| `local_zero_cost_exploratory_training` | `ALLOWED` |
| `routine_runpod_training` | `ALLOWED_WITHIN_ACTIVE_EXPERIMENT_BUDGET` |
| `materially_costly_run` | `EXPERIMENT_SCOPED_AUTHORIZATION` |
| `confirmatory_evidential_run` | `PREREG_PLUS_EXPERIMENT_SCOPED_AUTHORIZATION` |
| `phi_on_cloud` | `NOT_AUTHORIZED` |
| `phi_or_private_data` | `NOT_AUTHORIZED` |
| `clinical_claims` | `FORBIDDEN_WITHOUT_EXTERNAL_HUMAN_VALIDATION` |

## Current NanoLM Focus

NanoLM operates as an independent **2.0–4.0B general-purpose foundation language model** research and engineering initiative (`nanolm/`), strictly separated from historical Scribe application schemas and external DevX systems.

1. **Architecture & Parameter Discipline (`nanolm/config.py`, `nanolm/model/`)**: Verified 2.04B–3.89B candidate architectures (`nanolm-2b-dense`, `nanolm-2b-hybrid-swa`, `nanolm-2b-mla`, `nanolm-2b-loop`, `nanolm-3.3b-dense`, `nanolm-3.8b-moe`) with hard `<= 4.0B` parameter enforcement and exact live MPS hardware profiling (`E-INFRA-2B4B`).
2. **Multi-Domain Tokenization & Decontaminated Data (`nanolm/tokenizer/`, `nanolm/data/`)**: Digit-splitting Byte-Level BPE (`E-TOK-01`), intra-document packed causal sequences with RoPE position resets, and 13-gram benchmark decontamination.
3. **Controlled Ablations & Capability Validation (`nanolm/train/`, `nanolm/eval/`)**: 6-arm controlled architecture/objective ablation (`E-ARCH-01`) in tokenizer-invariant Bits-Per-Byte (`BPB`) and programmatic 6-suite general-purpose evaluation (`E-CAP-01`).

## Links

- [PROJECT_CHARTER.md](PROJECT_CHARTER.md) · [nanolm/CHARTER.md](nanolm/CHARTER.md) · [nanolm/PHASE1_AUDIT.md](nanolm/PHASE1_AUDIT.md) · [nanolm/ARCHITECTURE_AND_TRAINING.md](nanolm/ARCHITECTURE_AND_TRAINING.md) · [infrastructure/RUNPOD.md](infrastructure/RUNPOD.md)
