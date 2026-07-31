# Execution Queue

**Operating doc — what we are authorized to do *now*.**
**Adopted:** 2026-07-31
**Active wedge:** `papers/WEDGE_V1.md`

```text
PROGRAM_EXECUTION_STATUS: IDLE_AFTER_NOISY_DIAGNOSTIC
PRIMARY_U: clean-track classical (≈0.891)
PHASE3: ECLASS_CLOSED_WITHOUT_LM
NOISY_TRACK: NOISY_INGEST_NORMALIZE_SUFFICIENT
LM_PROBE: NOT_INDICATED
TRAINING: NOT_AUTHORIZED
NANOSCRIBE: STOP
E4_EXECUTE: BLOCKED
NEXT: IDLE — or owner auth for real private corpus / product packaging
```

## Queue

| Priority | Item | Status |
|----------|------|--------|
| 0 | Freeze integrity | Standing |
| 1–3 | Wedge Phases 1–3 | DONE |
| 4 | Noisy-track diagnostic | **DONE** — normalize recovers (gap≈0.032≤δ); raw U collapses |
| — | LM / training / NanoScribe | Not authorized |

## Results

| Artifact | Role |
|----------|------|
| `wedge_v1/results_wedge_v1_classical.json` | Primary clean U |
| `wedge_v1/results_wedge_v1_phase3.json` | E-class closed without LM |
| `wedge_v1/results_wedge_v1_noisy_diagnostic.json` | Diagnostic noisy raw vs normalized |

Idle is valid.
