# Execution Plan

Executable tasks under [ACTIVE_NOW.md](ACTIVE_NOW.md). Historical queue preserved at `papers/EXECUTION_QUEUE.md` (superseded stub).

## Phase A — Documentation reset v2 (CLOSED)

| ID | Task | Done when |
|----|------|-----------|
| A1 | Branch from `origin/master` @ `2ad06d2` (exclude `9fe5b6b6`) | Done |
| A2 | Typed `PROJECT_AUTHORITY` + full legacy archives | Done |
| A3 | Strengthen CI (`check_active_now`, `check_docs_integrity`) | Done |
| A4 | Owner review + merge to `master` | **CLOSED** @ `0107b7a` |

## Phase B — P1 scribe (CURRENT)

| ID | Task | Done when |
|----|------|-----------|
| B1 | Encounter representation schema v0 | JSON schema + docs; supports entity/event/evidence refs |
| B2 | Span/evidence bottleneck | Measured on held-out medical dialogue subset (no PHI in repo) |
| B3 | Verified record → note rendering | Note is view of record, not primary truth |
| B4 | External eval protocol draft | [domains/medical/EVALUATION_PROTOCOL.md](domains/medical/EVALUATION_PROTOCOL.md) instantiated |
| B5 | P1 exit gate checklist | Metrics + human review plan — not claimed passed |

## Explicitly out of scope (until separately authorized)

- P2/P3 implementation beyond interface contract
- Evidence Core / ledger edits
- Clinical deployment claims
- PHI / private clinical data on cloud (including RunPod)

## Training / compute (standing)

RunPod is Nano’s **primary GPU training backend** and is **active** infrastructure — see [ACTIVE_NOW.md](ACTIVE_NOW.md) and [infrastructure/RUNPOD.md](infrastructure/RUNPOD.md). Ordinary training/adaptation/CUDA research on RunPod is in-workflow. Materially costly or confirmatory runs remain **experiment-scoped** under the active plan’s budget/auth rules. Local Apple Silicon stays for development, smoke, analysis, preprocessing, evaluation, and small/cheap experiments.

## Verification commands

```bash
python3 scripts/check_active_now.py
python3 scripts/check_docs_integrity.py
python3 fabric/test_fabric.py
python3 trajectory/test_recompute_c3.py
```
