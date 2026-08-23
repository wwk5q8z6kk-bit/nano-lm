# Execution Plan

Executable tasks under [ACTIVE_NOW.md](ACTIVE_NOW.md). Historical queue preserved at `papers/EXECUTION_QUEUE.md` (superseded stub).

## Phase A — Documentation reset v2 (CLOSED)

| ID | Task | Done when |
|----|------|-----------|
| A1 | Branch from `origin/master` @ `2ad06d2` (exclude `9fe5b6b6`) | Done |
| A2 | Typed `PROJECT_AUTHORITY` + full legacy archives | Done |
| A3 | Strengthen CI (`check_active_now`, `check_docs_integrity`) | Done |
| A4 | Owner review + merge to `master` | **CLOSED** @ `0107b7a` |

## Phase B — P1 scribe foundation (CLOSED on master)

| ID | Task | Done when |
|----|------|-----------|
| B1 | Encounter representation schema v0 | PR #37 — `nanoscribe/encounter.py` |
| B2 | Span/evidence transport + eval | PR #38 — `test_evidence_transport.py` |
| B3 | Model adapter + baseline bridge | PR #40 — `adapt.py`, `adapters.py` |
| B4 | Qwen inference + three-track harness | PR #41 — `tracks.py`, serverless |
| B5 | External eval protocol draft | [domains/medical/EVALUATION_PROTOCOL.md](domains/medical/EVALUATION_PROTOCOL.md) — draft exists |

## Phase C — Accelerated research campaign (CURRENT)

| ID | Task | Done when |
|----|------|-----------|
| C1 | Campaign autonomous execution doc + control plane | [`artifacts/campaign/CAMPAIGN_AUTONOMOUS_EXECUTION.md`](../artifacts/campaign/CAMPAIGN_AUTONOMOUS_EXECUTION.md) + `campaign_control_plane.py inventory` |
| C2 | Tool calling stack (structured + tool) | `nanoscribe/test_tool_calling.py` green · [TOOL_CALLING.md](infrastructure/TOOL_CALLING.md) |
| C3 | Agent platform canary | `agent_canary_bench` reproducible · sandboxed `coding_tools` |
| C4 | Native Nano extended + evidence bottleneck | Round summaries in `artifacts/campaign/` |
| C5 | Student C1/C2 serverless fan-out | Manifest + metrics JSON committed |
| C6 | Harness regression on all active tracks | `nanoscribe/` test suite green |
| C7 | Selective port to `master` | Owner review — no Evidence Core diff |

See [research/ACCELERATED_CAMPAIGN.md](research/ACCELERATED_CAMPAIGN.md) · [subsystems/NANOSCRIBE.md](subsystems/NANOSCRIBE.md).

## Phase D — P1 exit gate (after C)

| ID | Task | Done when |
|----|------|-----------|
| D1 | Verified record → note rendering | Note is view of record, not primary truth |
| D2 | External medical dialogue eval | Licensed set + manifest — no PHI in repo |
| D3 | Blinded human evaluation | Protocol instantiated — not claimed passed |
| D4 | P1 mastery decision | Owner sign-off |

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
python3 -m pytest fabric/test_fabric.py trajectory/test_recompute_c3.py nanoscribe/test_encounter_v0.py nanoscribe/test_tool_calling.py -q
```
