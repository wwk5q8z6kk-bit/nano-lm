# Execution Plan

Executable tasks under [ACTIVE_NOW.md](ACTIVE_NOW.md). Historical queue preserved at `papers/EXECUTION_QUEUE.md` (superseded stub).

## Phase A — Documentation reset v2 (CLOSED)

| ID | Task | Done when |
|----|------|-----------|
| A1 | Branch from `origin/master` @ `2ad06d2` (exclude `9fe5b6b6`) | Done |
| A2 | Typed `PROJECT_AUTHORITY` + full legacy archives | Done |
| A3 | Strengthen CI (`check_active_now`, `check_docs_integrity`) | Done |
| A4 | Owner review + merge to `master` | **CLOSED** @ `0107b7a` |

## Phase B — P1 scribe foundation (BUILT)

| ID | Task | Done when |
|----|------|-----------|
| B1 | Encounter representation schema v0 | **Done** — PR #37 |
| B2 | Constrained evidence transport | **Done** — PR #38 |
| B3 | Model adapter and baseline bridge | **Done** — PR #40 |
| B4 | Three-track inference harness | **Done** — PR #41 |
| B5 | Nano Core, learned Nano AI, benchmark integration | **Done** — PRs #107–#110 |

## Phase C — P1 learned intelligence (CURRENT)

| ID | Task | Done when |
|----|------|-----------|
| C1 | Reconcile H6 terminal evidence | **Done** — exact historical result and decision are present on `master`; hashes verify |
| C2 | Localize H6 failures | **Done** — authenticated row-level analysis separates state selection from evidence binding; threshold and verifier effects remain explicitly unassessed |
| C3 | Define the next bounded intervention | **Current** — one falsifiable hypothesis targets one diagnosed mechanism; H6 is not rerun or relabeled |
| C4 | Run cheapest valid screen | Local/Kaggle first when runtime fits; manifest, termination condition, and disjoint data are frozen |
| C5 | Re-evaluate P1 exit gap | Automatic evidence plus external human-review plan; no mastery claim from synthetic data |

## Explicitly out of scope (until separately authorized)

- P2/P3 implementation beyond interface contract
- Evidence Core / ledger edits
- Clinical deployment claims
- PHI / private clinical data on cloud (including RunPod)

## Training / compute (standing)

Kaggle and RunPod are complementary active backends. Prefer Kaggle when the
free runtime fits; use RunPod for exact GPU/runtime requirements, persistence,
or managed inference. Materially costly or confirmatory runs remain
experiment-scoped under the active plan’s budget and authorization rules.

## Verification commands

```bash
python3 scripts/check_active_now.py
python3 scripts/check_docs_integrity.py
python3 fabric/test_fabric.py
python3 trajectory/test_recompute_c3.py
```
