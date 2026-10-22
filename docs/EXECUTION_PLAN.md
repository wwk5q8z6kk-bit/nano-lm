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
| C3 | Define the next bounded intervention | **Done** — H7-BM isolates local evidence-boundary ranking; H6 is not rerun or relabeled |
| C4 | Finish bounded H7 screen | **Executed; consumed** — both seeds and single screen completed for $0 in 4,017.465 seconds. Recorded REJECT is scientifically invalid because the evaluator replaced annotated gold with all-missing fields; exact evidence and diagnosis preserved, no rerun |
| C5 | Re-evaluate P1 exit gap | **Assessed** — checked reference scribing path and annotation transport exist; learned exact-span quality, independent semantic annotation review, longer-dialogue transfer and external human review remain open. No H7 treatment conclusion or P1 mastery claim |
| C6 | Import preserved compact-model comparison | **Built** — compact, six recovered OpenResearch prompt variants, corrected C2 and terminal H6 evidence with exact provenance and explicit denominators; H7 bytes imported with an explicit invalid-gold warning |
| C7 | Connect one end-to-end scribing path | **Built** — transcript-only reference or optional compact extraction → encounter → source-checked review draft; 149 NanoScribe tests pass. Compact pilot remains limited; external human validation is open |
| C8 | Validate future evaluation target transport | **Built** — separate annotation-owned evaluator with exact spans, metadata and independent field/state quota gates; toy oracle and all-missing negative control. H7/H6 remain untouched; semantic annotation review still required |
| C9 | Develop one learned scribing transfer pilot | **Executed, limited** — 20 synthetic development dialogues, separate same-agent annotation review, quota/oracle/negative controls; pinned base Qwen compact model connected to checked scribing. Rules prompt failed 20/20; worked examples yielded two useful review drafts, four abstaining drafts and 14 failures for $0. No fine-tuned adapter recovery or clinical claim. [Results](../artifacts/p1_pilot_v1/REPORT.md) |
| C10 | Improve compact format and paraphrase handling | Next: constrained per-field generation with exact source selection; separately tested context verification beyond legacy question wording. Preserve uncertainty/conflict and evaluate on additional development cases before any new scientific screen |

## Explicitly out of scope (until separately authorized)

- P2/P3 implementation beyond interface contract
- Evidence Core / ledger edits
- Clinical deployment claims
- PHI / private clinical data on cloud (including RunPod)

## Training / compute (standing)

Kaggle and RunPod are complementary active backends. Optimize useful evidence
per dollar and wall-clock hour: prefer Kaggle when the free runtime fits and
turnaround is acceptable; use RunPod when faster compute justifies the spend
within the active experiment budget, or for exact GPU/runtime requirements,
persistence, or managed inference. H7 retains its frozen CPU recipe and $0
ceiling. Authenticated OpenResearch access is available and six prior prompt
variant runs are recovered in the comparison; no new managed boxes were launched.
Materially costly or confirmatory runs remain
experiment-scoped under the active plan’s budget and authorization rules.

## Verification commands

```bash
python3 scripts/check_active_now.py
python3 scripts/check_docs_integrity.py
python3 fabric/test_fabric.py
python3 trajectory/test_recompute_c3.py
```
