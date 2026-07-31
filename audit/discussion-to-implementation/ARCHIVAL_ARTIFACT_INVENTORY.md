# Archival Artifact Inventory

*Regenerated 2026-07-31 (Phase 1 baseline recorded before freeze edits).
HEAD `0e01d73205e9c35ea32925fd4d6c7e5fceb61137` · branch `master` · `origin/master` matches HEAD after `git fetch`.*

## Baseline commands (recorded)

| Command | Result |
|---------|--------|
| `git fetch origin` | OK; no divergence vs `origin/master` |
| `git rev-parse HEAD` | `0e01d73205e9c35ea32925fd4d6c7e5fceb61137` |
| `git log --oneline -20` | Paper α camera-ready at tip; see `_freeze_baseline.txt` |
| `git ls-files` | 206 tracked paths |
| `git ls-files --others --exclude-standard` | ~139 untracked (research/audit/artifacts/E1–E3) |
| `git check-ignore -v` on `*.jsonl` | `.gitignore:3:*.jsonl` applies to **new** JSONL |
| `runpodctl pod list` | `[]` — **no active pods** |

## Status vocabulary

`TRACKED` | `UNTRACKED_LOCAL` | `IGNORED_LOCAL` | `EXTERNAL_ARCHIVED` | `REFERENCED_BUT_MISSING` | `PARTIAL` | `SUPERSEDED` | `NOT_EXPECTED`

## Inventory counts

See `artifacts/ARTIFACT_INVENTORY.json` (regenerated this freeze).

| Status | n (approx) | Notes |
|--------|------------|-------|
| TRACKED | 50 | Includes primary C-1b/C-3 JSONL (force-added despite ignore rule) |
| UNTRACKED_LOCAL | 44 | E1/E3 results, preregs, many post-α docs |
| IGNORED_LOCAL | 31 | Fabric ledgers + C-3 **replication** JSONL + checkpoints |
| PARTIAL | 6 | `trajectory/runpod_partial/e2_*` incomplete residue |
| REFERENCED_BUT_MISSING | 1 | `trajectory/results_e2_u3_earlystop.json` |

## Critical JSONL truth (do not overstate gap)

| Class | Paths | Status |
|-------|-------|--------|
| C-1b primary raw | `trajectory/outputs_if_seed{0,1}.jsonl` | **TRACKED** |
| C-3 primary raw | `trajectory/outputs_c3_seed{0,1,2}.jsonl` | **TRACKED** |
| C-3 replication raw | `trajectory/replications/c3/.../outputs_c3_seed*.jsonl` | **IGNORED_LOCAL**; SHA-verified copies in `artifacts/local_raw_archive/` |
| Fabric ledgers | `fabric/ledger_*.jsonl` | **IGNORED_LOCAL** |

Primary C-1b/C-3 JSONL are **not** a clean-clone gap. Replication JSONL and fabric ledgers remain a **REPRODUCIBILITY_LIMITATION** until owner-authorized Release/LFS/object-store publication.

## E2 external-result check (mandatory before GATED/STOP)

| Check | Result |
|-------|--------|
| `**/results_e2*.json` | **none** |
| Active pods | **none** (`runpodctl pod list` → `[]`) |
| Partial residue | `trajectory/runpod_partial/e2_*` (setup/host/pod metadata; `e2_monitor.log` size 0) |
| Runner present | `trajectory/e2/run_u3_earlystop.py` |
| Complete U3 artifact outside repo | **Not found** in local search |
| Conclusion | **NO_COMPLETE_U3_ARTIFACT_FOUND** → GATED/STOP justified |

## Per-experiment manifests

| Experiment | Manifest |
|------------|----------|
| Slot diversity | `artifacts/slot_diversity/MANIFEST.json` |
| Stage T / Pythia ladder | `artifacts/stage_t_v2/MANIFEST.json` |
| Own-stack factorial + corner | `artifacts/ownstack_corner/MANIFEST.json` |
| C-1b | `artifacts/c1b/MANIFEST.json` |
| C-3 primary | `artifacts/c3_primary/MANIFEST.json` |
| C-3 replication | `artifacts/c3_replication/MANIFEST.json` |
| E1 | `artifacts/e1/MANIFEST.json` |
| E3 | `artifacts/e3/MANIFEST.json` |
| Pointer P1 / P2 | `artifacts/pointer_p1/MANIFEST.json`, `artifacts/pointer_p2/MANIFEST.json` |
| Local raw archive | `artifacts/local_raw_archive/MANIFEST.json` + `SHA256SUMS` |

## Token-budget authority (Paper α methods error)

| Model | Pretrain tokens | Source |
|-------|-----------------|--------|
| nano 3.15M | **32.8M** | `pretrain/AUDIT.md` (PUBLIC) |
| scale 10M | ~200M | `scale/AUDIT.md` |
| own-stack 160M full-FT | 200M | `results_ownstack_v2_160m_fullft.json` |
| own-stack Chinchilla | 3.2B | `results_ownstack_v2_160m_chinchilla.json` |

Paper α methods claiming both anchors pretrained on ~200M is **false for 3.15M**. “50× parameter flatness” is not a clean parameter-only law under unequal token budgets. Exact proposed manuscript diffs: `OWNER_APPROVAL_REQUIRED_DIFFS.md` DIFF H.

## Recommended durable publication (owner authorize; not uploaded)

1. Git commit of UNTRACKED_LOCAL primary E1/E3 JSON + manifests + freeze docs.
2. GitHub Release `post-alpha-raw-jsonl` attaching `artifacts/local_raw_archive/*` (esp. replication JSONL).
3. Or Git LFS for ignored replication/fabric JSONL.
4. Proposed tag after commit: `post-alpha-evidence-freeze-2026-07-31` (**not created**).

## Machine-readable

- `artifacts/ARTIFACT_INVENTORY.json`
- `artifacts/SHA256SUMS`
- `audit/discussion-to-implementation/_freeze_baseline.txt`
