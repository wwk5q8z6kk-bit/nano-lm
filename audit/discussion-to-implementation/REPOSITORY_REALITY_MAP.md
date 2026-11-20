# Repository Reality Map

*Audit inventory 2026-07-31T12:40:09.520615+00:00. Privileges code/artifacts over prose.*

## Git / release state

| Item | Value |
|------|-------|
| Branch | `master` @ `0e01d73` = `origin/master` |
| Tags | `paper-alpha-v1`, `stage-t-v1`, `stage-t-v2`, `stage-t-v2-results`, `v0.1` |
| Dirty tree | ~14 modified tracked files; ~72 untracked paths (E1/E3/P1/P2 locks, CI, deps, results) |
| Active RunPod pods | **none** (`runpodctl get pod` empty table) |
| Local Python | 3.14.6 (`.venv`); CI targets 3.12 |

## Subsystem map

| Path | Purpose | Owner/subsystem | Status | Executable? | Tested? | Authoritative? |
|------|---------|-----------------|--------|-------------|---------|----------------|
| `trajectory/` | Empirical preregs, runners, results | Science track | Active; mixed stale docs | Yes (many scripts) | C3 recompute + interference helpers | **JSON results** authoritative; prereg Status often stale |
| `trajectory/e1/` | E1 harness | Kill-gate | Executed KILL | Yes | No dedicated pytest | `results_e1_utility.json` |
| `trajectory/e2/` | E2 U3 script | Mechanism | No RESULT | Partial | No | Prereg status conflict |
| `trajectory/e3/` | E3 normalize | Construct | Auto done | Yes | No | normalize + human JSONs |
| `fabric/` | Phase-1 verify slice | Systems harness | Regression only; expansion STOP | Yes | **8/8 PASS** | `schemas.py`+`results_slice_v1.json` |
| `papers/` | Program locks + drafts | Docs | Mixed; post-α locks untracked | N/A | N/A | EMPIRICAL_FOUNDATION + tagged paper1 |
| `scribe/` | Scribe task + Stages G/A + pointer | Legacy task | Closed experiments | Yes | Pointer gates in-script | AUDIT.md + pointer results |
| `sft/` `pretrain/` `scale/` | Train stack | Build | Historical DONE | Yes | Audit prose | AUDIT.md files |
| `stage_m/` | Induction curriculum kernel | Mechanism | Unmeasured | Kernel yes | No | Prereg only |
| `checkpoints/` | Anchor/chinchilla/e1 weights | Artifacts | Partial local | N/A | N/A | Local only; not full remote set |
| `.github/workflows/ci.yml` | CI | Discipline | Present (untracked?) | Yes | Local green | Intended authority |
| `audit/discussion-to-implementation/` | This audit | Audit | New | N/A | N/A | Audit outputs |

## Executable entry points (non-exhaustive)

| Script | Role |
|--------|------|
| `fabric/slice.py` | Vertical slice runner |
| `fabric/test_fabric.py` | Fabric unit tests |
| `trajectory/recompute_c3.py` | Deterministic C-3 recompute |
| `trajectory/e1/run_e1.py` | Non-LM E1 scoring |
| `trajectory/e1/runpod_official_m0.py` | Official M0 CUDA |
| `trajectory/e3/run_e3_normalize.py` | Normalize construct |
| `trajectory/e2/run_u3_earlystop.py` | E2 U3 (no result) |
| `stage_m/stage_m_kernel.py` | Stage M kernel (no RESULT) |

## Ignored / local-only artifacts

| Pattern | Reality |
|---------|---------|
| `*.jsonl` in `.gitignore` | **Present locally**: C3/IF outputs + fabric ledgers (34 files); **not in git** |
| Checkpoints | Sparse under `checkpoints/` |
| `trajectory/runpod_partial/` | E1 logs + E2 pod metadata residue |

## Test surface (measured this audit)

| Suite | Command | Result |
|-------|---------|--------|
| Root/configured | `pytest` | **15 passed** |
| Fabric | `pytest fabric/test_fabric.py` | **8 passed** |
| C3 | `pytest trajectory/test_recompute_c3.py` | **7 passed** |
| C3 recompute script | `python trajectory/recompute_c3.py` | T/B REFUTED, L UNRESOLVED (matches) |
