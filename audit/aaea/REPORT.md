# AAEA Report — nano-lm

*Session `20260731T141706Z` · generated 2026-07-31T14:17:06.932456+00:00*
*Repo HEAD (pre-AAEA-fix commit): `a9d12cb1c456f6c465284e1d469c6326cb14d329`*
*Mode: architecture/engineering audit under post-α evidence-freeze constraints*
*auto_commit: engineering hygiene only · no E2/E4/Fabric/NanoScribe expansion · no push/tag*

## Executive overview

nano-lm is a **measurement-first research repository** with a thin Fabric verification slice and a large trajectory of training/eval scripts. Scientific remediation for token budgets, E1/E3 claim scope, and freeze packaging already landed locally (ahead of origin). AAEA therefore focused on **engineering health compatible with freeze**, not product architecture expansion.

Primary actionable defect found and fixed: **default `pytest` collection was broken on CPU laptops** because `scale/kaggle_scale_test.py` matches pytest's `*_test.py` pattern and asserts CUDA at import time.

## Metrics

| Metric | Value |
|---|---|
| Python files (approx) | 69 |
| Markdown docs (approx) | 86 |
| Automated tests collected (after fix) | 15 |
| Tests passed | 15 |
| CRITICAL gaps | 0 |
| HIGH gaps | 3 |
| MEDIUM gaps | 6 |
| LOW gaps | 3 |
| Fixed this session | 2 |
| Deferred / owner-gated | 9 |

## Top gaps

### HIGH
1. **G-TEST-COLLECT-GPU** — FIXED via `pytest.ini`
2. **G-TEST-COVERAGE-THIN** — deferred; add E1/E3 offline recompute tests
3. **G-PUBLIC-PROVENANCE-LAG** — owner must push/tag

### MEDIUM
- Thin packaging/bootstrap story
- Import-unsafe training scripts
- Runtime L/C provenance
- stage_m auto-pip
- NanoScribe unimplemented (accepted under freeze)

### LOW
- Fabric docstring aspirational language
- tokenizer symlink fragility
- No hardcoded secrets found in scan

## What was fixed

| ID | Change |
|---|---|
| G-TEST-COLLECT-GPU | Added root `pytest.ini` (testpaths=`fabric`,`trajectory`; `python_files=test_*.py`) |
| G-GITIGNORE-THIN | Expanded `.gitignore` for caches, editors, venv, autonomous scratch |

## What was deliberately not done

- No E2/E4 execution
- No Fabric V2 / NanoScribe control-plane implementation
- No scientific claim edits in this AAEA pass (already remediated)
- No push/tag
- No invented dependency pins without inspecting historical env

## Validation

```
pytest --collect-only  → 15 tests, 0 errors
pytest                 → 15 passed
py_compile fabric/slice.py fabric/schemas.py trajectory/recompute_c3.py trajectory/e1/common.py trajectory/e1/methods.py → OK
```

## Recommendations

### P0 (done / immediate)
- Keep `pytest.ini` so CPU CI/dev loops stay green
- Owner push of freeze commits when authorized

### P1 (this sprint, freeze-compatible)
- Add `trajectory/test_e1_utility_recompute.py` and `trajectory/test_e3_normalize.py` offline
- Document E1 L/C normalization in a committed runtime schema
- Restore or recreate `requirements.txt` / `pyproject.toml` from the working venv freeze

### P2 (next)
- Refactor Kaggle/RunPod scripts behind `main()` guards without changing recipes
- Replace stage_m runtime pip with pinned env failure
- Align fabric/slice.py header comments with Fabric boundary README

### P3 (backlog)
- Broader unit coverage for scribe scorers
- Optional packaging as installable package
- Architecture exploration only under new owner-authorized scope (not AAEA auto)

## Freeze interaction

AAEA does **not** change the scientific freeze decision.

**Recommended next owner decision:** `OWNER_APPROVAL_REQUIRED` (push/tag freeze candidate), then `IDLE_AFTER_FREEZE`.

Do not treat AAEA P1 test additions as license to reopen E4.
