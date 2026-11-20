# Verification Command Log

*All commands run during this audit. No paid compute launched.*

## Environment

```text
date -u → 2026-07-31T12:37:23Z (start)
pwd → /Users/mac/Projects/nano-lm
git rev-parse / status → master @ 0e01d73 == origin/master; dirty + many untracked
```

## Tests

### Root pytest
```text
.venv/bin/python -m pytest -q --tb=line
...............                                                          [100%]
EXIT:0
```
Log: `_pytest_root.txt`

### Fabric pytest
```text
cd fabric && ../.venv/bin/python -m pytest -q --tb=short
........                                                                 [100%]
EXIT:0
```
Log: `_pytest_fabric.txt`

### C3 recompute tests
```text
.venv/bin/python -m pytest trajectory/test_recompute_c3.py -q --tb=short
.......                                                                  [100%]
EXIT:0
```
Log: `_pytest_c3.txt`

## Deterministic recomputation

```text
.venv/bin/python trajectory/recompute_c3.py
loaded 3000 records ...
H-transition +1.7 pts -> REFUTED
H-boundary  -8.3 pts -> REFUTED
H-length    +25.0 pts -> UNRESOLVED
T-full control: 100% -> PASS
EXIT:0
```
Log: `_recompute_c3.txt`

## Provider / compute residue (read-only)

```text
runpodctl get pod
ID	NAME	GPU	IMAGE NAME	STATUS
(empty)
```
`runpodctl` present at `/opt/homebrew/bin/runpodctl`. No active pods.

## Artifact presence checks

```text
find . -name '*.jsonl' ... → 34 local files (gitignored)
trajectory/outputs_c3_seed{0,1,2}.jsonl present
trajectory/outputs_if_seed{0,1}.jsonl present
fabric/ledger_*.jsonl present
ls results_e2_* → none
```

## OSS presence

```text
rg LangExtract|llguidance|Guidance|Outlines|Instructor|RAGChecker|OpenEvals|Sigstore → no matches
```

## TODO/FIXME sweep

```text
rg TODO|FIXME in *.py/*.md → essentially no actionable TODO markers in code
(status words DONE/BLOCKED/RUNNING appear heavily in docs — see conflict register)
```

## Branch vs origin

```text
git status -sb → ## master...origin/master
git fetch --dry-run → (no pending shown in session)
```

## Minor audit-only fixes applied

None to scientific code or lockfiles. Created only `audit/discussion-to-implementation/*` deliverables plus intermediate `_*.txt` logs.
