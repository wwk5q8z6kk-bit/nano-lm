# Next coding task (frontier/active-v1)

**Mandate:** `BUILD_SMALL_POWERFUL_USEFUL_SYSTEM_V1` — wedge A  
**Authorization:** owner `AUTHORIZED TO CONTINUE` — Active Frontier; Evidence Core frozen.

## Exact next task

```text
TASK: Habit recall CLI — list saved questions + scoped rerun with --doc
WHY: ask/find/compare now support exact --doc scope; habit save/recall
     already carries doc_ids in tests, but `habit --list` / scoped
     `--rerun` still needs a crisp owner-facing weekly K1 flow.
DO:
  1. `python -m wedge_v1 habit --list` prints saved question ids + scope
  2. `habit --rerun [--doc DOC_ID...]` re-asks with persisted or override scope
  3. Fail-closed when saved scope docs missing (REFRESH_FAILED already pinned)
  4. Pin: owner-dogfood --demo still 5/5; smoke green
DONE WHEN: pytest wedge_v1/test_document_scope.py + habit tests + smoke green
OUT OF SCOPE: training, paid LM, Evidence Core, OWNER_CORPUS requirement
```

## Just shipped

- `ask(..., escalate_stub=False)` + CLI `--escalate-stub` / `WEDGE_ESCALATE_STUB=1`
  — classical ABSTAIN only; OOS still abstains; default fail-closed
- CLI `--doc` exact scope on ask/find/scan/compare (runtime scope + CoE pins)
- Document-scope / review-habit product surfaces from prior WIP

## Owner-gated (not this coding task)

Human usefulness labels on real `$WEDGE_OWNER_CORPUS` via `review --interactive`
(≥10 tasks). Agent-applied labels do not count as clinician/human review.
