# Next coding task (frontier/active-v1)

**Mandate:** `BUILD_SMALL_POWERFUL_USEFUL_SYSTEM_V1` — wedge A  
**Authorization:** owner `AUTHORIZED TO CONTINUE` — Active Frontier; Evidence Core frozen.

## Exact next task

```text
TASK: Wire habit --list / scoped save into owner-ready one-liner path
WHY: list+rerun+--doc shipped; owner-ready should print the habit memory
     commands so weekly K1 dogfood has a single entrypoint.
DO:
  1. `python -m wedge_v1 owner-ready --demo` includes habit --list / --rerun hints
  2. Keep owner-ready fail-closed when corpus missing
  3. Pin: smoke + owner-dogfood --demo 5/5 green
DONE WHEN: pytest wedge_v1/test_owner_smoke.py + habit tests + smoke green
OUT OF SCOPE: training, paid LM, Evidence Core, inventing human usefulness labels
```

## Just shipped

- `habit --list` (+ `--json`) — saved question ids, state, scope
- `habit --save/--rerun --doc DOC_ID` — scoped save + scoped rerun override
- Fixed `--rerun` to actually call `session(rerun=True)` (was record-only)
- Prior: `ask --escalate-stub`, CLI `--doc` on ask/find/scan/compare

## Owner-gated (not this coding task)

Human usefulness labels on real `$WEDGE_OWNER_CORPUS` via `review --interactive`.
