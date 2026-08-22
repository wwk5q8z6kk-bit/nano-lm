# Next coding task (frontier/active-v1)

**Mandate:** `BUILD_SMALL_POWERFUL_USEFUL_SYSTEM_V1` — wedge A

## Exact next task

```text
TASK: Wire ΔU-gated hybrid stub into live ask() escalation path (opt-in flag)
WHY: eval-arms proves KEEP_CLASSICAL on clean fixtures; product still needs
     an explicit --escalate-stub (or env) so over-abstain recoveries can be
     tried per-query without changing default fail-closed behavior.
DO:
  1. Add ask(..., escalate_stub: bool = False) / CLI --escalate-stub
  2. On classical ABSTAIN only, call wedge_v1.eval.arms.escalate_stub_ask
  3. Keep OOS refuse (no TTL synonym on non-TTL queries) — covered by tests
  4. Pin: owner-dogfood --demo still 5/5 with escalate off (default)
  5. Optional: one fixture task where escalate on recovers a deliberate
     classical miss, scored under eval-arms ADMIT path
DONE WHEN: pytest wedge_v1/test_eval_arms.py + smoke + owner-dogfood --demo green
OUT OF SCOPE: training, paid LM, Evidence Core, OWNER_CORPUS requirement
```

## Just shipped

- `python -m wedge_v1 eval-arms` — U_classical vs hybrid-stub + ΔU gate
- Dense citation packing in `report` markdown (+ compare values table)
