# Next coding task (frontier/active-v1)

**Mandate:** `BUILD_SMALL_POWERFUL_USEFUL_SYSTEM_V1` — wedge A  
**Authorization:** owner `AUTHORIZED` (2026-08-22) — Active Frontier continues; Evidence Core frozen.

## Shipped (this slice)

- `ask(..., escalate_stub=)` + CLI `--escalate-stub` + `WEDGE_ESCALATE_STUB=1`
- Classical ABSTAIN → optional `escalate_stub_ask` recovery; OOS still abstains
- Tests: `wedge_v1/test_eval_arms.py` (8 pins including forced classical miss)
- Owner contact: `.local-data/owner_corpus` gate0 + study check **pass**

## Exact next task

```text
TASK: Human usefulness review on owner-contact study (not agent-applied labels)
WHY: Gate 0 proves classical contact; product value needs USEFUL / OVER_ABSTENTION labels
DO:
  1. export WEDGE_OWNER_CORPUS=/path/to/real/private/notes  # outside repo
  2. python -m wedge_v1 study run --corpus "$WEDGE_OWNER_CORPUS" \
       --tasks wedge_v1/data/owner_tasks/questions-v1.json \
       --dir wedge_v1/.studies/first-use
  3. python -m wedge_v1 review --corpus "$WEDGE_OWNER_CORPUS" --interactive
  4. If first_repeated_failure_class → fix classical (not LM) per evolve output
DONE WHEN: >=10 tasks reviewed with human labels; summary decision not FIX_REPEATED_FAILURE
OUT OF SCOPE: training, paid LM, public claims, Evidence Core edits
```

## Commands (now)

```bash
python3 -m wedge_v1 smoke
python3 -m wedge_v1 ask --escalate-stub --corpus wedge_v1/data/corpus "How long before cached entries expire?"
python3 -m wedge_v1 eval-arms --demo
export WEDGE_OWNER_CORPUS=.local-data/owner_corpus && ./scripts/gate0_contact.sh
```
