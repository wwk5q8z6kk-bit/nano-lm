# Where does the three-way router's stable-rule cost come from? (registered before any confirmatory run)

Follow-up to `THREEWAY-20261122.md`, whose prediction Z6 failed reversed. Stable-rule inference fell from 83% (no routing) to 73% (three-way) and to 70% (oracle), and the cost did not sit where false alarms landed. Two readings were offered after the result:
- the memory-line session's: the cost follows total write volume;
- mine: collisions in the private table.

The memory-line session proposed the masking diagnostic used here: unseen keys are never written privately, so their private read is pure leakage.

## Exploratory decomposition (development seed 1 only; this suggested the predictions)

| Run | Stable rule | Shared table only |
|---|---|---|
| none | 83.0% | 83.1% |
| oracle | 67.6% | 71.1% |
| oracle, no private writes | 70.9% | 71.2% |
| oracle, private 200 steps | 71.0% | 71.2% |
| oracle, private table ×4 | 70.1% | 71.1% |
| oracle, private table ×16 | 71.0% | 71.1% |
| oracle, no rule changes | 76.9% | 82.6% |
| threeway | 70.8% | 74.4% |
| threeway, private table ×16 | 74.8% | 74.7% |

Reading on seed 1: two sources.
1. Leakage through the private table: 3–6 points; removed by a 16× larger table.
2. Collateral from the shared revisions of rule changes: about 12 points in the oracle. It is absent when there are no changes.

## Confirmatory design

`threeway.py`, with the exploration settings `p_slots` and `changes_on` (defaults reproduce the registered study exactly; checked on oracle seed 7301). Seeds 7401–7405, new. Arms:

| Arm | Setting |
|---|---|
| `none` | — |
| `oracle` | — |
| `oracle_x16` | private table 262,144 slots |
| `oracle_nochange` | world without rule changes |
| `threeway` | — |
| `threeway_x16` | private table 262,144 slots |

Measures: `infer_stable` and `infer_stable_shared_only`, plus the registered study's measures.

## Predictions (a prediction holds on the five-seed mean; seeds meeting the bound are reported)

- C1 **Private leakage exists:** in `oracle`, shared-only minus full is at least 0.02.
- C2 **A larger private table removes it:** in `oracle_x16`, shared-only minus full is at most 0.01.
- C3 **The shared table carries a cost from rule-change revisions:** `oracle` shared-only is at least 0.08 below `none`.
- C4 **That cost needs rule changes:** `oracle_nochange` shared-only is within 0.02 of `none`.
- C5 **A larger private table helps the router itself:** `threeway_x16` stable rule is at least 0.02 above `threeway`.

**Reading.** C1–C4 holding: the stable-rule cost has two separable sources. One is a fixable collision effect in the private table. The other is the larger collateral of writing rule changes into a shared table whose addresses overlap across rules. Shrinking that collateral (narrower revisions, fewer steps, or less address overlap) is the open problem.
