# A detector cooldown, with narrow revisions, halves the router's stable-rule cost (2026-11-23)

`PROTOCOL-THREEWAY-COOLDOWN.md`, predictions K1–K5 and scorer committed in `071587e` before any confirmatory run. Corrected code (version 2). Seeds 7701–7705 (new). Raw files: `results_v2/threeway_cooldown/`; scored output: `results_v2/threeway_cooldown/verdict.json`. [Observed]

**Background.** After the corrections (`CORRECTIONS-G04-20261123.md`), the three-way router's stable-rule cost was back at 8–11 points, and narrow revisions alone did not reduce it. About 40% of false alarms and half of true flags re-fired on a group already flagged. **Cooldown:** once a group has been flagged for a value on a relation, it cannot be flagged for it again.

| Arm | Stable rule | Shared only | Old entities revised | New entities, 1–5 phases | Exceptions kept | Noise resisted | Flags / false / missed |
|---|---|---|---|---|---|---|---|
| No routing | **82.4%** | 82.9% | 15.1% | 28.5% | 34.1% | 90.6% | — |
| Router | 71.2% | 74.0% | **82.3%** | 80.9% | 86.8% | 90.0% | 52.6 / 28.2 / 1.2 |
| Router + cooldown | 72.7% | 75.6% | 79.6% | 79.4% | 89.0% | 91.6% | 25.0 / 14.0 / 1.0 |
| Router + narrow | 72.9% | 75.3% | 82.3% | 81.0% | 86.1% | 90.3% | 63.8 / 39.0 / 1.0 |
| **Router + narrow + cooldown** | **77.0%** | 80.0% | 68.5% | 70.8% | **94.7%** | **92.7%** | 22.2 / 11.8 / 1.6 |

## Against the registered predictions

| | Prediction | Result | Seeds meeting the bound |
|---|---|---|---|
| K1 | cooldown cuts flags to ≤ 0.6× | **held**, 0.48× | 5 of 5 |
| K2 | … without losing detection (misses within +0.5; revision within 0.05) | **held** on the mean (misses −0.2; revision −0.027) | 3 of 5 |
| K3 | narrow + cooldown lifts the stable rule ≥ 0.03 over the router | **held**, +0.059 | 5 of 5 |
| K4 | … and exceptions kept ≥ 0.03 | **held**, +0.079 | 5 of 5 |
| K5 | narrow + cooldown is within 0.08 of no routing on stable rule | **held**, 0.054 below | 5 of 5 |

## What this shows

- **Re-firing was a large share of the router's write volume.** The cooldown halves the flags (52.6 → 25.0), including the false alarms (28.2 → 14.0), and costs nothing in detection.
- **Cooldown and narrow revisions work together, not separately.** Each alone moves the stable rule by under 2 points. Together they move it 5.9 points, and keep 94.7% of exceptions against 86.8%. A plausible reading, not tested: narrow revisions only help when the writes are not repeated many times.
- **The cost is revision of old entities, which no prediction anticipated:** 68.5% against 82.3% for the router. Fewer, narrower writes revise old entities less. The overall trade-off moves from (82% revision, 71% stable rule, 87% exceptions) to (69%, 77%, 95%).
- **The stable-rule cost relative to no routing falls from 11.2 to 5.4 points.** This is a corrected, registered result, unlike the withdrawn N5.

## Limits

Typed addresses: the corrected learned-cluster gap is 29 points of revision, and this combination was not run on learned clusters. Development-seed choices: the share threshold and the gate. One synthetic world, five seeds, and seed-set variance of about ±10 points on revision. The revision loss would matter in a world where old entities are queried more than new ones.

## Next

1. Find out whether the revision loss comes from the cooldown blocking a needed second write on true changes. Half of true flags re-fire, so a second write may help. Candidate: a cooldown on false-looking groups only, or a limit of two writes per group.
2. Run the combination on learned clusters (EM).

## Exploratory follow-up (development seed 1 only): more writes per group

To recover the revision the cooldown costs, the cooldown was generalised to at most k flags per group (`cooldown_k`; k = 1 reproduces the registered arm exactly).

| Narrow + cooldown | Stable rule | Old entities revised | Exceptions kept | Noise resisted | Flags / false |
|---|---|---|---|---|---|
| k = 1 (registered) | 76.6% | 67.0% | 91.4% | 92.9% | 26 / 16 |
| k = 2 | 72.8% | 77.0% | 84.8% | 91.1% | 60 / 40 |
| k = 3 | 72.4% | 75.6% | 82.8% | 89.7% | 74 / 51 |

A second write per group brings back the false alarms along with the revision, so k only moves the router along the same trade-off line. It does not escape it. No confirmatory run was registered: no candidate beats the registered setting on both sides. Separating true from false groups before the second write, rather than limiting writes, is the open question.
