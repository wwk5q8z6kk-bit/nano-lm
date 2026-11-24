# On learned clusters, the router with narrow revisions and a cooldown handles rule changes at no stable-rule cost (2026-11-23)

`PROTOCOL-THREEWAY-EM-COOLDOWN.md`, predictions Q1–Q4 and scorer committed in `b65154b` before any confirmatory run. Corrected code. EM re-clustering from the memory-line session. Seeds 7801–7805 (new). Raw files: `results_v2/threeway_em_cooldown/`; scored output: `results_v2/threeway_em_cooldown/verdict.json`. [Observed]

| EM clusters | Stable rule | Old entities revised | New entities, 1–5 phases | Exceptions kept | Noise resisted | Flags / false / missed |
|---|---|---|---|---|---|---|
| No change detector | 72.1% | 16.3% | 27.9% | 99.2% | 88.4% | — |
| Router, narrow revisions | 66.9% | 60.4% | 57.5% | 90.2% | 87.3% | 43.8 / 30.8 / 5.6 |
| **Router, narrow revisions + cooldown** | **71.5%** | **53.9%** | **55.3%** | **96.3%** | **89.4%** | **16.6 / 11.0 / 6.4** |

EM purity was 99.5%, and no changed type was split into before/after clusters on these seeds (it happened in 21% of checks on 7601–7605).

## Against the registered predictions

| | Prediction | Result | Seeds meeting the bound |
|---|---|---|---|
| Q1 | stable rule within 0.02 of no detector | **held**, 0.006 below | 5 of 5 |
| Q2 | revises ≥ 0.25 more old entities than no detector | **held**, +0.376 | 4 of 5 |
| Q3 | the cooldown lifts the stable rule ≥ 0.02 over narrow alone | **held**, +0.046 | 5 of 5 |
| Q4 | exceptions ≥ 0.93, and noise within 0.02 of no detector | **held**, 96.3% and +0.010 | 3 of 5 |

## What this shows

- **On clusters it learns itself, the label-free router now costs nothing measurable on stable rules.** It revises 54% of old entities after a rule change (16% without it), keeps 96% of exceptions, and leaves noise resistance where it was.
- **The cooldown is the step that removes the cost.** Narrow revisions alone cost 5.2 points of stable rule and 9 points of exceptions here. Adding the cooldown recovers both, at 6.5 points of revision.
- **The remaining limit is revision, not damage.** About 6 of 12 changes are never detected with learned clusters, against about 1 with given types. The cause is still unexplained: `THREEWAY-EM-20261122.md`'s diagnostic ruled out over-splitting as the main cause, and before/after splits did not occur on these seeds.

## Where G04 stands (corrected code, this synthetic world)

| Learner | Revision of old entities | Stable rule vs no detector | Exceptions | Noise |
|---|---|---|---|---|
| Bilinear model, detector (study 2) | 93% | −0.4 pt | 100% | 98% |
| Typed memory, narrow + cooldown | 69% | −5.4 pt vs no routing | 95% | 93% |
| EM memory, narrow + cooldown | 54% | −0.6 pt vs no detector | 96% | 89% |

Not a language-model result. Development-seed thresholds; one world; seed-set variance of about ±10 points on revision.
