# Narrow revisions plus the cooldown, on EM-learned clusters (registered before any confirmatory run)

Joins `THREEWAY-COOLDOWN-20261123.md` (typed addresses: narrow + cooldown cut the stable-rule cost from 11.2 to 5.4 points) with the EM clusters of `THREEWAY-EM-20261122.md` (memory-line session's re-clustering; corrected results in `CORRECTIONS-G04-20261123.md`). Corrected code. Settings unchanged from the registered studies: 32 clusters, re-clustering every 5 phases with no replay, share ≥ 0.2, gate 0.12, cooldown k = 1.

## Development check (seed 1 only)

| EM clusters | Stable rule | Old entities revised | New entities, 1–5 phases | Exceptions kept | Noise resisted | Flags / false / missed |
|---|---|---|---|---|---|---|
| no detector | 73.1% | 23.6% | 32.1% | 100% | 92.4% | — |
| narrow | 69.6% | 61.2% | 66.7% | 93.1% | 89.7% | 27 / 16 / 6 |
| narrow + cooldown | 73.0% | 58.9% | 67.3% | 96.7% | 92.0% | 17 / 10 / 5 |

## Confirmatory design

Seeds 7801–7805 (new). Arms `em_nodet`, `em_narrow`, `em_narrow_cd`.

## Predictions (held on the five-seed mean; seeds meeting the bound reported)

- Q1 **With learned clusters, the full router costs no stable rule:** `em_narrow_cd` stable rule is within 0.02 of `em_nodet`.
- Q2 **…and keeps the detector's revision:** `em_narrow_cd` revises at least 0.25 more old entities than `em_nodet`.
- Q3 **The cooldown is what removes the cost:** `em_narrow_cd` stable rule is at least 0.02 above `em_narrow`.
- Q4 **Exceptions and noise hold:** `em_narrow_cd` keeps at least 0.93 of exceptions, and its noise resisted is within 0.02 of `em_nodet`.

**Reading.** Q1–Q3 holding: on clusters the memory learns itself, the label-free router with narrow revisions and a cooldown handles rule changes at no measurable stable-rule cost. Its remaining limit is revision (about 60%), not damage to stable rules.
