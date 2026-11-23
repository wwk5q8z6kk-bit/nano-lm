# A once-per-group cooldown for the change detector (registered before any confirmatory run)

Follows `CORRECTIONS-G04-20261123.md`. With corrected code, about 40% of the three-way router's false alarms and about half of its true flags re-fire on a group already flagged. Re-firing adds revision volume and no information. **Cooldown:** once a group (its majority cluster or type, relation, value) has been flagged, it cannot be flagged again (`threeway.run(..., cooldown=True)`; off by default, and the default reproduces `results_v2` exactly).

## Development check (seed 1 only; it shaped the predictions)

| Router | Stable rule | Shared only | Old entities revised | New entities, 1–5 phases | Exceptions kept | Noise resisted | Flags / false / missed |
|---|---|---|---|---|---|---|---|
| normal | 70.8% | 74.4% | 76.0% | 75.5% | 84.1% | 90.6% | 72 / 50 / 1 |
| normal + cooldown | 72.3% | 75.7% | 78.3% | 82.4% | 85.1% | 91.1% | 34 / 23 / 1 |
| narrow (share ≥ 0.2) | 71.8% | 75.6% | 68.2% | 71.1% | 82.5% | 90.6% | 76 / 55 / 2 |
| narrow + cooldown | 76.6% | 80.4% | 67.0% | 72.3% | 91.4% | 92.9% | 26 / 16 / 2 |

## Confirmatory design

Seeds 7701–7705 (new). Arms `none`, `threeway`, `threeway_cd`, `threeway_s02`, `threeway_s02_cd` (`cd` = cooldown). Typed addresses, standard private table, corrected code.

## Predictions (held on the five-seed mean; seeds meeting the bound reported)

- K1 **The cooldown removes most re-firing:** `threeway_cd` raises at most 0.6 times as many flags as `threeway`.
- K2 **…without losing detection:** `threeway_cd` misses at most 0.5 more changes per seed than `threeway`, and revises old entities within 0.05 of it.
- K3 **Narrow + cooldown improves the stable rule:** `threeway_s02_cd` is at least 0.03 above `threeway` (full read).
- K4 **…and keeps more exceptions:** `threeway_s02_cd` is at least 0.03 above `threeway`.
- K5 **…narrowing the gap to no routing:** `threeway_s02_cd` stable rule is within 0.08 of `none` (the corrected router sits 8–9 points below).

**Reading.** K1–K5 holding: re-firing was a large share of the router's write volume, and removing it, together with narrow revisions, recovers part of the stable-rule cost that the withdrawn N5 had claimed. K3 failing while K1 holds: re-firing is not where the cost comes from.
