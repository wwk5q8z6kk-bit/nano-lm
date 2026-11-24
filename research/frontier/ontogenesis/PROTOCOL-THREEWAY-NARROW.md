# Narrow shared revisions: can the rule-change collateral be cut without losing revision? (registered before any confirmatory run)

Follow-up to `THREEWAY-COST-20261122.md`. There, about 13 points of the stable-rule cost came from type-wide rule-change revisions written into shared typed slots that neighbouring rules also read (C3, C4).

**Intervention.** During a revision for changed pair (type z, relation r), only shared slots where (z, r)'s queries carry at least a share `s` of the slot's total read weight receive gradient (`threeway.run(..., narrow=True, share=s)`). Defaults are unchanged: with `narrow=False` the registered results reproduce (checked on oracle seed 7301).

## Development exploration (seed 1 only; it suggested the predictions)

| Oracle revision rule | Old entities revised | New entities, 1–5 phases | Stable rule, shared only | Stable rule, full |
|---|---|---|---|---|
| normal | 99.9% | 99.4% | 71.1% | 67.6% |
| s ≥ 0.2 | 93.8% | 92.5% | 80.5% | 75.9% |
| s ≥ 0.35 | 78.9% | 80.5% | 81.9% | 76.9% |
| s ≥ 0.5 | 59.4% | 61.6% | 82.5% | 77.2% |
| s ≥ 0.8 | 32.4% | 47.8% | 83.0% | 77.6% |
| strict (only slots no other pair reads) | 23.9% | 35.2% | 83.0% | 77.6% |
| no routing (reference) | 21.0% | 36.5% | 83.1% | 83.0% |

## Confirmatory design

Seeds 7501–7505 (new). Arms:

| Arm | Router | Revision | Private table |
|---|---|---|---|
| `none` | none | — | standard |
| `oracle` | oracle | normal | standard |
| `oracle_s02` | oracle | s ≥ 0.2 | standard |
| `oracle_s035` | oracle | s ≥ 0.35 | standard |
| `threeway` | three-way | normal | standard |
| `threeway_s02` | three-way | s ≥ 0.2 | standard |
| `threeway_fixed` | three-way | s ≥ 0.2 | ×16, which removes the private leakage (C2) |

## Predictions (held on the five-seed mean; seeds meeting the bound reported)

- N1 **Narrowing cuts the collateral:** `oracle_s02` stable rule (shared only) is at least 0.06 above `oracle`.
- N2 **…while keeping most of the revision:** `oracle_s02` old entities revised is at least 0.88.
- N3 **The trade-off is monotone:** from `oracle_s02` to `oracle_s035`, revision falls and stable rule (shared only) does not fall.
- N4 **The router benefits too:** `threeway_s02` stable rule (shared only) is at least 0.04 above `threeway`, and its revision is no more than 0.10 below.
- N5 **Both fixes together close most of the gap:** `threeway_fixed` stable rule (full read) is within 0.06 of `none`, with old entities revised at least 0.60.

**Reading.** N1–N2 holding: most of the collateral comes from writing into slots that other rules dominate, and restricting revisions to slots the changed rule dominates removes most of it at a small revision cost. N5 holding: with both fixes, the three-way router's stable-rule cost is mostly closed in this world. The typed-address caveat still applies.
