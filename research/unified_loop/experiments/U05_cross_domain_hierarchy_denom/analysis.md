# EXP-U05 Analysis: Cross-Domain Hierarchical & Denomination Verification Benchmark

- **Status:** Verified local result (CPU, `$0.00` compute, runtime `11.1s`)
- **Script:** `research/unified_loop/scripts/run_u05_cross_domain_eval.py`
- **Raw Artifact:** `research/unified_loop/experiments/U05_cross_domain_hierarchy_denom/results.json`
- **Zero-Domain-Branch Source Audit:** `zero_domain_branches_verified = True` (`hits = {}` across `nano/units.py`, `nano/schema.py`, and `nano/grounded_engine.py`)

## 1. Cross-Domain Results (Domains 1–3, `72` Cases)

The exact same `nano/` engine (`DenominationRegistry`, `OntologyGraph`, `resolve_world_state`) was evaluated across three non-medical domains (`24` cases each: `16` clean/equivalence/rollup + `8` hazard/conflict/fail-closed):

| Domain | `FlatString_NoDenom_NoHierarchy` | `DenomOnly_FlatTopics` | `Full_Modular_Nano` |
| --- | ---: | ---: | ---: |
| **Finance / Treasury (`24`)** | `6/24 (25.0%)` (haz `1/8`) | `22/24 (91.7%)` (haz `6/8`) | **`24/24 (100.0%)` (haz `8/8`)** |
| **Engineering / SRE (`24`)** | `5/24 (20.8%)` (haz `1/8`) | `22/24 (91.7%)` (haz `6/8`) | **`24/24 (100.0%)` (haz `8/8`)** |
| **Legal / SLA (`24`)** | `6/24 (25.0%)` (haz `2/8`) | `22/24 (91.7%)` (haz `6/8`) | **`24/24 (100.0%)` (haz `8/8`)** |
| **Aggregate (`72` Cases)** | **`17/72 (23.6%)`** | **`66/72 (91.7%)`** | **`72/72 (100.0%)`** |
| Equivalence / Rollups (`35`) | `11/35` | `35/35` | **`35/35`** |
| Conflicts / Missing-FX / Hazards (`24`) | `4/24` | `18/24` | **`24/24`** |

### Why Each Layer Earns Its Keep
1. **Unit & Denomination Algebra (`+68.1 pp` vs flat strings):** Resolves mixed prefixes/subunits (`$1.5M == 1500k USD`, `250 cents == $2.50`), separates binary IEC vs SI (`2 GiB == 2048 MiB != 2 GB`), unifies dimensionless rates (`150 bps == 1.5% == 15 permille`), applies evidence-backed `ConversionEdge` (`EUR -> USD`), and quarantines missing-FX mixes as `UNCERTAIN` (`fin_c02`, `leg_c05`) rather than guessing rates.
2. **Hierarchy & Conservation Rollups (`+8.3 pp`, `24/24` vs `18/24` over `DenomOnly_FlatTopics`):** `DenomOnly_FlatTopics` missed every `rollup_violation` and `missing_fx` aggregate test (`6/24` missed across `fin_c01`, `fin_c02`, `sre_c02`, `sre_c03`, `leg_c03`, `leg_c05`). Post-order `OntologyGraph.verify_rollups()` catches `sum` (`current_assets`, `balance_sheet_assets`, `e2e_latency`), `max` (`incident_severity` over `P3..P0`), and `min` (`overall_cap`) violations cleanly.

## 2. Domain 4: Legacy Clinical Regression (`C12–C14`, `72` Encounters / `276` Asserted Slots)

Using `CLINICAL_ENCOUNTER_SCHEMA` inside `nanoscribe/grounded.py` as a declarative schema plugin routed through `nano/`:

| Cohort | Asserted Slots | `Gold x Real` | Menu Coverage | `Real Menu Oracle x Real` (`e2e`) | Useful Notes | `N-03` Violations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `C12_independent_dev_20261023` | `104` | `77` | `91` | `89` | `24/24` | `0` |
| `C13_scope_validation_20261023` | `92` | `75` | `75` | `72` | `24/24` | `0` |
| `C14_retention_validation_20261026` | `80` | `76` | `76` | `74` | `24/24` | `0` |
| **Aggregate (`72` Encounters)** | **`276`** | **`228/276`** | **`242/276`** | **`235/276`** | **`72/72`** | **`0`** |

- **7-Adversarial Clinical Hazard Probe:** `0/7` false admissions.
- **Quarantine Posture:** `nanoscribe/` now carries zero research focus and serves strictly as a passive downstream schema/regression test on top of domain-general `nano/`.
