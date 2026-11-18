# EXP-U05 Protocol: Cross-Domain Hierarchical & Denomination Verification Benchmark

**Date:** 2026-11-18
**Status:** Preregistered BEFORE running `scripts/run_u05_cross_domain_eval.py`
**Worktree / Branch:** `research/unified-discovery-loop-v1`
**Compute Budget:** `$0.00` (local CPU deterministic execution)

## 1. Scientific Question
Can a single domain-agnostic hierarchical ontology + denomination verification engine
(`nano/units.py`, `nano/schema.py`, `nano/grounded_engine.py`) with **zero**
domain-specific verification branches operate seamlessly across disparate
domains—Financial Balance-Sheet & Treasury Accounting, Engineering/SRE Incident &
Resource Rollups, Legal/SLA Rate & Liability Schedules, and Clinical `NanoScribe`
(`C12–C14`)—while beating flat baselines and strictly preserving fail-closed
custody (zero unbacked FX conversions, zero third-party/hypothetical/quoted
hazard admissions)?

## 2. Systems Compared (Zero Domain Branches in `nano/`)
1. **`FlatString_NoDenom_NoHierarchy`**: Extracts raw clauses via surface regexes, compares raw normalized strings without unit/currency algebra, without `ConversionEdge` conversions, and without `OntologyGraph` sibling/rollup constraints.
2. **`DenomOnly_FlatTopics`**: Uses `DenominationRegistry` for unit parsing & comparison, but disables hierarchy (`exclusive_siblings=False`, `rollup_op=""`).
3. **`Full_Modular_Nano`**: Complete `DenominationRegistry` + `OntologyGraph` (`owning_siblings_win`, `verify_rollups`) + `resolve_world_state` (`nano/`), identical engine code across all four domains.

## 3. Benchmarks & Metrics
1. **Finance / Treasury (`24` cases: `16` clean/equivalence + `8` hazard/conflict):**
   - Multi-scale USD/EUR/cent equivalence (`$1.5M == 1500k USD`, `250 cents == $2.50`), observed `ConversionEdge` (`EUR -> USD`) with lineage vs missing `GBP -> USD` edge (`INCOMPARABLE_DENOMINATION`), `balance_sheet.current_assets` sum rollup (`cash + receivables + marketable_securities`), role separation (`stock` vs `flow` vs `limit`), competitor/subsidiary third-party exclusion, and past-year/global-reassignment traps.
2. **Engineering / SRE (`24` cases: `16` clean/equivalence + `8` hazard/conflict):**
   - Exact binary IEC vs decimal SI byte algebra (`2 GiB == 2048 MiB != 2 GB`), `e2e_latency` sum rollup (`db + cache + api`), `max` ordinal incident severity rollup across subsystems (`P0..P3`, `9/10`), quoted log-file protection, hypothetical load hazards, and prompt-injection blocks.
3. **Legal / SLA Contracts (`24` cases: `16` clean/equivalence + `8` hazard/conflict):**
   - Ratio equivalence (`150 bps == 1.5% == 15 permille`), duration/notice equivalence (`14 days == 2 weeks`), `min` hierarchical liability-cap / SLA rollup across parent & subsidiary clauses, explicit indemnity absence vs `except` carve-out hazards (`partial_exclusion_not_blanket_denial`).
4. **Clinical `NanoScribe` `C12–C14` (`72` encounters, `276` asserted slots + `7` adversarial hazards):**
   - Evaluates declarative `CLINICAL_ENCOUNTER_SCHEMA` (`nanoscribe/grounded.py`) routed through `nano/` via `evaluate_cohort_2x2` (`Gold x Real`, `Menu Coverage`, `Real Menu Oracle x Real`, `N-03 custody`, `0/7` hazards).

## 4. Preregistered Acceptance & Falsification Criteria
- **Zero-Domain-Branch Check:** `grep` / source verification confirms zero clinical/finance/SRE/legal conditionals inside `nano/units.py`, `nano/schema.py`, and `nano/grounded_engine.py`.
- **Fail-Closed Custody Check:** `0` unbacked FX conversions without an observed `ConversionEdge`, `0` false absence admissions on `except` carve-outs or hedged denials, and `0/7` clinical adversarial hazard admissions.
- **Gain Over Controls:** `Full_Modular_Nano` must outperform `FlatString_NoDenom_NoHierarchy` by `>= +20.0 pp` on Domains 1–3 exact case accuracy AND outperform `DenomOnly_FlatTopics` on hierarchical rollup/sibling-collision cases while maintaining `>= 233/276` end-to-end grounded retention on `C12–C14`.
