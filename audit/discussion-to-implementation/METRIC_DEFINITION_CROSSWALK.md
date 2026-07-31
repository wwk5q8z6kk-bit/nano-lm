# Metric Definition Crosswalk

*Executable definitions privilege code + frozen prereg. Prose conflicts classified.*

| Symbol/name | Executable definition | Code path | Document definitions | Conflict | Correct authoritative wording |
|-------------|----------------------|-----------|----------------------|----------|-------------------------------|
| ρ (E1 U) | `flagged / n_fields` (review/flagged load) | `trajectory/e1/common.py` `as_rates` | PREREG_E1: review load; DECISION_P1: review load; sensitivity narrative previously said "hallucination weight" — **corrected this freeze** to "review-load (ρ) weight" (numerics unchanged) | Was prose stale; code/prereg already agreed | **ρ = review load = fraction of fields flagged** |
| halluc | hallucination labels / n_fields | `e1/common.py` | Outside U v1 | None if kept distinct from ρ | Hallucination rate; not inside U v1 |
| liability_presented_bad | halluc that would present without verify | `e1/common.py` | PREREG_E1 liability proxy | None | Report-only liability proxy |
| P | presented_correct / presented | `e1/common.py` | PREREG_E1 | None | Presented precision |
| M | 1 − recall | `e1/common.py` | PREREG_E1 | None | Miss rate |
| L / L_p50 | median latency seconds | `e1/common.py` | PREREG_E1 | Docs L vs JSON L_p50 | p50 latency (seconds) |
| C | method-class relative cost | methods/scorer | PREREG_E1 | None | Relative compute vs 10M scribe |
| decision δ | 0.05 KILL margin | `results_e1_utility.json` | PREREG_E1 | Sensitivity JSON uses delta as cost weight | Decision margin δ ≠ sensitivity δ_C |
| exact match | pred == truth | scorers / E3 | Paper α | None | Exact string match |
| normalized match | normalize_value + PLURAL_MAP | `e1/common.py`, E3 | E3 prereg | Thin; not synonyms | Normalize-then-match (formatting only) |
| agent-rubric faithful | written rubric labels | `results_e3_human.json` | Often called "human" | Naming conflict | **agent-applied rubric audit**; not clinician |
| presented error (fabric) | presented_err/presented | `fabric/slice.py` | fabric README | None | Presented error after verify/abstain |
| review load (Stage A) | scribe-era ~19% | README / scribe audits | Distinct instrument from E1 ρ | Scope carefully | Cite as Stage G/A when using 19% |
| accepted violation of R | systems claim | CLAIM_GLOSSARY | ≠ open-world zero hallucination | — | Only under decidable R |
| within_Tavail_dose_rho | Spearman in C3 recompute | `results_c3_recompute.json` | Name collision with E1 ρ | Yes | Spearman correlation; not utility ρ |

## Classification of ρ conflict

| Hypothesis | Supported? |
|------------|------------|
| Code violated preregistration | **No** — matches PREREG_E1 review load |
| Prose stale / mislabeled | **Yes** — some narrative called ρ weight "hallucination" |
| Arithmetic still correct under review-load meaning | **Yes** |

**Action:** align remaining prose; do not recompute U (no verdict change from renaming).
