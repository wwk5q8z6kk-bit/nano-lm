# Narrow shared revisions close most of the three-way router's cost (2026-11-22)

> **Correction (2026-11-23).** Code review found bugs. With corrected code (`CORRECTIONS-G04-20261123.md`), **N4 and N5 fail**, and the headline "the router's stable-rule cost falls under 3 points" is **withdrawn**. The oracle-level result (N1–N3: narrowing cuts collateral at small revision cost) stands. Figures below are the original, superseded run.


`PROTOCOL-THREEWAY-NARROW.md`; development curve, predictions N1–N5 and scorer committed in `cf11f8e` before any confirmatory run. Seeds 7501–7505 (new). Raw files: `results/threeway_narrow/`; scored output: `results/threeway_narrow/verdict.json`. [Observed]

**Question.** `THREEWAY-COST-20261122.md` traced the router's stable-rule cost to two sources:
- about 3 points of private-table collision leakage;
- about 13 points of collateral from type-wide revisions written into shared slots that other rules also read.

Can revisions be confined to slots the changed rule dominates (here: carries at least 20% of the slot's read weight), so that the collateral falls and the revision stays?

| Arm | Stable rule | Stable rule, shared only | Old entities revised | New entities, 1–5 phases | Exceptions kept | Noise resisted |
|---|---|---|---|---|---|---|
| No routing | **83.0%** | 83.6% | 20% | 38% | 37% | 92% |
| Oracle, normal revision | 68.7% | 71.8% | 99.7% | 98% | 99.9% | 90% |
| Oracle, share ≥ 0.2 | 76.7% | 80.9% | 92.5% | 94% | 99.9% | 95% |
| Oracle, share ≥ 0.35 | 78.0% | 82.5% | 76.4% | 84% | 99.9% | 95% |
| Three-way, normal revision | 72.5% | 75.2% | 81.5% | 81% | 85% | 91% |
| Three-way, share ≥ 0.2 | 77.4% | 80.4% | 79.0% | 78% | 95% | 94% |
| **Three-way, share ≥ 0.2 + private table ×16** | **80.3%** | 80.4% | **76.7%** | 79% | **94%** | **94%** |

## Against the registered predictions

| | Prediction | Result | Seeds meeting the bound |
|---|---|---|---|
| N1 | narrowing lifts the oracle's stable rule (shared only) by ≥ 0.06 | **held**, +0.091 | 5 of 5 |
| N2 | … while revising ≥ 0.88 of old entities | **held**, 0.925 | 5 of 5 |
| N3 | a stricter share trades revision for stable rule | **held** | 5 of 5 |
| N4 | the router gains ≥ 0.04 (shared only), losing ≤ 0.10 revision | **held** on the mean (+0.052, −0.025) | 3 of 5 |
| N5 | both fixes bring the router's stable rule within 0.06 of no routing, with ≥ 0.60 revision | **held** (2.7 points below; 77% revised) | 5 of 5 |

## What this shows

- **Most of the collateral came from writing into slots that other rules dominate.** Restricting a revision to slots where the changed rule carries at least a fifth of the read weight recovers about 9 of the oracle's 12 collateral points. It costs about 7 points of revision.
- **The share is a dial.** At 0.2 the oracle keeps 92.5% revision and 76.7% stable rule; at 0.35, 76.4% and 78.0%. The study's earlier "the cost follows write volume" is more exactly "the cost follows how much of each write lands in slots other rules rely on".
- **With both fixes, the three-way router nearly closes its stable-rule cost.** It sits at 80.3% against 83.0% with no routing. It still revises 77% of old entities and keeps 94% of exceptions and 94% of noise resistance. That is a better balance than any arm in the earlier studies.
- **An unexpected gain:** narrowing raised exceptions kept in the router (85% → 95%) and noise resisted (91% → 94%). Fewer stray shared writes means less disturbance to the entities that hold exceptions. Post-hoc reading, not tested.

## Limits

- Typed addresses: this memory is given the types. The memory-line session showed that addresses regrouped by EM recover the given-type inference without labels (0.767 against 0.783), so the limit is about frozen addresses. The router has not been run on regrouped addresses.
- The share threshold and the router's gate were chosen on development seed 1 and are declared tuned.
- The ×16 private table multiplies its memory 16-fold.
- One synthetic world, five seeds per study, with seed-set variance noted in the cost study.
- Not a language-model result.

## Status of G04 in this world

| Kind of surprise | Without labels |
|---|---|
| Noise | repetition holds it back |
| Exceptions | repetition and a confidence gate send them to private storage |
| Rule changes | group agreement detects them, and a narrow shared revision writes them in |

Together these keep 77–79% revision, 94–95% of exceptions and of noise resistance, at a cost of about 3 points of stable-rule accuracy. The next test of the whole design is on a learner whose types are learned rather than given.
