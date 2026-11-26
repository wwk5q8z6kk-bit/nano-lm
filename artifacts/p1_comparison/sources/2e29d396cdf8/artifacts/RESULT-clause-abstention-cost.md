# RESULT — stacked clauses interfere (CONFIRMED); clause-cost-on-abstention NOT CONFIRMED

**Prereg:** `artifacts/PREREG-clause-abstention-cost.md`, committed before either run. The
analysis uses fresh instances i12–i35 only, with exact two-sided McNemar tests.

**Gates:** determinism checks passed for S and D+V, 0/192 mismatches each against the
earlier cells. The guard was eligible 3/3, and format compliance was 576/576.

## H-stack — CONFIRMED

On unbound assertion, D+V against D alone gives DV-only 50, D-only 15, **p = 1.6 × 10⁻⁵**.
Adding the verbatim clause on top of the shortest-span clause makes quotes bind
*worse* (unbound 23.4% vs 14.4%). The verbatim clause *reduces* unbound when used alone,
so the combined effect runs opposite to what either clause does separately.

**Confound, declared before the run:** D+V is 73 words and D is 56, so length is not
matched. The length effect has behaved like a threshold at the first added sentence
(P1 and P2 give the same commission), so the confound is expected to be small. It is
not eliminated.

Stacking has now interfered in 3 of 3 exploratory pairs and in 1 fresh confirmatory
test. **Future arms should use a single clause.**

## H-cost — NOT CONFIRMED

On correct abstention, S against P1 (length-matched, 55 vs 56 words) gives S-only 0,
P1-only 2, **p = 0.5**.

The direction matches the earlier results: every clause tested on fresh data has lost
correct abstentions relative to P1 (D 0v3, V 0v6, S 0v2). Only V reached significance,
though, and the new prediction made for S did not.

"Clauses in general cost abstention" therefore stays **exploratory**. I am not promoting
it by pooling the three arms after the fact. The measured cost stands only for V.

## Fresh-slot summary, all confirmatory cells

| cell | correct abstention (of 144) | unbound/asserted | grounded (of 240) |
|---|---|---|---|
| P1 placebo | 24 | 40.6% | 3 |
| S | 22 | 27.6% | 6 |
| V | 18 | 15.6% | 5 |
| D | 21 | 14.4% | **17** |
| D+V | 20 | 23.4% | 14 |

## Provenance

| cell | run |
|---|---|
| S, 36 instances | `df3e8b9d-ff4d-441f-9968-68477fcf453e` |
| D+V, 36 instances | `0e2a764d-1e6a-4c4b-8138-4dd8253ffc61` |
