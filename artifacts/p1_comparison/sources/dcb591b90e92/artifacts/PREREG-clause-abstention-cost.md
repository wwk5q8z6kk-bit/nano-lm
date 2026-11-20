# PREREG — do instruction clauses cost abstention? And does stacking interfere?

**Committed before either cell runs.** Fresh instances i12–i35 only, exact two-sided
McNemar per slot, `conf-P1` as the length-matched control.

## Why this needs new data

On fresh slots, both instruction clauses tested so far moved correct abstention against
themselves: D was 0 vs 3 (p = 0.25) and V was 0 vs 6 (p = 0.031). Pooling those two
arms after having seen them would be a post-hoc test. This round therefore makes the
prediction **before** seeing a third clause.

## H-cost — an instruction clause costs abstention beyond what length alone costs

The new cell is **S** (the sufficiency clause alone, 55 words), compared with P1 (56
words). Length is matched, so any difference comes from the clause's content.

The prediction is directional: **S has fewer correct abstentions than P1.**

- **CONFIRMED:** P1-only exceeds S-only on correct abstention, with p < 0.05.
- **REFUTED-IN-DIRECTION:** S-only exceeds P1-only, with p < 0.05.
- **NOT CONFIRMED:** everything else.

Note that S *literally instructs* the model to reply NOT_MENTIONED when information is
absent. A confirmation here would mean that an instruction to abstain lowers abstention
relative to a content-free sentence of the same length.

## H-stack — stacked clauses interfere

The new cell is **D+V** (73 words), compared with **D alone** (the `conf-D` run, 56
words) on unbound assertion. Exploratory basis: on i0–i11, D+V was worse than D alone
on unbound (39 vs 6).

- **CONFIRMED:** DV-only exceeds D-only on unbound, with p < 0.05.
- The comparison is confounded with length (73 vs 56 words). This is stated here in
  advance. The length effect is a threshold at the first added sentence (P1 = P2 on
  commission), so the confound is expected to be small, but it is not eliminated.

## Gates

- Credit guard eligible on all three axes.
- Format compliance ≥ 90%.
- Determinism: i0–i11 must reproduce `cal-Son_Voff` (for S) and `cal-DV` (for D+V).

## Metric semantics (Amendment 1)

- **correct abstention:** `joint_cell == "abstained_correct"`.
- **unbound:** `joint_cell == "asserted_unbound"`.
