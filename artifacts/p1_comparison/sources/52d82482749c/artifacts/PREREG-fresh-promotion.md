# PREREG — fresh-only promotion of two exploratory claims

**Committed before either cell runs.** This is the same design that confirmed D:
36-instance suite, analysis restricted to `FRESH_CONFIRMATORY_IDS` (i12–i35),
exact two-sided McNemar per slot, and the existing `conf-P1` run
(`5242ec03`) as the shared length-matched control. Two new cells: **P0**
(no added clause, 39 words) and **V** (verbatim clause, 56 words).

Metric semantics (Amendment 1 rule):
- **commission** means a slot where `gold_present` is False and `joint_cell` starts with
  `asserted`.
- **unbound** means `joint_cell == "asserted_unbound"`.
- **located** means the gold span is contained in the model's quote, on gold-present
  slots.
- **grounded** means `joint_cell == "asserted_grounded"`.

## Claim 1 — prompt length raises commission (P1 vs P0)

Exploratory basis: on i0–i11 the rate went from 48/72 to 60/72, with the identical 12
slots flipping under every added clause.

- **CONFIRMED:** commission is higher in P1 than in P0 on fresh absent slots, p < 0.05.
- **NOT CONFIRMED** otherwise.

## Claim 2 — the verbatim clause improves selection and unbound (V vs P1)

Exploratory basis: selection 13 vs 1, and unbound 42 fixed vs 2 broken.

- **CONFIRMED (unbound):** unbound is lower in V, p < 0.05.
- **CONFIRMED (selection):** located is higher in V, p < 0.05.
- These are two separate endpoints, each reported on its own. Either can confirm
  independently. The pair is not Bonferroni-corrected, and it is labelled as such.
- **Migration check:** grounded and correct abstention must not degrade significantly.
  "Not significant" means not detected at this n, and is not a proof of absence.

## Validity gates

- Determinism: i0–i11 must reproduce `cal-Soff_Voff` (for P0) and `cal-Soff_Von` (for V)
  slot for slot.
- Credit guard eligible on all three axes, and format compliance ≥ 90%.

## Stated in advance

The stacked-clause interference claim (3 of 3) is **not** tested here. Promoting it
requires S+V and D+V on the fresh suite, which is a separate round.
