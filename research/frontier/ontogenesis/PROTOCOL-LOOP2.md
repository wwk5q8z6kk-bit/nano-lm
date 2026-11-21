# Protocol: jump probability and timing, dense against looped (committed before any of these fits)

Follows the mixed result of `PROTOCOL-LOOP.md` (looped learned two-step chains in 2 of 3 seeds, dense in 0 of 3, at 13,000 steps).

**Question.** Does dense make the jump later, or not at all within twice the budget? Is the looped model's jump more likely?
**Task, optimiser.** Identical to `PROTOCOL-LOOP.md`.
**Budget.** 26,000 steps, evaluation every 1,000.
**Arms.** `dense`, `looped`. **Seeds.** 21, 22, 23, 24, 25 (new; none used before).
**Metric.** A run "learned" if two-step accuracy is at least 90% at any evaluation. Secondary: the step of the first such evaluation.
**Decision rules.**
- "Looping makes the jump more likely at this budget": looped learns in at least 4 of 5 and dense in at most 1 of 5.
- "Dense is only slower": dense learns in at least 3 of 5, all later than the looped median.
- "No difference": the counts differ by at most 1.
- Anything else is mixed.
**Limits.** One learning rate; five seeds give a coarse estimate only; no narrow dense control.
