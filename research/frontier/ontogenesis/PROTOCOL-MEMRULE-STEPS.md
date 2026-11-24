# Protocol: do corrections in the shared-path memory rise with the step budget?

Written 2026-11-22, before any run of this sweep. Follows [RESULTS-MEMRULE-DUAL.md](RESULTS-MEMRULE-DUAL.md) (`a2f61a5`).

## Why

Two accounts of why corrections fail in the typed shared-path memory are on record:
- **Step budget** (my account in `PROTOCOL-MEMRULE-DUAL.md`): a correction must outvote neighbours' accumulated values within one phase's 200 steps.
- **Conflict inside one objective** (raised in `RESULTS-MEMRULE-DUAL.md`): in the same phase, the corrected fact shares slots with new same-type facts, and the summed loss is won by the majority however long training runs.

A third, different limit was found by another session in a frozen nonlinear decoder: reachability, flat from 100 to 3,000 steps (commit `d6e05c3` on `research/g06-nullspace-durability`). That session proposed the falsifier used here: a budget limit rises with steps; the other two do not.

## What I had seen when writing this

- All results of the two-path test. In particular, the single shared path at ten times the learning rate applied corrections no more often (0.034 at 1e-3, 0.042 at 1e-2). That already counts against the budget account, and the prediction below was written knowing it.
- No run with any step count other than 200.

## Design

`memory_rule_dual.py shared_only`, learning rate 1e-3, **1,000 steps per phase**, seeds 4101–4105. Everything else as before; Adam reset each phase. The 200-step result is the existing typed M = 0 arm (`results/memory_rule_sharing/typed-m0-s*.json`); `shared_only` at 200 steps reproduced it exactly on the development seed. Five runs.

Only one new step count is run. The machine is shared and heavily loaded (load average above 300 when this was written), and a 3,000-step arm would take several hours.

## Prediction

"Holds" means on at least four of the five seeds.

- S1. Corrections at 1,000 steps are no more than 0.05 above that seed's 200-step value. If S1 holds, the step-budget account is refuted for this memory. If corrections rise by more than 0.10 on at least four seeds, the budget account stands and the conflict account is in doubt. Between the two, nothing is claimed.
- Recorded, no prediction: acquisition, inference, retention.

## Limits

- One additional step count. A rise that only begins beyond 1,000 steps would be missed.
- Steps and learning rate both change how far values move; together with the learning-rate observation this is two partial probes of one question, not a full sweep.
- Same synthetic world and given type information as before. Not evidence about a language model.
- Results under `results/memory_rule_steps/`, created exclusively.

## Addendum: a flaw in the interpretation, recorded before any sweep result was read (2026-11-22)

One of the five runs had finished when this was written; its file has not been opened, and no value from it has been printed.

The sweep changes the step count for **every** phase, not just for the correction. Adam moves each touched value by roughly the learning rate per step whatever the gradient's size. So the values that same-type neighbours accumulate in shared slots grow with the step count by about the same factor as a correction's own movement does. If the budget account is right, what matters is the correction's budget **relative** to the neighbours' accumulated mass, and a uniform change in steps (or in learning rate) leaves that ratio roughly unchanged. Under that reading, S1 holding would **not** refute the budget account, and the learning-rate observation in `RESULTS-MEMRULE-DUAL.md` does not count against it either.

This came from a development-seed pilot of the write-pass test (`memory_rule_writepass.py`, seed 4100). Written alone in the loss with the phase's step count and learning rate, corrections moved only from 0.031 to 0.038, and nothing else changed (collateral zero to three decimals). A write that moves nothing is what a budget limit predicts when the correction gets the same budget as each earlier write.

The sweep's runs will finish and be reported as registered, but S1 will be read under this addendum: it tests whether a uniform change in budget matters, not whether a correction's relative budget does. The test of the relative budget is the write-pass protocol, where only the correction pass gets more steps.
