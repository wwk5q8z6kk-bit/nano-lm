# Protocol: address × optimiser factorial for the acquisition collapse

Written 2026-11-21, before any confirmatory run. Base commit `46fd538` (`research/ontogenesis-phase1`). This is the next gate the ledger names for D-008 ("multi-seed address × optimizer factorial") and tests the untested mechanism recorded in `INVESTIGATION.md`, Question 7.

## What is already known, and what I had seen when writing this

- `memory_saturation.py`, one seed (3001): late acquisition 3.4% with the original addresses, 99.9% with hashed addresses, 86.1% with the optimiser state reset each phase, 59.9% at ten times the learning rate.
- Parity run, not a result: `memory_saturation_factorial.py original adam 3001` reproduced the recorded figures (late acquisition 0.0344, late retention 0.6387, early acquisition 0.8812) in 130 seconds on the laptop CPU with two threads. No other cell and no other seed had been run when this file was committed.
- Seed 3001 is exploratory from here on. It is used only for the parity run above and the learning-rate selection below.

## Question

Is the collapse an interaction between concentrated addresses and optimiser state carried across phases, and is the carried second-moment estimate the mechanism?

## Design

Script `memory_saturation_factorial.py`. Stream, model, step budget and metrics are those of `memory_saturation.py`: 1,024 entities, 8 relations, 60 phases, 128 new facts and 16 corrections per phase, 200 steps per phase, 16,384 slots read 8 at a time. Confirmatory seeds 3101–3105. The unit of analysis is the seed.

Factorial, learning rate 1e-3:

| | Adam, state carried | Adam, state reset each phase |
|---|---|---|
| Original addresses | `original/adam` | `original/adam_reset` |
| Hashed addresses | `hash/adam` | `hash/adam_reset` |

Mechanism arms, original addresses, state carried:

- `original/adam_b2_09`: AdamW with β2 = 0.9 instead of 0.999, learning rate 1e-3.
- `original/sgd`: plain gradient descent, no momentum.

Learning rate for gradient descent. It has no sensible default here, so it is chosen without looking at the cell under test: run `hash/sgd` on seed 3001 at 1, 3, 10, 30, 100 and 300, and take the rate with the highest late acquisition, the smaller rate on a tie within 0.01. If no rate reaches 0.9 on hashed addresses, gradient descent cannot fit this task in 200 steps even without congestion, and F6 is reported as uninformative rather than failed.

Primary endpoint: late acquisition, the mean accuracy on each phase's new facts over phases 46–60. Secondary: late retention of earlier facts over the same phases. All per-seed values are reported, including any run that fails to finish.

## Predictions

"Holds" means on at least four of the five seeds.

- F1. `original/adam` late acquisition is below 0.20. The collapse replicates.
- F2. `hash/adam` late acquisition is above 0.90.
- F3. `original/adam_reset` late acquisition is at least 0.70.
- F4. `hash/adam_reset` is within 0.05 of `hash/adam` on every seed. Together with F1–F3 this is the interaction: resetting matters only where addresses are concentrated.
- F5. `original/adam_b2_09` late acquisition is at least 0.70. If it is below 0.30 on at least four seeds, the second-moment hypothesis is refuted as stated. Between the two it is reported as partial and nothing is claimed.
- F6. `original/sgd` late acquisition is at least 0.70, subject to the selection rule above.

## What each outcome would mean

- F1–F4 hold: the one-seed reading stands across seeds. The collapse needs both ingredients.
- F3 fails: the one-seed reset result was a seed effect, and "fixed addresses trade retention for acquisition" is back on the table.
- F5 and F6 hold: the carried second-moment estimate is sufficient to explain the optimiser ingredient.
- F5 fails and F6 holds, or the reverse: carried state matters but the second moment is not the whole story (first moment, or step size).
- Both fail while F3 holds: resetting helps for a reason other than the one recorded. The hypothesis is wrong and stays open.

## Limits, stated in advance

- A value table trained by gradient steps on arbitrary facts. Not a language model and not evidence about one.
- An exact lookup table solves this task. Nothing here is a capability claim.
- Hashing changes the query representation as well as the address distribution, as recorded before.
- β2 = 0.9 also changes the effective step size early in each phase. A pass on F5 is consistent with the hypothesis, not proof of it.
- Loss of plasticity through optimiser state is published. This is a diagnosis of our own learner.
- Laptop CPU, two threads per run, other jobs present. Wall-clock times are not measurements of efficiency.
- Results are written with exclusive creation under `results/memory_saturation_factorial/` and are never overwritten.
