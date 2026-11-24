# Address × optimiser factorial: results

Run 2026-11-21 under [PROTOCOL-MEMSAT-FACTORIAL.md](PROTOCOL-MEMSAT-FACTORIAL.md), committed in `b82d0f0` before any confirmatory run. Seeds 3101–3105. Raw files: `results/memory_saturation_factorial/`. Scored by `memory_saturation_factorial_score.py`, which applies the registered rules and nothing else. [Observed]

## Late acquisition (new-fact accuracy, phases 46–60)

| Cell | 3101 | 3102 | 3103 | 3104 | 3105 | Mean | Mean late retention |
|---|---|---|---|---|---|---|---|
| Original addresses, Adam carried | 0.019 | 0.021 | 0.028 | 0.023 | 0.029 | 0.024 | 0.588 |
| Original addresses, Adam reset each phase | 0.547 | 0.630 | 0.640 | 0.665 | 0.694 | 0.635 | 0.600 |
| Hashed addresses, Adam carried | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.892 |
| Hashed addresses, Adam reset each phase | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.997 |
| Original addresses, Adam with β2 = 0.9 | 0.334 | 0.406 | 0.433 | 0.439 | 0.482 | 0.419 | 0.585 |
| Original addresses, plain gradient descent, rate 1 | 0.495 | 0.575 | 0.610 | 0.595 | 0.648 | 0.585 | 0.556 |

Learning rate for gradient descent, chosen by the registered rule on seed 3001 with hashed addresses: all of 1, 3, 10, 30, 100 and 300 reached 1.000, so the smallest, 1, was used (`dev-sgd-selection.json`).

## Against the registered predictions

- F1 (the collapse replicates, below 0.20): **held**, five of five seeds, 0.019–0.029.
- F2 (hashed addresses stay above 0.90): **held**, five of five, all 1.000.
- F3 (resetting Adam restores acquisition to at least 0.70): **failed**, none of five. Range 0.547–0.694.
- F4 (resetting makes no difference to acquisition under hashed addresses): **held**, identical on every seed.
- F5 (β2 = 0.9 restores acquisition to at least 0.70): **partial by the registered rule**. No seed reached 0.70 and none fell below 0.30. Nothing is claimed.
- F6 (plain gradient descent restores acquisition to at least 0.70): **failed**, none of five. Range 0.495–0.648. The arm was informative: the same rate fits hashed addresses perfectly.

## Reading, by the outcomes the protocol named

The interaction is real but the earlier account of it was too strong. Carried Adam state is a large part of the collapse: removing it moves late acquisition from 2% to 64%. It is not all of it. With no optimiser state of any kind, gradient descent on the original addresses reaches 59%, and no optimiser arm reaches the 70% bar, while every hashed arm is at 100%. So concentrated addresses limit acquisition on their own, by roughly 36 to 42 points here, and carried optimiser state takes away most of what is left.

Consequences for earlier statements:

- "Two interventions each remove most of the collapse" (`INVESTIGATION.md`, Question 7, one seed): hashing does; resetting removes about two thirds of it. The 86.1% on seed 3001 was above every one of the five fresh seeds.
- "With the optimiser reset, the same fixed addresses acquire 86% and retain 83%": on fresh seeds they acquire 55–69% and retain 50–66%. Resetting did not improve retention with the original addresses (0.600 against 0.588).
- The second-moment hypothesis is not supported as the whole mechanism. Shortening the second-moment memory recovers less (42%) than removing optimiser state (64%) or using no adaptive scaling at all (59%). It is not refuted either; F5 is partial.

## Not predicted

- Resetting Adam raised late retention under hashed addresses from 0.892 to 0.997 on every seed, with acquisition unchanged. No prediction covered retention, so this is an observation to test, not a result.
- Seed 3101 is the lowest in every original-address arm and 3105 the highest. The ordering of seeds is nearly the same across four optimisers, which points at the address draw rather than the optimiser. Seed 3001's early acquisition with carried Adam (0.881) was also above all five fresh seeds (0.722–0.787). The measured concentration of each seed's addresses was not recorded in this run; it should be.

## Deviations and limits

- The driver process was killed once, about an hour in, when the controlling session ended a turn. Twenty-one runs had finished. Three runs (`hash/adam_reset`, seeds 3101–3103) were in progress with empty result files and no value printed; they were restarted from the beginning with the same arguments. No completed result was rerun or replaced. `driver.log.run1` is the log up to the kill.
- The 0.70 thresholds were set by me after seeing the one-seed figure of 86.1%. They are a fair test of that figure and an arbitrary line otherwise. Effect sizes are in the table.
- The laptop was heavily loaded (load average above 30). Recorded seconds are not measurements.
- A value table trained by gradient steps on arbitrary facts. An exact lookup table solves this task. Nothing here is evidence about a language model.
- Hashing changes the query representation as well as the address distribution.
- Loss of plasticity through optimiser state is published. This is a diagnosis of our own learner.

## What would decide the remaining question

Whether the residual 36–42 points is collision between facts sharing slots or something else about the projection addresses. Two cheap arms: original addresses with each query's slot weights kept but the slot indices permuted per relation to break sharing; and hashed addresses with sharing deliberately raised to the measured 0.60. Record effective slot count and pairwise overlap per seed in both.
