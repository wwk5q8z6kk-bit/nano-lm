# Forward-transfer development pilot: result (2026-11-21, gate G07)

`forward_transfer.py`. Predictions T1–T7 committed in `bef6608` before any run; deviation 1 (menu arm redefined after it failed to fit its stream on an unregistered seed) committed in `30cf0e1` before any registered run; scorer committed in `4d281ef` before any result was read. Seeds 9101–9103. Raw files: `results/forward_transfer/`; scored output: `results/forward_transfer/verdict.json`. [Observed]

**Question.** After training on 24 task families built from 6 hidden primitives, does a learner need fewer examples for a new family built from the same primitives?

**Answer in this world: no, for every arm.** Accuracy on unshown inputs of new combinations stayed at guessing level (1/23 = 4.3%) however much experience the learner had, including for the arm whose structure was designed for the task.

## Primary measure

New combinations of known primitives, 8 examples shown, only the new task vector adapted, after 24 families. Mean of three seeds.

| Arm | Parameters | Stream still answered correctly | New combinations |
|---|---|---|---|
| Fresh network, no experience (all weights trained) | 24,599 | — | 0.0% |
| Dense, one family at a time | 24,599 | 8.5% | 4.4% |
| Dense, with replay | 24,599 | 100% | 2.8% |
| Loop, one family at a time | 20,631 | 8.5% | 3.6% |
| Loop, with replay | 20,631 | 99.8% | 4.2% |
| Menu, one family at a time | 8,464 | 64.6% | 2.8% |
| Menu, with replay | 8,464 | 88.5% | 4.2% |
| *Menu with the true primitives set by hand (post hoc, untrained)* | 8,464 | — | *71.7%* |

No curve rises with experience: at 0, 2, 4, 8, 16 and 24 families every arm is between 0% and 6%. Showing 16 examples instead of 8 changes nothing. Adapting all weights instead of the task vector changes nothing.

## Against the registered predictions

- T1 (a fresh network cannot generalise to unshown inputs): **held**, and more strongly than predicted: exactly 0% in every cell. The targets are permutations, so a network that answers with outputs it has already seen is always wrong.
- T2 (dense without replay shows no transfer): **held**, 4.4%.
- T3 (dense with replay transfers weakly, 15–50%): **failed**, 2.8%. Perfect multi-task fit of 24 combinations gave no reusable parts.
- T4 (loop with replay reaches 60%): **failed**, 4.2%. A shared two-step block with a perfect fit did not factor the families into steps that a new task vector could recombine.
- T5 (menu with replay near the ceiling): **failed**, 4.2% at 8 examples and 4.6% at 2.
- T6 (transfer grows with experience): **failed**; there was nothing to grow.
- T7 (nothing transfers to never-used primitives): **held**; no arm above 8.3%.

Under the reading written before the run, the T4 branch does not apply either: not even the menu transferred. Handing over the structure was not sufficient. The menu's learned tables are not the primitives (measured in the positive control below).

## Instrument check (post hoc)

Because every arm is at guessing level, the probe was checked after the results were read (`forward_transfer_oracle.py`): the menu's tables were set by hand to the true primitives and the registered probe applied unchanged. It scored 25–62% at 2 examples, 53–88% at 8 and 88–100% at 16 on new combinations, and stayed at guessing level on never-used primitives. So the probe detects transfer when the parts exist, and the null result for the trained arms is a property of what they learned. It also shows the probe's own adaptation step is lossy: with perfect primitives it reaches 72% at 8 examples, so T5's 90% bound could not have been met by any arm. That bound was mis-set.

## Positive control (post hoc)

The hand-set menu validates the probe only for the menu architecture. For dense and loop, a guessing-level score could also mean that adapting a zero task vector simply fails. `forward_transfer_control.py` (written after the results were read) retrains each replay arm as registered and applies the unchanged probe to 8 families the arm **was trained on**, where a suitable task vector certainly exists. It also logs accuracy on the examples that were shown. Raw file: `results/forward_transfer/control.json`. Mean of three seeds.

| Arm (with replay) | Trained family, unshown inputs, 8 / 16 examples | New combination, unshown inputs, 8 / 16 | Trained family, the shown examples, 8 | New combination, the shown examples, 8 / 16 |
|---|---|---|---|---|
| Dense | 41% / 66% | 3% / 3% | 89% | 34% / 16% |
| Loop | 42% / 64% | 4% / 4% | 86% | 37% / 13% |
| Menu | 24% / 38% | 4% / 4% | 67% | 55% / 32% |

- For dense and loop the probe does find a family the network already holds, well above guessing, so it can read these arms. It is lossy: 41% at 8 examples, not 100%.
- On a new combination the task vector cannot even fit the examples it is shown, and the more examples, the worse the fit (16% and 13% of 16). No setting of the task vector makes these networks compute a new combination. That is direct evidence of no recombinable parts, and stronger than the guessing-level score alone.
- Starting the menu's probe vector at random instead of zero changes nothing (4% on new combinations), so a dead start is not the explanation for the menu.
- The menu's learned tables agree with their best-matching true primitive on 4–26% of inputs (guessing is 4%). The tables are not the primitives.
- The probe is weak for the menu even on trained families (24% at 8 examples), so the menu's result is less well supported than the dense and loop results.

## What this does and does not establish

- **Observed:** fitting a set of compositions, even perfectly and with a matching architecture, did not yield parts that recombine. Fit and reuse came apart completely here.
- **Likely cause, partly tested:** a composition f_b(f_a(x)) can be split into two steps in many ways, and nothing in the training loss prefers the split that is shared across families. Consistent with this, the menu's tables do not match the primitives. Why dense and loop fail was not examined beyond the control above.
- **Not established:** that forward transfer is impossible for these arms. No arm was trained with an objective or episode structure that rewards reuse (for example, meta-training on held-out combinations), and no learning-rate or adaptation search was done.
- One synthetic world, three seeds, two-step compositions only. Nothing here is a language-model result.

## Next discriminating test

Train the same arms so that reuse is what the loss rewards: each step, adapt a task vector on a few examples of a held-out combination and back-propagate the query loss. If the loop or dense arm then transfers, the missing ingredient is the training signal, not the architecture. If only the menu does, structure plus signal are both needed. The probe's adaptation step should be tightened first, using the hand-set menu as the calibration target.
