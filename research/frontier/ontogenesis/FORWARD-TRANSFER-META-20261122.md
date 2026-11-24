# Forward transfer with a reuse-rewarding training signal: result (2026-11-22, gate G07 follow-up)

`forward_transfer_meta.py`. Settings and predictions V1–V5 committed in `1fa5384` before any registered run; scorer committed in `5b0a7b0` before any result existed. Seeds 9101–9103. Raw files: `results/forward_transfer_meta/`; scored output: `results/forward_transfer_meta/verdict.json`. [Observed]

**Question.** The pilot (`FORWARD-TRANSFER-20261121.md`) found that learners which fit 24 two-step task families gain nothing on new combinations of the same primitives. Its stated next test: train so that reuse is what the loss rewards. Here each training step adapts a task vector on a few examples of a family and updates the network on how well that vector answers the rest. Does that produce transfer to new combinations?

**Answer in this world: no.** Few-shot training made all three learners recognise their training families from a handful of examples. New combinations stayed at guessing level (1/23 = 4.3%).

| Learner (few-shot trained) | Parameters | Trained family, 2 / 8 examples | New combination, 2 / 8 / 16 examples | Shown examples fitted on a new combination, 8 / 16 |
|---|---|---|---|---|
| Dense | 24,615 | 90% / 100% | 1.8% / 4.4% / 4.2% | 16% / 11% |
| Loop | 20,647 | 72% / 99% | 4.4% / 5.0% / 4.2% | 15% / 12% |
| Menu | 8,480 | 75% / 86% | 3.6% / 6.4% / 4.8% | 23% / 15% |

Unshown inputs unless stated; mean of three seeds. With the pilot's own probe (Adam, 200 steps, zero start) new combinations score 1.9–5.0%.

## Against the registered predictions

- V1 (positive control: trained families recovered from 8 examples, ≥85% dense and loop, ≥60% menu): **held**, 100%, 98.6%, 85.6%.
- V2 (dense does not transfer, ≤15%): **held**, 4.4%.
- V3 (loop transfers, ≥30%): **failed**, 5.0%.
- V4 (menu transfers, ≥40%): **failed**, 6.4%.
- V5 (nothing transfers to never-used primitives, ≤15% at every k): **held**, at most 11.9%.

The reading registered for this outcome (V1 holding, V2–V4 all failing) applies: few-shot training produces recognition without reusable parts. Compositional transfer here needs something not tried in either study.

## What the numbers show

- **Recognition, not composition.** Two examples identify one of 24 trained families 72–90% of the time. On a new combination the adapted task vector cannot reproduce even the examples it was given, and the more examples it sees, the worse it fits (dense: 71% of 2 shown, 16% of 8, 11% of 16). The task vector space covers the trained families as separate points; it has no direction that composes a seen first step with a seen second step in a new pairing.
- **Architecture did not supply the missing structure.** The loop, which applies one shared block twice with each half of the task vector, behaved like the dense network. The menu, built as two banks of step tables, also did not compose.
- **Both training signals agree.** Plain multi-task fitting (pilot) and few-shot adaptation (here) give the same null on new combinations, with positive controls passing in both.

## Limits

One synthetic world, permutations on 23 inputs, 6 stream primitives, 24 of 36 pairs seen, three seeds, one setting of inner step size and length (chosen on an unregistered seed by training-family accuracy). No combination was ever held out during training, so nothing in either loss distinguished "recognise this family" from "recognise its parts". Not a language-model result.

## Next discriminating tests

1. **Hold out combinations during meta-training:** each step adapts on one set of seen pairs and is scored on a different set that shares their primitives. This rewards composition, not recognition. If the loop or menu then transfers to the registered new combinations, the missing ingredient was a loss that tells the two apart.
2. **Explicit factorisation pressure:** the menu with its two halves of the task vector adapted on separate one-step tasks first, then composed. If this transfers and (1) does not, structure has to be taught step by step.
