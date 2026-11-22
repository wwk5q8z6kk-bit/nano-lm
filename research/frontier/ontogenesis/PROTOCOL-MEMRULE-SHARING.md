# Protocol: does spreading a fixed-address memory's reads cost generalisation?

Written 2026-11-22, before any confirmatory run. Follows [RESULTS-MEMSAT-SHARING.md](RESULTS-MEMSAT-SHARING.md) (`b292420`), which showed that spreading half of each query's 8 slots to random slots restores acquisition in a world of arbitrary facts. That world cannot show a cost, because arbitrary facts have nothing to generalise.

## Why a new memory variant was needed

`cab.py`'s fixed-address memory computes addresses from random, frozen entity vectors (`cab.py` line 55 freezes every address parameter). Its addresses carry no type information, so in `cab_s.py`'s rule world it could not generalise through its addresses no matter how they were spread. Running it there would have been an empty test. `memory_rule_sharing.py` therefore adds one variant: entity vectors are a hidden-type prototype plus noise of equal expected norm ("typed"), next to the original random vectors ("random"). The type information is given to the memory for free. This is a mechanism probe of the trade-off, not a learner to compare with `cab_s.py`'s.

## What I had seen when writing this

Development seed 4100 only, one run per arm, not results:

| Arm | Effective slots | Late acquisition | Late inference on unseen relations of new entities | Late retention | Late correction |
|---|---|---|---|---|---|
| random, M = 0 | 3,401 | 0.666 | 0.103 | 0.637 | 0.225 |
| typed, M = 0 | 2,117 | 0.850 | 0.740 | 0.783 | 0.038 |
| typed, M = 8 | 14,844 | 0.999 | 0.036 | 0.964 | 0.500 |

So in this world the typed addresses generalise and fully random slots destroy it. No other dose, seed or arm had been run.

## Design

World and memory as in `memory_rule_sharing.py`: `cab_s.py`'s world (16 hidden types, 8 relations, 32 values, 10% exceptions, 40 phases of 64 new entities with 3 of 8 relations shown, 16 corrections per phase); 16,384 slots read 8 at a time; value table trained by Adam at 1e-3 for 200 steps per phase, reset each phase. Late = mean over the last 10 phases, as in `cab_s.py`. Confirmatory seeds 4101–4105.

Arms: typed with M ∈ {0, 2, 4, 6, 8}; random with M = 0. Thirty runs.

Endpoints: late acquisition (this phase's shown facts) and late inference on the unseen relations of this phase's entities. Secondary: retention, inference on earlier entities, corrections.

## Predictions

"Holds" means on at least four of the five seeds.

- H1. Typed M = 0 inference is at least 0.50; random M = 0 inference is at most 0.15. The probe has structure to lose.
- H2. Typed M = 8 inference is at most 0.10. Fully random slots destroy generalisation.
- H3. As M rises, inference never rises by more than 0.02 per step and acquisition never falls by more than 0.02 per step.
- H4, the question this test exists for, with no prediction of its answer: some M in {2, 4, 6} gives acquisition at least 0.95 while keeping inference at least 0.8 of that seed's typed M = 0 inference. If no dose does, spreading reads trades generalisation for acquisition one for one here, and the arbitrary-fact result does not carry over as a free remedy.

## Limits, stated in advance

- The type information is an oracle. Nothing here says a memory could learn such addresses.
- Thresholds were set after one development seed.
- A value table, frozen addresses and a synthetic world. Not evidence about a language model.
- Result files are created exclusively under `results/memory_rule_sharing/`.
