# Protocol: does weight-tied looping help two-step chains? (committed before any of these fits)

Pilot that prompted this (seed 1000, not part of the test): looped 98.2%, dense 21.7% two-step accuracy after 15 minutes.

**Task.** BINDINGS, symbols 512, live 48, churn 2, hops {1, 2}, 2 groups, 16 queries per group.
**Budget.** 13,000 optimiser steps, batch 32 (the pilots took about 12,700–12,900 steps in 15 minutes). Step-matched so that other jobs on the machine cannot bias the comparison.
**Optimiser.** AdamW, learning rate 1e-3, 300 warm-up steps, gradient clipping 1.0, for every arm. No per-arm tuning in this test; that is a stated limitation.
**Seeds.** 11, 12, 13.
**Arms.**
- `dense`: 4 distinct layers, width 64 (about 278k parameters).
- `looped`: 1 layer applied 4 times, width 64 (about 129k). Same compute per step as `dense`.
- `dense1`: the same single layer applied once. Tests whether looping matters or only the smaller model.
- `looped_wide`: 1 layer applied 4 times, width 96. Parameters close to `dense`.
- `looped2x2`: 2 layers applied twice. Intermediate sharing.
**Primary metric.** Two-step accuracy at the end, on 128 held-out sequences (evaluation seed = training seed + 7).
**Secondary.** Steps to 90% two-step accuracy, from an evaluation every 1,000 steps.
**Decision rules.**
- "Looping advantage replicated": `looped` exceeds `dense` by at least 30 points in all three seeds.
- "It is model size, not looping": `dense1` within 10 points of `looped` in at least two seeds.
- "Not replicated": `dense` within 10 points of `looped` in at least two seeds.
- Anything else is reported as mixed.
**Not tested here.** Longer budgets for dense, per-arm learning-rate sweeps, longer chains. A pass shows faster learning at this budget, not that dense cannot learn the task.
