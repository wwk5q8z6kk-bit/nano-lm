# Corrections written alone, with and without a larger relative budget: results

Run 2026-11-22 under [PROTOCOL-MEMRULE-WRITEPASS.md](PROTOCOL-MEMRULE-WRITEPASS.md), committed in `0126fb7` before any confirmatory run. The write-alone falsifier and the collateral caveat were proposed by another session (nano-lm-d5). Seeds 4101–4105, shared typed path, learning rate 1e-3, 200 steps per phase on new facts; then a correction-only pass. Last 10 of 40 phases. Raw files: `results/memory_rule_writepass/`. [Observed]

## Means over five seeds

| Correction pass | Corrections, before → after the pass | Corrected entities' other facts, before → after | Same-type peers' facts, before → after | Late inference, new entities | Late acquisition | Late retention |
|---|---|---|---|---|---|---|
| 200 steps (same as each phase) | 0.026 → 0.035 | 0.772 → 0.771 | 0.772 → 0.772 | 0.735 | 0.833 | 0.769 |
| 2,000 steps | 0.029 → 0.919 | 0.747 → 0.729 | 0.756 → 0.752 | 0.609 | 0.752 | 0.752 |

Seeds agree closely: corrections after the 2,000-step pass range 0.887–0.944, and the inference gap between the two arms ranges 0.119–0.138.

## Against the registered predictions

- W1 (written alone with the phase's budget, corrections rise by at most 0.05): **held**, five of five. Removing the phase's majority from the loss does not make corrections take.
- W2 (with ten times the steps for the correction pass only, corrections reach at least 0.80): **held**, five of five (0.887–0.944).
- W3 (collateral at writing is small: corrected entities' other facts fall by at most 0.05, same-type peers' by at most 0.02): **held**, five of five (falls of 0.012–0.022 and 0.003–0.004).
- W4 (the cost appears later: late inference on new entities at least 0.05 lower): **held**, five of five (0.12–0.14 lower).

## Reading

Of the three accounts the protocol set out to separate:
- **Conflict inside one objective** is not the main cause: with the majority removed and the same budget, corrections still do not take (W1).
- **Relative budget** is supported: giving only the correction pass ten times the movement makes nine in ten corrections take (W2).
- **Reachability**, the limit another session found in a frozen nonlinear decoder (commit `d6e05c3` on `research/g06-nullspace-durability`), does not apply to this memory: the corrected values are reachable.

The cost is real but delayed. A forced correction barely disturbs other facts when it is written, yet over the following phases inference on new entities falls from 0.74 to 0.61, acquisition from 0.83 to 0.75, and inference on old entities from 0.73 to 0.60. The correction pass's gradient reaches only the 8 slots each corrected fact reads, so the write is already as local as this addressing allows. The cost comes from those slots being shared with same-type entities: the pass raises the new value and lowers the rule's value in rows the type's other members also read. (This interpretation came out of an exchange with nano-lm-d5. It contrasts with its G06b result, where the later cost came from protecting too much shared structure and narrowing the protection removed it.) A narrower write cannot remove a cost of this kind; a different place to write might. In the two-path test, a private slot raised corrections from 0.04 to 0.22 without this pass.

Together with the step sweep (`RESULTS-MEMRULE-STEPS.md`): uniform changes in budget leave corrections flat, and a relative change makes them take. That is what the relative-budget account predicts.

## Limits

- The correction pass is told which facts are corrections, as an explicit edit request would be. A learner that has to detect them itself would not have that label (`INVESTIGATION.md`, Question 8).
- Ten times the steps is one point. Whether a smaller boost buys most corrections with less later cost is not measured.
- Thresholds were set after one development seed.
- Synthetic world, type information given, value table, frozen addresses. Not evidence about a language model.
- The 16 corrections of a phase were written in one pass, not one pass each, as the protocol records.
- All ten runs finished on the first attempt; the machine was shared and loaded, so timings are not measurements.
