# Protocol: nested dense-loop computation (committed before any of these fits)

**Idea under test (owner's proposal, 2026-11-21).** Loop on the outside, dense on the inside: a block of several distinct layers applied repeatedly. Treat inner depth and outer repetition as two separate dimensions and ask whether they interact.

**Prior art, stated so no novelty is implied.** Repeating a block of layers with shared weights is published (Universal Transformer, ALBERT, looped Transformers, deep equilibrium models; learned halting in ACT and PonderNet). [Published, from memory, not re-checked] Our own `looped2x2` arm was already this design and learned two-step chains in 2 of 3 seeds. What is open here is narrower: on this task, does the combination do something neither dimension does alone, with controls for size and compute.

**Task, optimiser.** Identical to `PROTOCOL-LOOP.md` and `PROTOCOL-LOOP2.md`: BINDINGS, 512 symbols, 48 live, churn 2, hops {1, 2}; AdamW 1e-3, 300 warm-up steps, clip 1.0, no per-arm tuning.
**Budget.** 26,000 steps, evaluation every 1,000. **Seeds.** 21–25.

**Arms** (layers × loops, width; parameters; layer applications per pass).

| Arm | Design | Parameters | Applications | Role |
|---|---|---|---|---|
| `dense` | 4 × 1, 64 | 278k | 4 | reused from LOOP2; four seeds already known (0 of 4 learned) |
| `looped` | 1 × 4, 64 | 129k | 4 | reused from LOOP2; four seeds already known (3 of 4 learned) |
| `tf-2x2-d64` | 2 × 2, 64 | 178k | 4 | nested, same compute as the two above |
| `tf-4x2-d64` | 4 × 2, 64 | 278k | 8 | nested: the whole dense model applied twice |
| `tf-8x1-d64` | 8 × 1, 64 | about 477k | 8 | untied control with the same compute as 4 × 2 |
| `tf-1x8-d64` | 1 × 8, 64 | 129k | 8 | pure loop with the same compute as 4 × 2 |
| `tf-4x1-d40` | 4 × 1, 40 | 125k | 4 | dense narrowed to the looped model's size |

**Primary metric.** A run "learned" if two-step accuracy is at least 90% at any evaluation. Secondary: step of first such evaluation; one-step accuracy.

**Decision rules** (counts out of 5).
- R1 "Reuse matters, not depth or compute": `tf-4x2-d64` learns in at least 4 and `tf-8x1-d64` in at most 1.
- R2 "Nesting adds something beyond plain looping": a nested arm learns in at least 2 more seeds than the pure loop of equal compute (`tf-2x2` against `looped`; `tf-4x2` against `tf-1x8`).
- R3 "Nesting costs reliability": the reverse of R2 by the same margin.
- R4 "It is model size": `tf-4x1-d40` learns in at least 3.
- If neither R2 nor R3 holds: "no evidence that nesting differs from plain looping at this scale". This is the expected outcome if looping and density are simply independent.
- Five seeds can detect only large differences. A difference of one seed is not a finding.

**Recorded after training, no further fitting (descriptive, no pass/fail rule).**
- Accuracy on three-step chains, which are never trained, at the trained number of loops, one more, and double. Tests whether extra outer iterations at test time buy extra steps.
- How far the state still moves at each outer loop (relative change), to see whether iteration settles toward a fixed point.

**Not in this stage.** Learned halting, energy or phase-transition analysis, representation geometry beyond the one measure above, language modelling. These follow only if R1 or R2 holds; otherwise they would be analysis of a non-effect.
