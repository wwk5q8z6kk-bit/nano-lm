# Protocol: per-arm learning rates for the nested experiment (committed before any of these fits)

**Why.** The nested result (`INVESTIGATION.md`, Question 12 section "Results of 22 November"; `PROTOCOL-NESTED.md`) found that arms reusing weights learned two-step chains in 14 of 20 runs, and arms without reuse in 1 of 15. All arms used one learning rate, 1e-3. The closest published study reports untied models failing at 1e-3 but succeeding at 3e-4 [from memory, not re-checked]. So the reuse advantage may be tolerance of one learning rate rather than an effect of reuse. This is the control named in that result's caveats.

**Question.** Does the reuse advantage survive when each arm gets its best of three learning rates?

**Arms:** two compute-matched pairs from `PROTOCOL-NESTED.md`, with reuse and without.

| Pair | Reuse | No reuse | Layer applications |
|---|---|---|---|
| A | `looped` (1 × 4, 64) | `dense` (4 × 1, 64) | 4 |
| B | `tf-4x2-d64` | `tf-8x1-d64` | 8 |

**Learning rates.** 3e-4 and 3e-3, new. The 1e-3 results are the existing cloud-CPU runs in `results/nested/`, which used the same device class (one CPU thread on a TPU host). Every other setting is unchanged from `PROTOCOL-NESTED.md`:
- BINDINGS, 512 symbols, 48 live, churn 2, hops {1, 2};
- AdamW, 300 warm-up steps, clip 1.0;
- 26,000 steps, evaluation every 1,000;
- seeds 21–25;
- same `train.py`, PyTorch 2.11.0 (CPU), NumPy 2.5.1, one thread per run.

40 new runs.

**Primary metric**, unchanged: a run "learned" if two-step accuracy reaches at least 90% at any evaluation.

**Predictions** (counts out of 5):
- L1: `dense` at 3e-4 learns in at least 3.
- L2: `tf-8x1-d64` at 3e-4 learns in at least 3.
- L3: at 3e-3 no arm learns in more seeds than it did at 1e-3.

**Decision rule.** Each arm's best count is its maximum over the three rates.
- "Reuse advantage stands" if, in both pairs, the reuse arm's best count is at least 2 above the no-reuse arm's.
- "Explained by learning rate" if, in both pairs, the no-reuse arm's best count is within 1 of the reuse arm's.
- Otherwise "mixed". The pair-level outcome is then reported for each pair.

A one-seed difference is not a finding. Taking each arm's maximum over three rates on the same five seeds favours every arm alike; this design cannot remove that selection effect, and the result is stated with it.

**Execution.** The host CPU of the spot TPU node that is training the 386M model (Question 13). The 386M training runs on the TPU chips; these runs use about 40 of the host's 112 cores.
- If the 386M model's throughput (105,900 tokens per second) falls by more than 2% after these runs start, they are stopped, and that is recorded.
- If the node is reclaimed, unfinished runs are restarted from the beginning on the replacement node. `train.py` has no resume, and a restart is the same computation.
- Results are copied to `gs://nanolm-lab-eu-0478806098/runs/nested-lr/` as each run finishes.
