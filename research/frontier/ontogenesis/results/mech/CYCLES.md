# Mechanism cycles: results log (exploratory unless marked confirmatory)

## Cycle 1: budgeted exact memory on MQAR (exploratory), 2026-11-23

Setup: `tpu/mech_mqar.py` at commit 83c54dc. 16 key-value pairs, 8 queries, vocabulary 128, sequence 48 tokens. Models: d=64, 4 heads, 2 layers, short causal conv; 4,000 steps of AdamW. 3 seeds (0, 1, 2), the same data stream for every arm within a seed; evaluation on 16,384 answers per run. Raw results: `mqar-cycle1.json`.

Mean exact-answer accuracy (number of seeds that learned the task, accuracy > 0.9):

| Arm | M = 8 | M = 16 | M = 32 |
|---|---|---|---|
| attn (no budget) | 1.000 (3/3) | | |
| gdn (no memory) | 0.660 (2/3) | | |
| mem, recency | 0.379 (1/3) | 0.082 (0/3) | 0.082 (0/3) |
| mem, learned (teacher-guided) | 0.972 (3/3) | 0.788 (2/3) | 0.381 (1/3) |
| mem, oracle | 1.000 (3/3) | 0.737 (2/3) | 0.140 (0/3) |

Learned policy, share of kept tokens that are value tokens (0.5 = indifferent): M=8: 0.65, 0.56, 0.58; M=16: 0.59, 0.55, 0.54; M=32: 0.50 (all pair tokens fit).

**Reading: inconclusive.**
- **Outcomes are bimodal.** Each run either learns (about 0.9–1.0) or stays near 0.08. With 3 seeds, the means mostly count successes.
- **The pilot was misleading.** The single-seed "GDN 8%" figure was an unlucky seed; GDN learned in 2 of 3 seeds. So the recall gap between attention and GDN is *not* established at this size.
- **Larger budgets did worse for the oracle** (M=32: 0/3). That shows the outcome is driven by optimisation difficulty in the read layer, not memory capacity.
- **One hint.** At M=8 the learned policy succeeded in 3/3 seeds against recency's 1/3, and it kept value tokens more often than chance. Not significant with 3 seeds.

**What changed in the design:**
- Success rates must be measured with many seeds.
- The memory read layer needs an easier optimisation path.
- **Next (cycle 1b):** 8 seeds, 8,000 steps, M in {8, 16}; report the success rate (with a Wilson interval) and accuracy among successful runs.
