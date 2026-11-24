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

## E0a: scorer conditions on our 101M checkpoint (2026-11-23)

Checkpoint `scale1-v6e/dense-L-lr0.001-s1000-tok8e+08-dp4-hs-v2.npz` (sha256 8745652c49cf…). Task file sha256 ec09b3fca312…, the same file as on the bucket (CRC32C match). Run on the laptop CPU with JAX 0.6.2; the stored reference was scored on a TPU. Raw files: `results/e0a/`.

| Change | Largest effect observed | Scope |
|---|---|---|
| CPU against TPU, both bf16 | ≤ 0.003 accuracy; LAMBADA perplexity 141.69 vs 141.76 | this checkpoint, these 7 tasks |
| float32 against bf16, same CPU | up to 0.008 (WinoGrande +0.008; ARC-C acc_norm −0.005) | same |
| No end-of-text prefix | LAMBADA accuracy −0.012 and perplexity 141.7 → 152.6; other tasks ≤ 0.004 | same |
| lm-eval length rule (choice without leading space) | ARC-E acc_norm −0.008; OBQA acc_norm +0.006 | acc_norm only |

**Item-level, bf16 against float32:**
- WinoGrande: 22 of 1,267 items (1.7%) change their argmax. ARC-Challenge: 7 of 1,172 (0.6%).
- The flipped items are near-ties: median float32 margin 0.028 and 0.012 nats, against medians of 0.51 and 1.93 nats over all items.
- bf16 shifts log-probabilities by a median of about 0.03 nats (99th percentile about 0.13).

**Reading:**
- **Hardware is not the main source.** The earlier "0.003 floor" is withdrawn. Precision moves scores more than hardware does, and how much depends on how many near-ties a benchmark has.
- **The end-of-text prefix is a real bias in our favour** on LAMBADA (about 1.2 points of accuracy) when comparing with the Hugging Face models, which were scored without it.
- **Sources of uncertainty stay separate:** implementation and precision (this table), seed variation (measured per experiment), benchmark sampling (bootstrap), and evaluation configuration (prefix, length rule).

**Decisions:**
- Score reference comparisons in float32.
- Score our models both with and without the prefix when comparing with public models.
- Report acc_norm with the lm-eval length rule.
