# Matched M21 versus dense attention on held-out code

Run 2026-11-20 under [protocol.md](protocol.md), which was committed before any fit. Two 5.05M-parameter models (native M21: 5,046,637; dense GQA Transformer: 5,046,656) were each trained three times on the same 4.096M sampled predictions per seed, then scored once on 516,033 targets from 329 held-out repositories. All six fits finished before any held-out score was read.

| Seed | M21 bits/token | Dense bits/token | Dense − M21 |
|---|---:|---:|---:|
| 0 | 5.0088 | 6.0684 | 1.0596 |
| 1 | 5.0277 | 5.7765 | 0.7488 |
| 2 | 5.0234 | 5.7645 | 0.7411 |
| **Mean** | | | **0.8499** |

M21 had lower loss on every seed. The mean repository-level gain was 0.706 bits per token, with a 95% repository bootstrap interval of 0.657 to 0.757. The prespecified rule (all seeds positive, mean gain at least 0.02, interval above zero) is met, so the protocol's outcome is a follow-up comparison on a different distribution.

## Where the gain comes from

| Targets | Count | M21 | Dense | M21 base head |
|---|---:|---:|---:|---:|
| Correct token is among M21's pointer candidates | 232,931 | 1.790 | 4.006 | 5.552 |
| Correct token is not a pointer candidate | 283,102 | 7.677 | 7.403 | 7.417 |

The whole advantage is on tokens M21 can copy from earlier in the file. Where copying is unavailable, M21 is 0.27 bits per token worse than dense. M21's base head alone (6.575 pooled) is worse than dense (5.870).

## Cost and limits

- M21 trained at about 7,400 tokens per second against about 40,800 for dense on the same device, roughly 5.5 times slower. At equal wall time the dense model would see about 5.5 times more data; this trial matched tokens, not time.
- Both models are small and briefly trained (losses near 5 to 6 bits per token). A dense model learns to copy with more training, so the gap may shrink at scale. This result does not establish that it persists.
- One code distribution, three seeds, a tokenizer derived from upstream training shards.

An independent CPU replay of eight repository groups per cell agreed with the stored scores to within 0.0000024 bits per token across 72 comparisons. [Summary](results/summary.json) · [Independent replay](results/independent-cpu.json) · [Fit receipts and per-cell scores](results/)
