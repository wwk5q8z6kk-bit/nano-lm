# Follow-up protocol: dense attention at M21's training time

Committed 2026-11-20 before any fit described here. The token-matched trial ([RESULTS.md](RESULTS.md)) found M21 ahead of the dense Transformer by 0.850 bits per token, with M21 training several times slower. This follow-up asks whether M21 still leads when the dense model is given the same training time on the same device.

**Calibration.** On an otherwise idle machine, measure training throughput of each architecture on the training split only: 40 unrecorded warm-up updates, then 300 timed updates, in the order M21, dense, M21, dense, on MPS in fp32 with the original batch (8 × 128). The ratio R is mean dense tokens per second over mean M21 tokens per second. The dense update count is S = 4,000 × R rounded to the nearest 100. Calibration never touches the validation split. R and S are written to `results/time-matched/calibration.json` and are final.

**Fits.** Train the same dense GQA architecture, constructor seeding, optimizer, peak learning rate, clipping and batch as the original trial for S updates on seeds 0, 1, 2. Warm up over the first 10% of updates and cosine-decay to 10% of peak, the original shape stretched to S. Sample windows with the original sampler and seed rule, drawing S rows. All three fits finish before any is scored.

**Scoring.** Use the unchanged evaluator on the same 516,033 held-out targets. Compare against the three stored M21 results; no M21 model is retrained or rescored. Gain is time-matched dense bits per token minus M21 bits per token.

**Rule.** M21's lead survives equal training time when all three seeds have positive pooled gain, the mean gain is at least 0.02 bits per token, and the 95% repository bootstrap interval (329 repositories, 10,000 samples, PCG64 seed 22026, seed pairs retained) has a lower endpoint above zero. If the interval's upper endpoint is below zero, dense leads at equal time. Otherwise the result is inconclusive. All cells are reported.

**Limits known in advance.** The held-out repositories were already scored once for the token-matched trial, so this is a second look at the same validation data; nothing is tuned on it, and S is fixed by calibration alone. Throughput is specific to this device and these implementations, neither of which is optimized. Models remain 5M parameters on one code distribution.
