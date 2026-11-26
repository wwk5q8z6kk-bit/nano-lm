# Dense attention at M21's training time: results

Run 2026-11-20 under [TIME_MATCHED_PROTOCOL.md](TIME_MATCHED_PROTOCOL.md). Calibration on the training split measured dense at 5.28 times M21's throughput, fixing the dense arm at 21,100 updates (M21: 4,000). Both the protocol and the update count were committed before any fit.

| Seed | M21 (4,000 updates) | Dense (21,100 updates) | Dense − M21 |
|---|---:|---:|---:|
| 0 | 5.0088 | 4.6538 | −0.3550 |
| 1 | 5.0277 | 4.6548 | −0.3728 |
| 2 | 5.0234 | 4.6334 | −0.3900 |
| **Mean** | | | **−0.3726** |

Bits per token on the same 516,033 held-out targets; lower is better. Dense is ahead on every seed. The repository-level mean is −0.482, with a 95% bootstrap interval of −0.520 to −0.444. Under the prespecified rule, **dense leads at equal training time**. The earlier token-matched advantage for M21 (0.850) does not survive when the faster model is allowed to use the time.

| Targets | M21 | Dense, time-matched | Dense, token-matched |
|---|---:|---:|---:|
| Correct token is a pointer candidate | 1.790 | 2.150 | 4.006 |
| Correct token is not a pointer candidate | 7.677 | 6.702 | 7.403 |

M21 still predicts copyable tokens better (by 0.36 bits per token). Dense closes most of that gap with more updates and is 0.97 bits per token better everywhere else.

## Limits

- Wall time was matched by calibration, not enforced. Dense fits took 697, 659 and 547 seconds against 529, 536 and 607 for M21, so dense had somewhat more time on two seeds and less on one. The gap is far larger than that difference would explain, and seed 2, where dense had less time, shows the largest dense lead.
- This is a second look at the same held-out repositories. Nothing was tuned on them.
- Throughput is specific to this device and to two unoptimized implementations. A faster M21 implementation would change the exchange rate.
- 5M parameters, one code distribution, three seeds.

[Summary](results/time-matched/summary.json) · [Calibration](results/time-matched/calibration.json)
