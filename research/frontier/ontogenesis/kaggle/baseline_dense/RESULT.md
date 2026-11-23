# Kaggle dense baseline: result and cross-stack check (recorded 2026-11-23)

Run: private Kaggle kernel `nanolm-baseline-dense-v1`, status COMPLETE, finished 2026-11-22 00:30 UTC. Raw output copied unchanged to `results/kaggle/baseline-dense-v1-s1000.json` (sha256 `f1ef3c4d…f617`). The source the kernel ran is byte-identical to `baseline.py` in this folder (sha256 `91f63686…a6b6`, last changed in `46fd538`). [Observed]

This run was registered as a baseline, with no prediction attached, so there is no verdict to score. It is recorded because `theory/RESEARCH-DECISION-20261121.md` still lists it as running.

## Result

40,379,904 parameters (d 512, 8 layers, FFN 1408, tied embeddings), AdamW 1e-3, warm-up then cosine to 10% of peak, batch 64 × 512, one pass of 800M tokens, seed 1000, PyTorch float16 autocast on one Tesla T4.

| | Web | Textbook | Mathematics | Code | Mix | Wall-clock | Throughput | Peak memory |
|---|---|---|---|---|---|---|---|---|
| Kaggle T4, PyTorch | 1.1095 | 0.7936 | 1.1859 | 0.8787 | **1.0506** | 8.36 h | 26.6k tokens/s | 10.4 GB |

Bits per byte on the held-out crops (`default_rng(0)`, 20 batches of 16 per domain), the same crops `tpu/train_jax.py` uses.

## Cross-check against the TPU harness

The same 40M configuration at the same recipe and token budget was run three times on TPU v6e with `tpu/train_jax.py` (`results/recipe/m-v6e/dense-M-lr0.001-s100{0,1,2}-hs.json`, bfloat16, JAX).

| Domain | TPU v6e, mean of 3 (sd) | Kaggle T4 | Difference |
|---|---|---|---|
| Web | 1.1072 (0.0021) | 1.1095 | +0.0023 |
| Textbook | 0.7910 (0.0006) | 0.7936 | +0.0026 |
| Mathematics | 1.1813 (0.0021) | 1.1859 | +0.0046 |
| Code | 0.8744 (0.0022) | 0.8787 | +0.0043 |
| Mix | 1.0478 (0.0012) | 1.0506 | **+0.0028** |

- **The two harnesses agree to 0.003 bits per byte** on the mix, so a PyTorch number and a JAX number at this scale can be compared directly, with that offset as the cross-stack tolerance.
- The Kaggle run is worse on every domain, by about two TPU seed standard deviations on the mix. One Kaggle seed, so this is a direction, not an established offset.
- The gap opens during training rather than being present from the start. At equal tokens against the TPU seed-1000 run, it is −0.003 at 65M tokens and +0.004 from 525M tokens onward. That pattern fits a numerical difference (float16 with a loss scaler on the T4 against bfloat16 on the TPU; PyTorch warns that RMSNorm received mixed dtypes and fell back to the unfused kernel) better than a difference in data, but neither has been isolated. The crop samplers also differ (`default_rng(seed)` here, `default_rng(seed + 1)` on the TPU).

## Cost

- Kaggle T4: 8.36 hours, free. One TPU v6e chip: 0.47 hours. The free T4 is about 18 times slower in wall-clock for this run.
- At 26.6k tokens per second, one free Kaggle session (12-hour limit) fits about 1.1B tokens at this size. The script saves weights but cannot resume, so a longer run would need resume support first. Kaggle is a cross-check and overflow lane at 40M, not a route to the 100M+ ladder.

## Not established

- Whether the +0.003 is a stable offset: it needs a second Kaggle seed, or the same script run in bfloat16 on a GPU that supports it.
- Nothing about architecture: this is the reference configuration only.
