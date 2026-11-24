# Dense 40M baseline on Kaggle (PyTorch, T4): cross-framework check of the TPU harness

Kernel `hassaneljesr/nanolm-baseline-dense-v1`, script `kaggle/baseline_dense/baseline.py` (model code copied from
`research/frontier/m21_dense_code/dense.py`). Finished 2026-11-22. Raw output: [result.json](result.json).
The checkpoint (`model.pt`, 161,544,125 bytes, SHA-256 `df21d3e6c385b6ddfa7b1cf204656d4c2fce539f1dca7fe59357c1c39e9d13b6`)
is in the kernel output, not in Git.

## Setup

Same recipe as the TPU 40M baselines in `results/recipe/m-v6e/dense-M-lr0.001-s100{0,1,2}-hs.json`: 40,379,904
parameters (d 512, 8 layers, FFN 1408), AdamW 1e-3, β (0.9, 0.95), weight decay 0.1, 500 warm-up steps, cosine decay to
10%, batch 64 × 512, one pass of the 800M-token v1 corpus with uniform host-side crops, seed 1000. The evaluation is the
same code path in both harnesses: `numpy default_rng(0)`, 20 batches of 16 per domain, bits per byte from the
manifest's bytes-per-token.

Differences: PyTorch with float16 autocast on one Tesla T4 against JAX bfloat16 on TPU v6e; 24,414 steps against
24,400; different initialisation streams for the same seed number.

## Result

Bits per byte at the end of the pass, lower is better:

| Run | Web | Textbook | Mathematics | Code | Mix |
|---|---:|---:|---:|---:|---:|
| Kaggle, PyTorch, T4, seed 1000 | 1.1095 | 0.7936 | 1.1859 | 0.8787 | **1.0506** |
| TPU, JAX, seed 1000 | 1.1050 | 0.7911 | 1.1807 | 0.8747 | 1.0464 |
| TPU, JAX, seed 1001 | 1.1075 | 0.7904 | 1.1837 | 0.8764 | 1.0485 |
| TPU, JAX, seed 1002 | 1.1091 | 0.7916 | 1.1796 | 0.8720 | 1.0486 |

The Kaggle run is 0.0028 above the three-seed TPU mean (1.0478) and 0.0020 above the worst TPU seed; it is higher on
every domain. **The two harnesses agree to within about 0.003 bits per byte**, close to the TPU seed range (0.0022).
With one Kaggle seed, a small framework or precision offset cannot be separated from seed variation. Either way, no
TPU result on this branch that differs by 0.01 or more between arms is explained by the choice of harness.

## Cost

8.36 hours on a free Kaggle T4 at 26,578 tokens per second (peak 10.4 GB), against about 0.47 hours on the TPU v6e
machine. $0 paid compute.

## Licence note

The run trained on corpus v1. Its code stream was built without a source-licence allowlist; the audit on master
(`experiments/native_public_language/training-license-check.json`) found GPL and AGPL sources in it. The score is a
measurement; the checkpoint should not be distributed as is.
