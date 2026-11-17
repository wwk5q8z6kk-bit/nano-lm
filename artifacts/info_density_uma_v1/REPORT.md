# Information Density & Apple Silicon Unified-Memory Benchmark (`v1`)

**Date:** 2026-11-16 · **Schema:** `nano.info_density_uma_benchmark.v1` · **Paid compute:** $0 (local Apple M3 Max 40-core GPU, 64 GB UMA)

This report records the implementation and live Apple Silicon verification of [`nanoscribe/info_density.py`](../../nanoscribe/info_density.py), entropy-aligned shared-prefix KV decoding in [`nanoscribe/compact.py`](../../nanoscribe/compact.py), and [`scripts/benchmark_info_density.py`](../../scripts/benchmark_info_density.py). Raw machine outputs are in [`benchmark_results.json`](benchmark_results.json).

## 1. Epistemic Operator Information Density (`nano-lm`)

Per [EMPIRICAL_FOUNDATION.md](../../papers/EMPIRICAL_FOUNDATION.md) (`Paper α` `D5→D80` and `E1`), open-vocabulary patient values belong in verbatim source pointer spans (`0` weight bits), isolating model parameter capacity to the **epistemic operator** $I(Y_{\text{state}}; \hat{Y} \mid X)$:

| Evaluation Cohort | Model | Weight MiB (FP16) | Fields | State MI (bits/field) | MI Density (bits/field/MiB) | Grounded Exact Fields / MiB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Authenticated H6 (`cb612410…`) | Native Nano H6 (`3.29M`) | `6.27` | `5,000` | `0.8130` | **`0.1297`** | **`622.20`** |
| Pilot `run-09` (exposed grammar) | Qwen2.5-1.5B (`1,543.7M`) | `2,944.40` | `100` | `1.3660` | `0.000464` | `0.0323` |
| Fresh validation `C14` (`4097d42d…`) | Qwen2.5-1.5B (`1,543.7M`) | `2,944.40` | `120` | `0.0771` | `0.000026` | `0.0071` |

- **279.45× higher MI density per parameter byte:** Even against Qwen2.5-1.5B's best exposed-pilot ceiling (`95/100`), Native Nano H6 delivers **279.45×** higher epistemic mutual information per MiB of weights (`0.1297` vs `0.000464` bits/field/MiB) and fits completely inside M3 Max L2 / System Level Cache (`6.27 MiB < 48 MiB`).
- **C14 Fano diagnosis:** On C14 (`120` fresh fields), conditional entropy $H(Y \mid \hat{Y}) = 1.8221\text{ bits}$ (out of $H(Y) = 1.8992\text{ bits}$) induces a **42.03% Fano minimum error floor** (`81.67%` observed error) driven by candidate-menu starvation (`4/120` concrete slots) rather than hardware throughput.

## 2. Entropy-Aligned Prefix-KV Reuse & Unary-Trie Coalescing (`mps` Live Run)

Measured head-to-head across all 20 dialogues (`101` constrained slot calls) of `artifacts/p1_pilot_v1/dialogues.json` using pinned `Qwen2.5-1.5B-Instruct` (`dd924a11b4c2…`) on `mps` (`float16`):

| Decoder Mode | Exact Slot Matches | Forward Passes | Coalesced 0-Entropy Tokens | DRAM Weight Reads | Wall-Clock Time | Checked Grounded Exact |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `typed-context-source-trie-v8` (uncached) | baseline | `450` | `0` | `1,395.95 GB` | `43.55 s` | `0.62` |
| `typed-context-source-trie-v9-prefix-kv` | **`20 / 20` (100%)** | **`125`** | **`244`** | **`387.76 GB`** | **`20.98 s` (`2.08×` faster)** | **`0.62`** |

- Reusing the shared `(system, user)` `DynamicCache` across all 5 encounter slots via `crop()` and coalescing zero-entropy unary trie transitions ($H(T_i \mid T_{<i}) = 0$) eliminates **`1,008.19 GB` (`72.22%`)** of redundant unified-memory weight traffic with **20/20 exact output identity**.

## 3. Commit-Gated KV Eviction & Zero-Copy `.safetensors` Alignment Audit

1. **Zero-erasure two-tier KV eviction (`CommitGatedKVCacheLedger`):** Auditing all 20 encounters shows unbounded FP16 KV retention consumes `60.38 MB`, whereas committing verified span offsets to the symbolic `Encounter` ledger + immutable UTF-8 source requires `11,489 B` (**`99.981%` memory reduction**, `0` uncommitted erasure events). By contrast, a naive 256-token sliding KV window over a 960-token encounter erases `4/5` early evidence spans (`9.29` bits lost; `80.0%` Fano error floor).
2. **Upstream `.safetensors` 8-byte Metal misalignment caught & solved:** Read-only POSIX `mmap` inspection of Qwen2.5-1.5B's `model.safetensors` (`3,087,467,144 B`, `338` tensors) revealed `header_size_bytes = 38,528` ($38528 \equiv 0 \pmod{64}$), placing `data_section_offset = 38,536` ($38536 \equiv 8 \pmod{64}$ and $\not\equiv 0 \pmod{16384}$). Raw upstream `.safetensors` files therefore fail both Metal 64-byte alignment and 16 KiB Apple Silicon VM page alignment for `MTLBuffer(bytesNoCopy:)`. `repack_safetensors_page_aligned()` pads the JSON header with spaces so `8 + header_len` lands cleanly on a `16,384 B` Darwin VM page boundary without altering tensor bytes.
3. **Portable UMA runtime routing (`nano.uma_runtime.v1`):** `recommend_uma_tier()` maps host thermal state (`nominal`/`fair`/`serious`/`critical`), `low_power_mode`, and available UMA headroom to `compact_1_5b_fp16_prefix_kv`, `compact_1_5b_q4_prefix_kv`, `native_nano_3_3m`, or `deterministic_reference` over portable `posix_mmap_readonly` sharing.
