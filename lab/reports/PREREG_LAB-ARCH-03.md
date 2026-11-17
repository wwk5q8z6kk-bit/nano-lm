# PREREG LAB-ARCH-03 & LAB-SCALE-01 — Next-Gen First-Principles Mechanisms & 125M Scaling Check (Track A)

*Registered 2026-11-17 after LAB-ARCH-02 Wave-1/2 multi-seed findings.*

## 1. Motivation from LAB-ARCH-02 Multi-Seed Evidence
In LAB-ARCH-02 (`600M` tokens, `n=2` seeds, seed half-range `0.0000–0.0005` BPB on winning arms):
- `T1_vres` achieved `0.9919 ± 0.0002` BPB (`-3.66%` vs `T0_dense`), winning Code (`0.8245`), **but** collapsed
  on 512-token OOD random-sequence induction (`copy_loss = 9.207` nats vs `3.314` on `T0_dense`).
- `H3_delta_vres` achieved `0.9942 ± 0.0000` BPB (`-3.44%` vs `T0_dense`) with **`-80%` global KV cache** and
  solved induction (`copy_loss = 0.879` nats, `93.8%` acc), despite only sharing values between the 2 Attention layers.
- `R2_delta_loop` achieved `0.9953` BPB (`#1` Math `1.1510`, `#1` Copy `0.754` nats), showing recurrent depth
  synergizes with stateful Gated Delta-Rule layers where pure RoPE attention looping (`R1_loop`, `1.0415`) failed.

## 2. Registered Arms (`lab/arch/train_nextgen.py`)
All `46M` arms match `T0_dense` non-embedding parameters (`29.5M ± 1%`) and run `600M` sequential tokens on
identical `LAB-DATA-01` data:
1. **`X1_gated_vres`**: `DDDADDDADA` + universal per-head input-dependent sigmoid Layer-0 Value Highway across
   **all 10** Delta + Attention layers. Hypothesis: beats **both** `T1_vres` (`0.9919` BPB) and `H3_delta_vres`
   (`0.9942` BPB) while retaining `>90%` OOD copy accuracy and `-80%` KV cache.
2. **`X2_sharp_delta_vres`**: `X1` + QK-decoupled `ShortConv` (`sharp_v=True`, unblurred value writes) + QK-Canon
   on Attention. Hypothesis: sharpens associative recall and code/math BPB further over `X1`.
3. **`X3_sgrd_delta`**: `DDDADDDADA` `loop=(2,6,2,2)` + **Surprise-Gated Recurrent Depth (`SGRD`)** coupling
   halting logits to log online delta prediction error `gamma * log ||e_t||`. Falsification: reject if `X3` fails
   to beat `H2_delta` or if `halt_gamma -> 0`.
4. **`X4_nanolm_v2_muon`**: `X2_sharp_delta_vres` + orthogonalized `Muon` optimizer. Tests constructive stacking
   of architecture + optimizer.
5. **`LAB-SCALE-01` (`S0_dense_125m` vs `S1_nanolm_v2_125m`)**: Scales `d=512, L=10` (`29.5M` non-emb) by `3.26×`
   to `d=768, L=16` (`96.3M` non-emb / `121M` total, `600M` tokens) to test whether the Next-Gen hybrid advantage
   persists or widens with scale!

## 3. Compute & Budget Cap
- Reuses freed RTX 4090 grid pods (`$0.89/h`, zero re-upload cost because `/workspace/data/train.bin` is already
  verified on disk!).
- 4 arms (`46M`, ~45 min each) + 2 scale-up arms (`125M`, ~95 min each) across freed 4090s = **~$6.50 total**.
- Hard cumulative campaign watchdog remains active (`$45` cap, well inside `$180` autonomous budget).
