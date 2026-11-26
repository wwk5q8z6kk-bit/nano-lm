# NANOLM Evolution Roadmap (Cycles 4–6)

Grounded strictly in the empirical findings of Cycles 1–3 and Kaggle CUDA `EXP-INV-06`, this roadmap defines the exact engineering and scientific gates required to move `G3_HMWR` from validated prototype to a frontier `< 4B` trained foundation architecture.

---

## 1. Cycle 4A: Systems Acceleration & Memory Optimization (Prerequisite)
- **Objective:** Close the un-fused runtime gap (`G3_HMWR` `2.13x` training overhead vs `B0`, `7.88 GiB` unchunked `32k` prefill peak on T4, and `406–452 ms/tok` `3.88B` M3 Max decode).
- **Concrete Engineering Targets:**
  1. **Streaming Chunked Prefill (`C_stream = 1024`):** Utilize `G3_HMWR`'s strictly $O(1)$ persistent state (`83.71 MiB` at `3.88B`, `13.22 MiB` at `222M`) to evaluate prompts in bounded `1,024`-token slices during inference prefill — immediately cutting `32k` peak T4 VRAM from `7,876.7 MiB` down to `~1.1 GiB` (`< B0`'s `3,646.6 MiB`).
  2. **Fused WY + SCAM Kernels (Triton CUDA & Metal/MPS):** Replace the Python `for c in range(n_chunks)` chunk-prefix loop and `forward_step` per-layer Python dispatch with fused register-resident Triton (for CUDA) and Metal shaders.
  3. **Acceptance Gate:** `>= 22,000 tok/s` training on `20.55M` Tesla T4 (`>= 80%` of `B0`) and `<= 50 ms/tok` (`>= 20 tok/s`) live `3.88B` decode across `512–32k` context on M3 Max, preserving `< 1e-4` exact step duality.

---

## 2. Cycle 4B: Extended-Horizon & Pure-CE `Muon` Crossover Law
- **Hypothesis to Falsify:** At `4.1M` tokens (`1,000` steps), `G3_HMWR` closed to `-0.0154 BPB` of `B0` in-distribution while leading `+0.0204 BPB` at `2x` OOD, and pure-CE `Muon` fell `-0.3593` BPB (`250->1000`) vs `-0.3088` for `AdamW`. Does `G3_HMWR` cross `B0` in-distribution between `15M` and `50M` tokens (`~1–2x` Chinchilla on `20.55M`), and does pure-CE `Muon` beat `AdamW` past `3,000` steps?
- **Experimental Design (`Kaggle` free `30h/wk` CUDA quota or single cloud run within budget):**
  - Train `b0_transformer`, `g3_hmwr (AdamW)`, `g3_hmwr (Muon CE)`, and a `G3_HMWR` variant replacing SCAM softmax with causal linear associative eviction (`SCAM-64`) across **`10,000` steps (`40.96M` tokens)** at `20.55M` and `47.43M`, logging validation BPB across `256 -> 512 -> 1024 -> 2048 -> 4096` context extrapolation.
  - **Kill / Promote Gate:** Promote to flagship `1B–3.88B` only if `G3_HMWR` matches or beats `B0_Transformer` on **both** `1x` ID BPB and `>= 2x` OOD context at Chinchilla scale (`20 tokens/param`).

---

## 3. Cycle 5: P1 NanoScribe Epistemic Grounding Integration
- **Directly Addresses Integrated `H6` Failure Modes (`artifacts/nano_h6/H6_TERMINAL_DECISION.json`):**
  - Integrated `NanoEvidenceQueryPointerModel` passed direct value copying (`76.23%`) but failed `absence` (`48.18%`), `conflict` (`57.20%`), and `uncertainty` (`76.00%`), while `10M` `Stage S` retained a `23-pt` OOD held-out gap.
  - **`G3_HMWR` Mapping to Clinical Scribing:**
    1. Wire `HMWROperator`'s non-inverted causal epistemic uncertainty $u_t^{(G_3)}$ (alongside logit entropy, `AUROC = 0.875`) directly into NanoScribe's 5-state classifier (`absent` / `uncertain` / `conflicting`).
    2. Use `SCAM-32`'s persistent content-addressable high-surprise slots to retain low-frequency patient medications, lab values, and allergies verbatim across multi-turn transcripts without $O(T)$ KV eviction.
  - **Acceptance Gate:** Strictly beat the `H6` baseline across `absence`, `conflict`, and held-out `OOD` slot transfer under the synthetic NanoScribe encounter protocol (no PHI / no clinical deployment claims until human dual-IAA verified).

---

## 4. Cycle 6: Flagship `3.88B` (`< 4.000B`) Foundation Run
- **Target Specification:** `g3_hmwr_4b` (`3,882,612,303` deployed parameters, `d=2816, L=32, ffn=7552, V=65,536`, constant `83.71 MiB` persistent inference state across `32k–128k` context — `67.3x` smaller than `b0_transformer_4b`'s `5,632 MiB` @ `32k`).
- **Execution Policy:** Requires explicit Cycle 4B Chinchilla-crossover pass before authorizing cloud GPU compute (`$180` autonomous budget envelope / `$200` hard cap per `docs/infrastructure/RUNPOD.md`), using free Kaggle kernels strictly for fused-kernel CI, ablations, and `<= 50M` curriculum validation.
