# NanoLM Autonomous Architectural Discovery Research Log

## Phase A & B: Repository Inspection & Bottleneck Identification
- **Audited Existing Branches & Baselines (`2026-11-17`):**
  - Historical `pretrain/train.py`: canonical 3.15M SwiGLU + GQA Transformer (`6` layers, `d=192`, `V=4096`).
  - `nanolm_frontier_v1` (`813cfab`, RunPod A40 BF16): benchmarked 5 hybrid `38M` architectures (`C0_DenseGQA` `0.6316 BPB`, `C3_CorticalPKM` `0.6326 BPB` at `-34.5%` active FLOPs, `C4_JCLE` best `math BPB = 0.7884` with `60.5%` early exit) and exposed two acute post-training bottlenecks on `Qwen/Qwen3-1.7B`:
    1. 5-epoch SFT on 35 concise examples overfitted (`0.032` loss), collapsing held-out Python coding from `3/4` (`75%`) to `0/4` (`0%`).
    2. Over-converged SFT raised curriculum rollouts to `~96.5%` accuracy, starving Verifier GRPO of variance (`71/72` prompt groups skipped, only `1` update taken).
- **Fundamental Architectural Bottlenecks Identified:**
  1. Transformer $O(T)$ KV-cache growth reaches `1,536 MB` at `16K` context for a `3.84B` model.
  2. Commutative linear recurrences (`U(1)` rotary/diagonal SSMs) mathematically lose order sensitivity over unbroken intervals ($e^{i\theta_2}e^{i\theta_1} = e^{i\theta_1}e^{i\theta_2}$).
  3. Feedforward `SwiGLU` computes single-pass static projections regardless of token prediction surprise.

## Phase C & D: Hypothesis Formulation & Implementation (`nanolm/holonomic-discovery-v1`)
- Implemented [spinor_algebra.py](nanolm_holonomy/spinor_algebra.py), [architecture.py](nanolm_holonomy/architecture.py), [registry.py](nanolm_holonomy/registry.py), [sidecar_evolution.py](nanolm_holonomy/sidecar_evolution.py), and [experiments.py](nanolm_holonomy/experiments.py).
- Verified `17/17` local unit/algebraic/O(1)-decode/budget tests before GPU execution (`5/5` CUDA-selected tests also verified inside RunPod container before X1–X4).

## Phase E & F: Live RunPod `NVIDIA L40S` GPU Discovery Cycle (`2v2dt0j536rh58`, `$1.3026 USD`)
1. **X1 (`3` seeds `[101, 202, 303]`, `2.46M`):** `400` training steps proved insufficient for any architecture (`B0`, `B1`, `H1`, `H2`, `H3`) to solve 24-class `S_4` permutations (`2.1%–7.3%`). On 4x OOD `SET/OVERRIDE` recall (`ev=40`), constant-state recurrences (`B1` `24.5%`, `Gen1_H1` `21.4%`) retained roughly 2x higher accuracy than `B0_Transformer` (`13.0%`).
2. **X2 (`2` seeds `[42, 137]`, `14.0M`, `6.55M` tokens/arm):**
   - `Gen1_H1_HSPA_Spinor` (`0.3800`) beat `B1_Abelian_SSM` (`0.3826`) -> validates non-abelian $SU(2)$ > commutative $U(1)$.
   - **`Gen1_H2_HSPA_PredCode` (`0.3723 ± 0.00235`) beat `B0_Transformer` (`0.3794 ± 0.00090`) on BOTH seeds**, lifting `state` BPB (`0.4745 -> 0.4438`) and exact `state` probe (`1.04% -> 13.54%`), earning `BEST` promotion in `ExperimentRegistry`.
   - `Gen2_H3` (Wavepacket Conv + Kuramoto *without* PredCode) dropped to `0.3803` -> **REJECTED**, isolating `PredictiveCodingCell` as the true causal mechanism of Gen1_H2's gain.
3. **X3 (`3.845B` BF16 scaling on 48GB L40S):** `nanolm-hspa-4b` kept VRAM flat at `8.31–8.32 GB` (`1.781 MB` state) through `T=16,384` vs `nanolm-transformer-4b`'s rising `7.62 -> 13.82 GB` (`1,536 MB` KV), while exposing `3.2x` slower pure-PyTorch decode latency (`135.0` vs `41.8 ms/tok`).
4. **X4 (`Qwen/Qwen3-1.7B` + `HolonomicRegisterSidecar` + Early-Stop SFT + Challenging RLVR):**
   - Exact step-0 identity (`0.00e+00`).
   - Stopping SFT at 2 epochs (`loss 0.309`) restored Python coding (`0/4` LoRA_SFT -> `2/4`), unlocked `18` active GRPO updates (raising training rollout reward `0.227 -> 0.364 -> 0.428`), retained `planning_decomposition = 1/4` (vs `0/4` Base/LoRA_SFT), lifted `math_logic = 4/4`, `structured_json = 3/4`, and `hard = 55%` (`35/56 = 62.50%`), while losing `1` task on diagrams/abstention from higher KL (`0.742`).

## Phase G: Next Hypothesis Queue
1. Combine `PredictiveCodingCell` (winner from `Gen1_H2`) WITH `Gen2` Wavepacket Conv in a single fused Triton scan kernel so we test `SU(2) + Wavepacket + PredCode` cleanly AND eliminate the `8.2x` Python kernel launch bottleneck.
2. Re-run `X4` with tighter GRPO `kl_beta` (`0.08` vs `0.03`) to capture `2/4 -> 3/4` coding + `1/4` planning without regressing Mermaid/abstention formatting.
