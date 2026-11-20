# NanoLM Frontier Architecture & Post-Training Evolution (`<= 4B` Ceiling, `~2B` Target)

> **Program Identity:** Independent general-purpose small foundation language model maximizing intelligence, reasoning, coding, summarization, structured extraction, diagram/chart synthesis, tool use, DAG planning, and robustness per parameter under a strict **4,000,000,000 (`4B`) total parameter limit** (`~2B` preferred target).

---

## 1. Architectural Formulations

Implemented in [architectures.py](../../nanolm_frontier/architectures.py) with exact parameter accounting verified against instantiated PyTorch modules (`abs(analytical - actual) == 0` across all 5 candidates and strict `ParameterBudgetExceededError` above `4,000,000,000`):

### 1.1 Scaling Configurations
| Preset | Family | Total Parameters | Delta / Softmax Layers | PKM Slots / Layer | KV Cache / Token (BF16) | Active FLOPs / Token @ 2K |
|---|---|---:|---:|---:|---:|---:|
| `C0..C4` Micro Tournament | All 5 | `38.01M – 38.27M` | `0/12` (`C0`) or `9/3` (`C1..C4`) | `4,096` (`C3, C4`) | `6,144 B` (C0), `1,536 B` (C1–C3), `2,560 B` (C4 at two loops) | `89.0M – 143.8M` |
| **`FRONTIER_2B_CONFIG`** | `C4_JCLE_Frontier` | **`1,915,969,316`** | `24 / 8` | `65,536` (`C=256`) | `26,624 B` (two loops) | `5.63 BFLOPs` |
| **`FRONTIER_4B_CONFIG`** | `C4_JCLE_Frontier` | **`3,865,666,516`** | `36 / 12` | `65,536` (`C=256`) | `40,960 B` (two loops) | `11.86 BFLOPs` |

### 1.2 Competing Innovations Investigated
1. **`C1_GatedDeltaHybrid` (3:1 Chunkwise Gated Delta-Rule Control):** Interleaves 3 chunkwise WY Householder Gated Delta-Rule layers ($S_t = a_t S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$, exact match to token-by-token reference recurrence within `2.38e-6`) with 1 global GQA layer (`-75%` KV cache).
2. **`C2_DiffNSDelta` (Diagonal Gauss-Newton Key Whitening + `DiffGQA`):** Maintains a causal chunked EMA key-energy trace $c_t = \gamma c_{t-1} + (1 - \gamma) k_t^2$ and normalizes whitened keys $\tilde{k}_t = \text{norm}(k_t / \sqrt{c_t + \epsilon})$, paired with dual-call FlashAttention Differential Softmax $(A_1 - \lambda A_2) V$ in global layers.
3. **`C3_CorticalPKM` (Sparse Cartesian Product-Key Cortical Memory):** Replaces 25% of dense SwiGLU FFN width with $M = C^2$ value slots addressed via two $(d_{\text{key}}/2) \times C$ sub-codebooks and `top_k = 8` sparse embedding reads in $O(\sqrt{M} d)$ active FLOPs.
4. **`C4_JCLE_Frontier` (Thalamocortical Prelude-Anchored Contractive Equilibrium Loop):** Weight-shared Core iterates $r \in \{1, \dots, R_{\max}\}$ steps anchored to Prelude state $h_{\text{pre}}$:
   $$h^{(r)} = (1 - \alpha_r) h^{(r-1)} + \alpha_r \, \mathcal{F}_{\text{core}}\big(\text{RMSNorm}(h^{(r-1)} + W_{\text{inj}} h_{\text{pre}} + e_r)\big), \quad \alpha_r = \sigma(\eta_r) \in (0, 1)$$
   with token-level **Shannon Entropy Early Exit** ($H(p_t^{(r)}) < \tau_{\text{ent}}$) at test time.
5. **`DexDiffAttentionWrapper + ThalamocorticalVAnchor + Verifier-Guided GRPO` ([surgery_and_rlvr.py](../../nanolm_frontier/surgery_and_rlvr.py)):** Zero-initialized cross-head differential gating ($\gamma = 0$) on `o_proj` plus Layer-0 Value Residual Anchor on upper layers guarantees **exact `0.00e+00` logit delta at step 0** while enabling on-policy Verifier-Guided GRPO with zero-VRAM frozen-base KL computation via `_bypass_adapters`.

---

## 2. Empirical Results Summary (RunPod `NVIDIA A40` BF16, `\$0.3699` Attributed USD)

See full tables, per-task deltas, bootstrap 95% CIs, failure taxonomies, and Retain / Modify / Reject verdicts in [REPORT.md](../../artifacts/nanolm_frontier_v1/REPORT.md).

- **Experiment E1 (`2.87M` tokens, 5 iso-parameter architectures):**
  - `C0_DenseGQA` achieved fastest training throughput (`63,110 tok/s`, `2,491 MB` VRAM) and lowest overall BPB (`0.6316`).
  - `C3_CorticalPKM` used an estimated `29.5%` fewer active FLOPs per token at 2K (`89.0M` vs `126.4M`, recomputed with the differential value width), lowered overall BPB relative to `C1/C2` (`0.6326` vs `0.6331 / 0.6335`), and outperformed `C0/C1/C2` on `math` (`0.8001`).
  - `C4_JCLE_Frontier` achieved the lowest `math` BPB overall (`0.7884` at `r=4` with entropy halting), with **60.5% token early-exit rate** (`mean_loops_used = 2.184` vs `4.000` unhalted), while showing pure-PyTorch recurrent throughput penalties (`5,181 tok/s`) and `structured` BPB regression (`0.4288`) — identifying fused CUDA delta kernels and task-conditioned looping as the priority engineering targets.
- **Experiment E2 (`Qwen/Qwen3-1.7B`, `1.72B–1.75B <= 4B`, 15-Dimension Suite, `0` contamination overlaps):**
  - `Base`: `34/56` (`60.71%`, `CI95 [48.2%, 71.4%]`), `33.53 tok/s`.
  - `LoRA_SFT`: `36/56` (`64.29%`, `CI95 [51.8%, 75.0%]`), lifting `math_logic` (`25% -> 100%`), `diagrams_charts` (`75% -> 100%`), `instruction_following` (`75% -> 100%`), and `hallucination_resistance` (`75% -> 100%`), while exposing acute 5-epoch SFT overfitting in `coding` (`75% -> 0%`).
  - `Frontier_DexDiff_SFT_GRPO`: `35/56` (`62.50%`, `CI95 [51.8%, 73.2%]`), verifying exact `0.00e+00` step-0 preservation, achieving the highest **`hard`-difficulty accuracy (`60.0%` vs `55.0%` SFT and `45.0%` Base)**, uniquely solving hard DAG planning (`pln-04`), and revealing why over-trained SFT (`~96.5%` rollout reward) starved GRPO (`71/72` zero-variance groups skipped) — establishing an actionable curriculum fix (early-stop SFT at 1–2 epochs + harder RL prompts).

---

## 3. Reproducibility & Verification Commands

```bash
cd /path/to/repo
PYTHONPATH=. python3 -m pytest -q nanolm_posttrain/ nanolm_frontier/
PYTHONPATH=. python3 -m nanolm_frontier.tournament --mode all --out-dir artifacts/nanolm_frontier_v1
```
