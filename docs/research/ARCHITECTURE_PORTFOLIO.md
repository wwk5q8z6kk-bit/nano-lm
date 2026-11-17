# Architecture Portfolio & Strict Parameter Accounting

**Constraint:** Preferred ~2B total deployed learned parameters; absolute hard cap strictly `< 4,000,000,000` parameters across backbone, input/output embeddings, routers, shared/routed experts, fast weights, adapters, and any auxiliary learned modules retained at inference.

---

## 1. Competing Architecture Portfolio (`A`–`G`)

| ID | Candidate | Core Design | Rationale | Key Weakness | Next Falsifying Test |
|---|---|---|---|---|---|
| **A** | **Dense Transformer + GQA/MLA Control** | Full causal GQA (`QK-Norm`, SwiGLU) or MLA, matched `AdamW` & `Muon` | Hard-to-beat quality/latency baseline; `48.36 tok/s` at `ctx=512` on M3 Max | $O(T)$ KV cache growth (`2.25 GB` @ 32k for `nanolm-2b-dense`); memory-bandwidth collapse at long context | Matched `125M`–`350M` iso-FLOP + multi-needle control (`X05`, `X10`, `X11`) |
| **B** | **Modern 2026 Hybrid SSM / DeltaNet** | 3:1 recurrent mixers (`GDN-2`, `KDA`, `Mamba-3`) per sparse global attention anchor | Contemporary 2026 expressivity and independent erase/write (`GDN-2`) or complex $SO(2)$ dynamics (`Mamba-3`) | Fixed-state capacity ceiling for verbatim recall; kernel/MPS complexity | Interleaved `GDN-2` vs `Mamba-3` vs `Householder` parity & state-tracking (`X03`–`X05`) |
| **C** | **Nano Geometric-Memory Hybrid** | `B` + decoupled erase/reflection/write $M_{t+1} = \rho_t(I-\beta_t u_t u_t^\top)M_t + \eta_t k_t(v_t - M_t^\top k_t)^\top$ + `ExactEvidenceStore` | Separates reversible/orthogonal fast state from lossless verbatim evidence (`MM01`) | Spectral bound $\|A_t\|_2 \le 1$ does not bound nonlinear write Jacobian; extra gating overhead | Full Jacobian & OOD retrieval vs `GDN-2` / `Mamba-3` (`X03`, `X04`, `X07`) |
| **D** | **Recursive / Adaptive-Compute Foundation** | Shared recurrent core or entropy-routed refinement (`ATRD`), bounded halting, task verifier | Increases effective depth per parameter on hard reasoning | Prior failure: `X3_sgrd_delta` collapsed `expected_loops -> 1.00`; `R1_loop` lost `+8.26%` BPB at matched FLOPs | Shared-budget VoI controller vs uniform best-of-N (`X08`; `X5_atrd_muon` in `nanolm-lab`) |
| **E** | **Sub-4B Sparse MoE + Conditional Lookup** | `1 shared + 16 routed` Aux-Loss-Free MoE (`<4B` total) + Engram-style hash lookup | Lower active FLOPs (`1.14B` active on `nanolm-v3-3.9b-moe`) + static-knowledge lookup | Eager MPS decode bottleneck (`4.90 tok/s` @ `ctx=4096`); small-model routing overhead | Admitted only if fused kernels + active-FLOP/total-byte Pareto beats dense (`AX14`, `MM05`) |
| **F** | **Masked Diffusion / Latent Refinement** | Non-autoregressive proposal + constraint-aware iterative editing (`LLaDA`-class) | Bidirectional span/JSON repair | Slower iterative decoding; unverified 2B foundation trade-offs | Equal-wall-clock JSON/span editing vs AR (`AX15`, challenger track only) |
| **G** | **Structured World-State System** | Compact `<4B` foundation core + explicit event graph (`nano/`), immutable evidence ledger, planner, deterministic verifiers, sandboxed tools | Provable source conservation, temporality, supersession, and calibration (`Track 2`) | System metrics cannot substitute for naked foundation capability | Separate `Track 1` vs `Track 2` scorecards on `P1` and non-medical incident/doc QA (`X15`, `X16`) |

**Active starting portfolio:** `A`, `B`, `C`, `G` (plus isolated spike for `D`). Challengers `E` and `F` are admitted to scale training only after winning controlled low-cost screens.

---

## 2. Strict `< 4B` Parameter Census (Internal & External References)

| Model / Preset | Origin / Status | Total Deployed Params | Active Params / Tok | Eligible (`<4B`)? | Architecture Notes & Caveats |
|---|---|---|---|---|---|
| **Native Nano H6** | In-tree (`nano_ai/`, **VERIFIED / REJECTED H6**) | `3,290,000` | `3,290,000` | Yes (mechanism rung) | `6.27 MiB` FP16; rejected on H6 abstention gates (`3901/5000`) |
| `nanolm-v3-35m` | `LOCAL_WORKTREE_UNMERGED` (`nanolm/arch-evolution-v3@86371cb`) | `34,693,858` | `34,693,858` | Yes (research rung) | `12L` (`9D + 3A`), `1,536 B/tok` KV (`bf16`); architecture spec only |
| `nanolm-2b-dense` | `LOCAL_WORKTREE_UNMERGED` (`nanolm/arch-evolution-v3@86371cb`) | `2,040,687,616` | `2,040,687,616` | Yes | `36L` Dense GQA, `73,728 B/tok` KV (`2.25 GB` @ 32k); **untrained profile only** |
| `nanolm-v3-2b` | `LOCAL_WORKTREE_UNMERGED` (`nanolm/arch-evolution-v3@86371cb`) | `2,072,984,566` | `2,072,984,566` | Yes (preferred target) | `36L` (`27D + 9A`), `18,432 B/tok` KV (`0.5899 GB` @ 32k); **untrained profile only** |
| `nanolm-v3-2b-sgrd` | `LOCAL_WORKTREE_UNMERGED` (`nanolm/arch-evolution-v3@86371cb`) | `2,081,375,224` | up to `2,727,642,448` | Yes | Adaptive recurrent middle stack (`48` eff layers); **untrained profile only** |
| `nanolm-v3-3.9b-moe` | `LOCAL_WORKTREE_UNMERGED` (`nanolm/arch-evolution-v3@86371cb`) | `3,928,815,192` | `1,143,273,048` | Yes (`98.2%` of cap) | `22L` (`17D + 5A`), `1 shared + 16 routed`; **untrained profile only** |
| `nanolm-2b` (`posttrain-v1`) | `CROSS_BRANCH` (`origin/nanolm/posttrain-v1@ee3df6b`) | `1,978,345,602` | `~2.6B+` loop equiv | Yes | `36` physical / `54` looped GQA layers; architecture spec + 1.54B LoRA pilot |
| `nanolm-4b` (`posttrain-v1`) | `CROSS_BRANCH` (`origin/nanolm/posttrain-v1@ee3df6b`) | `3,864,791,042` | looped | Yes | `48` physical / `72` looped GQA layers; architecture spec only |
| `Qwen2.5-1.5B-Instruct` | External reference (`Apache-2.0`) | `1,543,714,816` | `1,543,714,816` | Yes | Dense GQA control used in P1 pilots and Track B baseline |
| `Qwen3-1.7B` | External reference (`Apache-2.0`) | `1,720,000,000` (tied) / `2.03B` (untied) | `1.72B` | Yes | Track B post-training base (`74.98%` GSM8K base) |
| `Qwen3.5-2B` | External reference (`Apache-2.0`) | `2,270,000,000` (reported incl. VLM) | `1,880,000,000` | Yes (verify VLM drop) | `24L` `3:1` Gated-DeltaNet + gated attention; critical 2026 comparator for Nano-v3 |
| `MiniCPM5-2B` | External reference (`openbmb`) | `2,516,756,480` | `2,516,756,480` | Yes | Actual published total `~2.52B` including embeddings (not `2.0B`) |
| `SmolLM3-3B` | External reference (`Apache-2.0`) | `3,080,000,000` | `3,080,000,000` | Yes | Dense GQA + NoPE/RoPE reference |
| `Granite-4.0-H-Micro` / `4.2 3B` | External reference (`Apache-2.0`) | `~3,190,000,000` | `~3,190,000,000` | Yes (verify 4.2 exact) | Mamba-2 + GQA hybrid baseline |
| `Phi-4-mini-instruct` | External reference (`MIT`) | `3,840,000,000` | `3,840,000,000` | Yes | Dense compact ceiling near 4B boundary |
| **`Gemma-4-E2B`** | External outside-cap reference | `~5,100,000,000` (incl. embeddings) | `~2B+` active | **NO — OUT OF CAP** | Disqualified from sub-4B portfolio; outside-cap reference only |
| **`Qwen3.5-4B+` / `Qwen3-8B` / `32B` / `gpt-oss-120b`** | External teacher / ceiling | `> 4,000,000,000` | varies | **NO — OUT OF CAP** | Teacher/ceiling reference only; never counted as deployable Nano candidate |

*Accounting rule:* Never compare a post-trained external checkpoint (`Qwen3.5-2B`, `Phi-4-mini`) to an untrained or 600M-token prototype and attribute capability gaps to architecture alone. Maintain separate Base/Pretrain, Matched-Adaptation, and System leaderboards.
