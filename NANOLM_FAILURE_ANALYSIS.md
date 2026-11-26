# NANOLM Ω — Top 10 Technical Limitations & Falsified Hypotheses (`TASK D`)

This failure analysis catalogs the ten highest-impact architectural, mathematical, training, and systems bottlenecks uncovered across Phases 1–4, Frontier E1–E2, and HSPA Cycles 1–2—together with all rigorously falsified approaches.

---

## 1. Ten Largest Technical Limitations Uncovered in Audit

1. **Cayley Pole Singularities & Double-Cover Spinor Sign Cancellation (`C2-X1`):**
   *Observed Failure:* In `C2-X1` (`artifacts/nanolm_holonomy_v2/c2_x1_s4_grokking.json`), even unbroken $S_4$ chains achieved only `2.8%–5.6%` (`1/24` chance) across all four full-LM arms. First-principles diagnosis revealed two distinct mathematical causes: (i) the Cayley map $(1+i\omega)/(1-i\omega)$ maps pure $180^\circ$ permutations (like transpositions in $S_4$) to $|\omega| \to \infty$, distorting gradient norm around $w=0$; (ii) direct linear readout of spinors $q \in SU(2)$ conflates $+q$ and $-q$, both of which project to the **same** permutation in $SO(3) \cong SU(2)/\mathbb{Z}_2$!
2. **LoRA Plasticity-Stability Clash in Code Synthesis (`C2-X4` vs `E2`):**
   *Observed Failure:* Unconstrained standard LoRA + 3-epoch SFT improved `tool_use` (`2/4 -> 4/4`) and `summarization` (`3/4 -> 4/4`) on `Qwen3-1.7B` (`35/56`), yet degraded held-out Python `coding` from `3/4 (75%)` (Base) to `0/4` (`E2` 5-ep SFT) and `1/4` (`C2-X4` 3-ep SFT) via `NameError` (`heapq` omitted) and slice `IndexError`. Gradient updates along top right singular vectors $V_k$ of frozen base weights $W_0$ overwrite high-energy pretrained coding circuits.
3. **Abelian / Contraction State-Tracking Ceilings (`B1`, `Mamba/GLA`, $\beta \in (0, 1)$):**
   *Mathematical Proof & Evidence:* Any homomorphism $\phi: S_4 \to U(1)^k$ factors through $S_4 / [S_4, S_4] = S_4 / A_4 \cong \mathbb{Z}_2$ (parity sign only), collapsing accuracy to $\approx 4\%$ (`~1/24`) for even/odd subgroups. Standard transformers additionally fall inside $TC^0$ (`B0` `3.5%–5.2%` on unbroken $S_4$).
4. **Syntax-Only Test-Time Compute Blindness (`C2-X4` `SyntaxTTC` `0` Triggers):**
   *Observed Failure:* In `C2-X4`, `NanoLM_1.7B_Holonomic_v2_SyntaxTTC` triggered **`0` times** (`recovered=0`) across all `56` tasks despite `21` failures. Why? Small-model failures were *runtime semantic bugs* (`NameError`, `AssertionError`, `IndexError`), *arithmetic/combinatorial planning errors* (`wrong_makespan`, `subtotal:400!=520`), and *missing Mermaid graph edges* (`edges=1`)—all of which pass syntactic `ast.parse` and `json.loads`!
5. **Eager-PyTorch Scan & Layer-Loop Overhead (`X2`, `X3`, `E1`):**
   *Observed Failure:* Pure-PyTorch Blelloch scan in Cycle 1 ran `8.2x` slower in training (`13.2k` vs `107.4k tok/s`) and `3.2x` slower in `3.845B` decode (`135.0` vs `41.8 ms/tok`). Even after Cycle-2's `35x` Triton scan kernel speedup (`C2-X0`), `C2-X3` `nanolm-hspa-4b` decode remained `~74 ms/tok` at short contexts because Python per-layer `hamilton_product`, `cayley_rotor`, and norm launches dominate `batch=1` latency until CUDA Graphs / fused projection-scan kernels are used.
6. **Sparse-Reward GRPO Variance Collapse (`E2` `71/72` Skipped, `C2-X4` `75/94` Skipped):**
   *Observed Failure:* Running GRPO *after* overfitting SFT drives training-set rewards to `0.95–1.00`, causing group standard deviation $\sigma \approx 0$ (`71/72` skipped in `E2`, `75/94` skipped in `C2-X4`). Furthermore, overly low `kl_beta` degrades syntax without improving combinatorial planning (`planning_decomposition = 0/4` in `C2-X4`).
7. **Full-From-Scratch 2B Pretraining Compute Ceiling vs Budget:**
   *Economic Constraint:* Chinchilla-optimal from-scratch pretraining of a `2B` parameter model requires `~40B` tokens ($\approx 4.8 \times 10^{20}$ FLOPs, or `~480` H100 GPU-hours $\approx \$1,400+$), exceeding an autonomous `~$100–$180` campaign budget. Hence genuine foundational `<4B` research must combine (a) small-to-medium (`14M–125M`) from-scratch architectural discovery + `3.85B` hardware scaling verification with (b) spectral/holonomic surgical adaptation of open weights (`Qwen3-1.7B`, `SmolLM3-3B`).
8. **Absent Pre-Flight Hardware & VRAM Calibration Before Pod Launch:**
   *Operational Defect:* Prior campaigns experienced multiple wasted RunPod launches from CUDA driver mismatch, missing `--ports 22/tcp`, uncalibrated batch/sequence VRAM OOMs, and naive long-context activation over-estimates (`6.58 GB` logits materialization at `32,768` seq_len before `logits_to_keep=1` fix in Cycle 2).
9. **Multi-Turn Budget Starvation in Think-Mode Orchestration (`System-v2`):**
   *Observed Failure:* Escalating `reasoning_effort` blindly during recovery exhausted `max_tokens` inside `<|channel|>analysis` before emitting structured final answers or tool calls (`gpt-oss-120b` tools dropped `75% -> 15%` in `System-v2`).
10. **Decoupled Epistemic Abstention & Value Copying (`H6` + `T1_vres`):**
    *Observed Failure:* Naive residual skip connections (`T1_vres` `9.207` copy loss) blur exact retrieval, while small boundary adapters (`H6` 640-param gate) trade off epistemic states (`absence` `-19.6 pp`) unless precision-weighted conflict/uncertainty supervision is explicitly modeled.

---

## 2. Preserved Negative & Falsified Experiments (`No Cherry-Picking`)

| Candidate / Mechanism | Source Experiment | Observed Result | Verdict & Mechanistic Cause |
|---|---|---|---|
| `R1_loop` (iso-FLOP dense loop) | `nanolm-lab` (`LAB-ARCH-02`) | `+8.26%` worse BPB (`1.1147` vs `1.0296`) | Halving unique token coverage for uniform depth wastes compute without early exit. |
| `T1_vres` (ungated Layer-0 VRes) | `nanolm-lab` (`LAB-ARCH-02`) | `9.207` copy loss (`8.3%` copy) | Unconditional value addition corrupts attention key-value specificity. |
| `P1_gla` / `P2_delta` (0 global anchors) | `nanolm-lab` (`LAB-ARCH-02`) | `0%` exact copy (`11.65–12.03` loss) | Fixed-size linear state alone saturates on arbitrary random-access token lookup. |
| `H6` clinical boundary residual | `H6_TERMINAL_DECISION.json` | `absence 48.2%` (`-19.6 pp` vs H5) | Scalar boundary modulation shifted thresholds without modeling epistemic uncertainty. |
| `ERN-Gen2` (`nanolm-paradigm`) | `RESEARCH_LOG.md` | Distant recall `56.25% -> 21.09%` | Diagonal gating + unguarded `+r_ctx` skip diluted associative recall. |
| `Gen2_H3_HSPA_Evolved` | Cycle-1 `X2` (`L40S`) | `0.3803` BPB (worse than `0.3723` H2) | Arbitrary cross-register Kuramoto phase coupling blurred register specialization. |
| `Gen3_H4_HSPA_WavePred` | Cycle-2 `C2-X2` (`L40S`) | `0.3696` vs `0.3693` `Gen1_H2` (tie/loss on BPB) | Did **not** beat `Gen1_H2` on overall BPB (`reg=did_not_beat_best`), though lifted `state` probe (`15.28%` vs `13.20%`) — preserved honestly in `ExperimentRegistry`. |
| `Gen4_Omega_SO3_PredCode` (`14M`) | Cycle-3 `EXP-Ω5` (Kaggle T4) | `0.3775±0.0044` vs `0.3701±0.0047` `Gen1_H2` (`state` `3.47%` vs `13.19%`) | Beat Transformer (`0.3824`), saved `-14.3%` VRAM (`3,263.9` vs `3,810.0 MB`), and topped `math` (`9.03%` vs `8.33%`), yet did **not** beat `Gen1_H2` on 800-step BPB because 7/8 deeper layers initialized at `res_mix_gate=0` (`reg=did_not_beat_best`). |
| Unconstrained 3–5 ep LoRA SFT | `E2` & `C2-X4` | Python `coding` `3/4 -> 0/4..1/4` | Gradient projection onto top-`k` singular subspace of $W_0$ overwrote base coding circuits. |
| Uncheckpointed `1.7B` SFT on `T4` (`14.56 GiB`) | Cycle-3 `Ω-v2` kernel v1 | CUDA OOM entering `EXP-Ω7` (`5211.42s`) | Resolved in kernel v2 (`1450.81s`, `$0.00`) via `use_reentrant=False` gradient checkpointing + `use_cache=False` during SFT/GRPO backward (`Merged_Greedy` `38/56` T4 FP16 vs `40/56` A40 BF16). |
| `dynamic_rewire_ph` & `sheaf_port_hamiltonian` vs `unsigned_gat` (`6x` OOD) | Cycle-4 `EXP-FP1` (Kaggle T4) | `unsigned_gat` `83.81%` > `sheaf_ph` `58.09%` > `dynamic_rewire_ph` `43.81%` | Simpler degree-normalized softmax in `unsigned_gat` generalized better to `N=160` degree shifts than `sheaf_ph` (though `sheaf_ph` beat `flat_transformer` `47.62%` & `signed_mesh` `38.09%` with `0.0` Lyapunov violation); dynamic rewiring hurt OOD. |
| `Gen5_FP_PH_Holonomy` on unbroken $S_4$ & macro BPB | Cycle-4 `EXP-FP2A/B/C` (Kaggle T4) | $S_4$ `L256`: `Gen5` `3.12%` vs `Gen4` `97.66%`; 14M BPB: `Gen5` `0.3756` (AdamW) / `0.3793` (Muon) vs `Gen1_H2` `0.3751` / `0.3749` | Active deeper episodic mixing interfered with layer-0 $S_4$ transport, and `Gen1_H2_PredCode` retained `BEST` macro BPB; `Gen5` was **not** promoted on BPB despite its `201x` bitemporal copy-gate breakthrough (`96.67%` vs `0.48%`). |
| `FP16` `-1e9` mask overflow in `CausalSourceCopyGate` | Cycle-4 Stage A v1 (Kaggle T4) | `RuntimeError: value cannot be converted to type c10::Half` | Fixed by wrapping `CausalSourceCopyGate` attention score matmul/mask/softmax in `torch.amp.autocast(device_type=h.device.type, enabled=False)` (Stage A v2 completed cleanly, `36/36` CUDA pytest). |
| `WorldController_v3` on `causal_vs_associational` (`FP-5Axis-30`) | Cycle-4 `EXP-FP4` (Local MPS) | `1/6 (16.7%)` vs `Naked` `2/6 (33.3%)` (overall `15/30` vs `9/30`) | Ledger/bitemporal routing drove `+50.0 pp` on relational and `+33.3 pp` on bitemporal tasks, but lacked a dedicated structural causal model (`do()`-calculus) solver, slightly regressing `1` causal task. |
