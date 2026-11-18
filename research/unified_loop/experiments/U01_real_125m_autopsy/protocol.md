# Experiment U-01 — Forensic Mechanistic Autopsy of Real 600M-Token Checkpoints (`S0_dense_125m`, `S1_nanolm_v2_125m`, `X4_nanolm_v2_46m`)

- **Experiment ID:** `EXP-U01`
- **Status:** Preregistered before execution
- **Hardware / Cost:** Local Apple Silicon M3 Max (`mps` `bfloat16`), `$0.00` paid compute
- **Primary Question:** Do the failure mechanisms hypothesized on tiny synthetic toys (`SiLU` positive-orthant key collapse, output-gate saddle attenuation, `VRes` anchor saturation, and length-extrapolation / needle-recall degradation) actually occur inside real `600M`-token Muon-trained `46M` and `125M` checkpoints (`S0_dense_125m`, `S1_nanolm_v2_125m`, `X4_nanolm_v2_muon`)?

## Competing Hypotheses

1. **H-U01-A (`SiLU` Key-Cone Collapse Survives Scale):** `NextGenDeltaMixer` applies `F.silu` to `QK` before `L2` normalization, forcing keys into a positive-orthant cone ($\mathbb{E}_{i \neq j}[\langle k_i, k_j\rangle] \gg 0$) even after `600M` tokens of `Muon` training at `125M`, degrading long-range copy and multi-needle recall relative to `S0_dense_125m`.
   - *Alternative H-U01-A0:* `600M` tokens of `Muon` optimization learns orthogonalizing output/proj rotations or negative pre-activations so key cosines approach `~0` and the real bottleneck is instead attention-layer spacing (`3:1`) or `beta in (0,1)` / decay horizon.
2. **H-U01-B (Zero-Shot Length Extrapolation `T=256..2048` Diverges):** Because `S0_dense_125m` uses RoPE ($\theta=10{,}000$) trained strictly at `seq=1024`, `S0` will suffer a perplexity cliff at `T=2048` (`2x` training length) whereas `S1_125m` (`75%` Delta, position-free recurrence) will extrapolate better on language BPB while losing on long-distance exact literal copy / needle recall.
3. **H-U01-C (Rare/Unseen Needle Recall Gap):** On zero-shot Multi-Needle Associative Recall (`MQAR`) embedded in real `val_web` distractors, `S1_125m` will underperform `S0_dense_125m` as needle count `n` and distance `D` grow, diagnosing exact associative capacity limits of `DDDADDDADDDADDDA` vs `A*16`.

## Controls & Falsification Criteria

- Reproduce published `S0_dense_125m`, `S1_nanolm_v2_125m`, and `X4_nanolm_v2_muon` `val_*` BPB and `512`-tok copy accuracy within `|ΔBPB| < 0.002` before interpreting internal probes.
- Measure per-layer key cosine, key effective rank, fraction positive, decay $\exp(\text{loga})$, write $\beta_t$, output gate $|\mathrm{SiLU}(g)|$, and `v_gate` $\alpha_t$ across ALL Delta and Attention layers of `S1_125m` and `X4_46m`.
- Report all positive and negative measurements verbatim.
