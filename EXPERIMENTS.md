# Experiments — Protocols, Configurations & Verdicts

Executed on **RunPod `NVIDIA L40S` 48GB (`CUDA 13.2`, `torch 2.9.1+cu128`, BF16)** under pod `2v2dt0j536rh58` (verified removed; `4,302s` @ `USD 1.09/hr` = **USD 1.3026** pod-attributed spend in [SPEND_AND_PROVENANCE.json](artifacts/nanolm_holonomy_v1/SPEND_AND_PROVENANCE.json)).

---

## Experiment X1 — Non-Abelian `S_4` Group Tracking & Dynamic Associative Recall (`3 Seeds`)

- **Source Artifact:** [x1_state_tracking.json](artifacts/nanolm_holonomy_v1/x1_state_tracking.json)
- **Setup:** Iso-parameter `~2.46M` models (`vocab=512`), seeds `[101, 202, 303]`.
  - Task 1A (`S_4`): Trained on interleaved prefixes up to `L=48` generators (`400` AdamW steps, `bs=32`), tested unbroken on `L = 16, 48` (ID) and `96, 192` (2x, 4x OOD).
  - Task 1B (`SET/OVERRIDE`): Trained on `10` events (`~65` tokens, `350` steps), tested at `10, 20, 40` events (`4x` sequence length).

| Architecture (`~2.46M`) | Params | `S4` `L=48` (mean ± std) | `S4` `L=96` | `S4` `L=192` | Recall `ev=10` | Recall `ev=20` | Recall `ev=40` (4x OOD) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `b0-transformer-small_x1` | 2,460,480 | `0.0208 ± 0.0074` | `0.0312 ± 0.0255` | **`0.0729 ± 0.0195`** | `0.2240 ± 0.0368` | `0.1302 ± 0.0265` | `0.1302 ± 0.0195` |
| `b1-abelian-ssm-small_x1` | 2,460,864 | `0.0312 ± 0.0128` | `0.0417 ± 0.0074` | `0.0469 ± 0.0338` | **`0.2813 ± 0.0337`** | **`0.1875 ± 0.0221`** | **`0.2448 ± 0.0410`** |
| `gen1-h1-hspa-spinor-small_x1` | 2,465,472 | `0.0417 ± 0.0074` | `0.0677 ± 0.0321` | `0.0417 ± 0.0321` | `0.2240 ± 0.0390` | `0.1354 ± 0.0321` | `0.2136 ± 0.0321` |
| `gen1-h2-hspa-predcode-small_x1` | 2,466,630 | `0.0260 ± 0.0266` | `0.0312 ± 0.0000` | `0.0417 ± 0.0321` | `0.0885 ± 0.0769` | `0.0677 ± 0.0410` | `0.0937 ± 0.0510` |
| `gen2-h3-hspa-evolved-small_x1` | 2,469,696 | **`0.0521 ± 0.0195`** | **`0.0729 ± 0.0390`** | `0.0417 ± 0.0321` | `0.1771 ± 0.1023` | `0.1406 ± 0.0255` | `0.1771 ± 0.1023` |

- **Verdict:** Inconclusive / undertrained for 24-class `S_4` grokking across all arms at 400 steps; `O(1)` constant-state models (`B1` `24.5%`, `H1` `21.4%`) retained higher 4x OOD associative override accuracy than Transformer (`13.0%`).

---

## Experiment X2 — Iso-Parameter LM Evolution (`2 Seeds`, `~14.0M`, `6.55M` Tokens/Arm)

- **Source Artifact:** [x2_lm_evolution.json](artifacts/nanolm_holonomy_v1/x2_lm_evolution.json)
- **Setup:** Controlled multi-domain corpus (`code`, `math`, `prose`, `state`), `vocab=4096`, `seq=256`, `batch=16`, `800` steps/seed on seeds `[42, 137]`. Automatic `ExperimentRegistry` gate promoted strictly on multi-seed held-out mean BPB (`min_delta = 0.0002`).

| Architecture (`~14.0M`) | Params | State @ 8K | Mean Val BPB (`± std`) | Code BPB | Math BPB | Prose BPB | State BPB | State Probe Acc | Math Probe Acc | Registry |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `b0-transformer-micro` | 13,966,464 | 24,576 KB | `0.3794 ± 0.00090` | **0.2014** | 0.7505 | 0.0914 | 0.4745 | 1.04% | 4.17% | `BASELINE` |
| `b1-abelian-ssm-micro` | 14,010,240 | **24.0 KB** | `0.3826 ± 0.00016` | 0.2095 | 0.7456 | **0.0904** | 0.4849 | 1.04% | 2.08% | Rejected |
| `gen1-h1-hspa-spinor-micro` | 14,019,456 | **24.0 KB** | `0.3800 ± 0.00040` | 0.2096 | 0.7534 | 0.0905 | 0.4666 | 1.04% | 1.04% | Rejected |
| **`gen1-h2-hspa-predcode-micro`** | 14,022,536 | **24.0 KB** | **`0.3723 ± 0.00235`** | 0.2094 | **0.7456** | 0.0906 | **0.4438** | **13.54%** | **8.33%** | **`PROMOTED BEST`** |
| `gen2-h3-hspa-evolved-micro` | 14,027,648 | 33.0 KB | `0.3803 ± 0.00180` | 0.2096 | 0.7466 | **0.0904** | 0.4748 | 2.08% | 4.17% | Rejected |

---

## Experiment X3 — 4B-Class (`~3.84B`) Context-Length Scaling on `L40S` (BF16)

- **Source Artifact:** [x3_4b_scaling.json](artifacts/nanolm_holonomy_v1/x3_4b_scaling.json)

| Context `T` | `TRANSFORMER_4B` (`3.843B`) Peak VRAM | `TRANSFORMER_4B` KV State | `TRANSFORMER_4B` Decode (`ms/tok`) | `HSPA_4B` (`3.845B`) Peak VRAM | `HSPA_4B` State | `HSPA_4B` Decode (`ms/tok`) |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 7,622.35 MB | 48.0 MB | 35.932 ms | 10,877.26 MB | **1.781 MB** | 122.590 ms |
| 2,048 | 8,220.44 MB | 192.0 MB | 34.197 ms | 8,308.06 MB | **1.781 MB** | 144.367 ms |
| 8,192 | 10,607.49 MB | 768.0 MB | 34.511 ms | **8,317.38 MB** | **1.781 MB** | 143.188 ms |
| **16,384** | 13,824.34 MB | 1,536.0 MB | 41.778 ms | **8,318.09 MB (-5,506 MB)** | **1.781 MB (`862x` smaller)** | 135.042 ms |

---

## Experiment X4 — `Qwen/Qwen3-1.7B` Holonomic Sidecar + Early-Stop SFT + Challenging GRPO

- **Source Artifact:** [x4_foundation_holonomy.json](artifacts/nanolm_holonomy_v1/x4_foundation_holonomy.json) vs `nanolm_frontier_v1` [e2_foundation_15d.json](artifacts/nanolm_frontier_v1/e2_foundation_15d.json)
- **Decontamination:** `overlaps = 0` for both SFT (`35` examples) and challenging RLVR (`24` examples) vs `56` held-out tasks.

| Model Arm (`<= 4B`, `Qwen3-1.7B` Backbone) | Params | Passed / 56 | Accuracy `[CI95]` | Mean Partial | Coding | Math | Struct JSON | Planning | Hard (`20`) | Active GRPO Updates |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `Base` (`frontier_v1`) | 1.721B | 34/56 | 60.71% `[48.2, 71.4]` | 0.7439 | **3/4** | 1/4 | **3/4** | 0/4 | 45.0% | — |
| `LoRA_SFT` (5 ep, `frontier_v1`) | 1.738B | **36/56** | **64.29%** `[51.8, 75.0]` | **0.7708** | 0/4 (collapsed) | **4/4** | 2/4 | 0/4 | 55.0% | — |
| `Frontier_v1` (`DexDiff + VAnchor`, 5 ep + GRPO) | 1.753B | 35/56 | 62.50% `[51.8, 73.2]` | 0.7554 | 1/4 | 2/4 | 2/4 | **1/4** | **60.0%** | 1 / 72 |
| **`Holonomic_SFT2_GRPO` (`X4`)** | 1.747B | 35/56 | 62.50% `[51.8, 73.2]` | 0.7661 | **2/4** (restored) | **4/4** | **3/4** | **1/4** | 55.0% | **18 / 72** |

- **Verdict:** Early-stopping SFT (`2` epochs, `0.3092` loss) plus dynamic RLVR curriculum fixed the coding collapse (`0/4 -> 2/4`), kept `math_logic = 4/4`, lifted `structured_json = 3/4`, raised GRPO training reward monotonically (`0.2271 -> 0.3636 -> 0.4279` across `18` active updates), and verified exact `0.00e+00` step-0 logit preservation (`mean_abs_sidecar_gate = 0.000816`).
