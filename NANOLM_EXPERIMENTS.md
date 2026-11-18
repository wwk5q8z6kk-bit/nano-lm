# NANOLM Experimental Protocol, Telemetry & Verdict Ledger

All experiments were executed under strict iso-parameter bounds (`±0.45%` deployed parameter parity, `< 4.000B` ceiling), fixed PRNG seeds, identical decontaminated validation shards (`val_{web,textbook,math,code}.bin`), and immutable cryptographic SHA-256 verification ([`artifacts/nanolm_discovery_v1/SHA256SUMS`](artifacts/nanolm_discovery_v1/SHA256SUMS)).

---

## 1. Execution Hosts & Dataset Provenance (`$0.00` Cloud Spend)

| Backend Host | Hardware / Runtime | Experiments Executed | Paid Spend |
|---|---|---|---:|
| Local Apple Silicon | Apple M3 Max (`16` cores, `64 GB` UMA), `macOS-26.7.3-arm64`, `torch 2.11.0`, `mps` (`bfloat16`) | `EXP-INV-01` (Cycles 1–3), `EXP-INV-02` (Cycles 1 & 3), `EXP-INV-03`, `EXP-INV-04`, `EXP-INV-05A`, `EXP-INV-05B` (`3.89B`) | `$0.00` |
| Kaggle Free GPU | `2x NVIDIA Tesla T4` (`CUDA 13.0`, `14.56 GiB`/GPU), `Linux-x86_64`, `torch 2.11.0+cu130` (`float16` TensorCore + `fp32` scan) | `CUDA Invariant Audit`, `EXP-INV-06A` (`1,000`-step `4x` horizon), `EXP-INV-06B` (`16 GB` KV wall) | `$0.00` |
| Corpus Stream (`nanolm.lab.corpus.v1`) | `vocab=32,768`, `train_sha256=867beb0d...` (`800M` tok locally; `40M`-tok shuffled prefix + full `4x 1.0M` val shards on Kaggle) | `bpt`: `web=4.5163`, `textbook=5.0293`, `math=3.0994`, `code=3.3728` | `$0.00` |

---

## 2. `EXP-INV-01`: Mechanistic Falsification & 4x Extrapolation (`~115k`, `3` Seeds `[101, 202, 303]`, `360` Steps)

All synthetic benchmarks use zero external storage (`seed` deterministic generation via [`nanolm_discovery/eval_benchmarks.py`](nanolm_discovery/eval_benchmarks.py)) with `chunk_size=32`.

| Cycle | Candidate | Params | `S3_ID` (`T=32`) | `S3_4x` (`T=128`) | `S3_8x` (`T=256`) | `OW_ID` (`T=48`) | `OW_4x` (`T=192`) | `Needle_ID` (`96/4`) | `Needle_OOD` (`256/8`) | `EpistAUROC` (`int` / `ent`) | `OOD_Comp` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | `b0_transformer` | `115,008` | `0.226` | `0.182 ± 0.003` | `0.175` | `0.255` | `0.175 ± 0.004` | `0.292` | `0.188 ± 0.007` | `0.603 / 0.603` | `0.182` |
| C1 | `b1_gdn2_hybrid` | `114,768` | `0.236` | `0.188 ± 0.003` | `0.176` | `0.998` | `0.965 ± 0.012` | `0.767` | `0.693 ± 0.420` | **`0.965` / `0.965`** | `0.615` |
| C1 | `i1_phmn` | `115,032` | **`0.348`** | **`0.295 ± 0.079`** | **`0.264`** | **`1.000`** | **`0.986 ± 0.009`** | `0.999` | `0.962 ± 0.012` | `0.961 / 0.961` | `0.747` |
| C1 | `i2_nebn` | `114,860` | `0.228` | `0.185 ± 0.005` | `0.175` | `0.987` | `0.849 ± 0.012` | `0.986` | `0.952 ± 0.037` | `0.362 / 0.654` | `0.662` |
| C1 | `i3_hrwn` | `115,138` | `0.233` | `0.186 ± 0.003` | `0.175` | `0.235` | `0.146 ± 0.020` | `0.294` | `0.194 ± 0.011` | `0.204 / 0.585` | `0.176` |
| C2 | `g2_mprn` | `114,786` | `0.295` | `0.242 ± 0.088` | `0.213` | `1.000` | `0.971 ± 0.010` | `0.999` | `0.999 ± 0.001` | `0.530 / 0.947` | `0.737` |
| C2 | `g2_no_reflection` | `114,778` | `0.228` | `0.184 ± 0.003` | `0.175` | `0.999` | `0.956 ± 0.061` | `0.360` | `0.364 ± 0.551` | `0.773 / 0.923` | `0.501` |
| C2 | **`g2_no_precision`** | `115,162` | `0.346` | `0.285 ± 0.076` | `0.230` | `1.000` | `0.969 ± 0.014` | **`1.000`** | **`1.000 ± 0.001`** | `0.934 / 0.934` | **`0.751`** |
| C2 | `g2_no_scam` | `115,168` | `0.298` | `0.224 ± 0.059` | `0.196` | `1.000` | `0.982 ± 0.019` | `0.833` | `0.729 ± 0.400` | `0.514 / 0.956` | `0.645` |
| C2 | `g2_std_swiglu` | `114,658` | `0.245` | `0.185 ± 0.003` | `0.177` | `1.000` | `0.971 ± 0.003` | `0.999` | `0.989 ± 0.018` | `0.483 / 0.882` | `0.715` |
| C3 | **`g3_hmwr`** | `115,024` | `0.319` | `0.280 ± 0.087` | `0.245` | `0.999` | `0.977 ± 0.016` | `0.995` | `0.961 ± 0.063` | `0.686 / 0.875` | **`0.739`** |
| C3 | `g3_no_depth_adapt` | `114,638` | `0.235` | `0.185 ± 0.003` | `0.176` | `0.997` | `0.954 ± 0.018` | `0.999` | `1.000 ± 0.000` | `0.732 / 0.871` | `0.713` |

---

## 3. `EXP-INV-02` & `EXP-INV-04`: Multi-Domain `~20.5M` Corpus & Ablations (`2` Seeds `[42, 137]`, `250` Steps = `1,024,000` tok/seed, `AdamW`, M3 Max `bfloat16`)

| Exp | Candidate | Params | `compBPB` | `ood2x` (`512`) | `web` | `textbook` | `math` | `code` | Throughput | `LineageRegistry` Verdict (250-step gate) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 02 | `b0_transformer` | `20,550,528` | `2.4652 ± 0.0019` | `2.4703 ± 0.0016` | `2.0743` | `1.9667` | **`2.7830`** | `3.0367` | `18,200 tok/s` | `BASELINE` |
| 02 | `b1_gdn2_hybrid` | `20,567,544` | `2.4818 ± 0.0009` | `2.4782 ± 0.0009` | `2.0922` | `1.9850` | `2.7925` | `3.0574` | `14,929 tok/s` | `REJECTED` (`bpb_delta=-0.0166`) |
| 02 | `i1_phmn` | `20,569,176` | `2.4822 ± 0.0026` | `2.4794 ± 0.0025` | `2.0923` | `1.9849` | `2.7960` | `3.0555` | `8,929 tok/s` | `REJECTED` (`bpb_delta=-0.0170`) |
| 02 | `i2_nebn` | `20,553,336` | `2.4889 ± 0.0063` | `2.4857 ± 0.0062` | `2.1004` | `1.9936` | `2.8029` | `3.0586` | `9,897 tok/s` | `REJECTED` (`bpb_delta=-0.0237`) |
| 02 | **`i3_hrwn`** | `20,598,918` | **`2.4520 ± 0.0015`** | **`2.4560 ± 0.0015`** | **`2.0694`** | **`1.9561`** | `2.7959` | **`2.9869`** | `8,555 tok/s` | **`PROMOTED_TO_BEST`** (`+0.0131` BPB) |
| 04 | `g2_mprn` | `20,527,782` | `2.5161 ± 0.0025` | `2.5134 ± 0.0032` | `2.1162` | `2.0071` | `2.8249` | `3.1161` | `7,395 tok/s` | `REJECTED` (`bpb_delta=-0.0640` vs best) |
| 04 | `g2_no_reflection` | `20,573,790` | `2.5114 ± 0.0065` | `2.5104 ± 0.0069` | `2.1115` | `2.0051` | `2.8262` | `3.1027` | `7,628 tok/s` | `REJECTED` (`bpb_delta=-0.0593`) |
| 04 | `g2_no_precision` | `20,571,486` | `2.5047 ± 0.0048` | `2.5032 ± 0.0057` | `2.1120` | `2.0045` | `2.8131` | `3.0892` | `8,008 tok/s` | `REJECTED` (`bpb_delta=-0.0526`) |
| 04 | `g2_no_scam` | `20,525,472` | `2.4897 ± 0.0039` | `2.4866 ± 0.0040` | `2.0991` | `1.9910` | `2.8016` | `3.0671` | `7,846 tok/s` | `REJECTED` (`bpb_delta=-0.0376`) |
| 04 | `g2_std_swiglu` | `20,534,694` | `2.5264 ± 0.0067` | `2.5243 ± 0.0061` | `2.1220` | `2.0178` | `2.8332` | `3.1327` | `6,186 tok/s` | `REJECTED` (`bpb_delta=-0.0744`) |
| C3 | `g3_hmwr` | `20,551,430` | `2.5085 ± 0.0004` | `2.5070 ± 0.0002` | `2.1104` | `2.0042` | `2.8171` | `3.1024` | `4,235 tok/s` | `REJECTED` at 250 steps (reversed at `1000` in `06A`) |
| C3 | `g3_no_depth_adapt` | `20,539,122` | `2.5210 ± 0.0003` | `2.5188 ± 0.0003` | `2.1215` | `2.0166` | `2.8316` | `3.1141` | `6,522 tok/s` | `REJECTED` (`-0.0125` worse than `g3_hmwr`) |

---

## 4. `EXP-INV-03`: $2 \times 2$ Optimization Factorial (`discovery_18m`, `250` Steps, `2` Seeds)

| Architecture | `AdamW + CE` `compBPB` (`ood2x`) | `Muon + 0.10 JEP` `compBPB` (`ood2x`) | `Muon+JEP` Penalty at `250` Steps | Winner Within Arm |
|---|---:|---:|---:|---|
| `b0_transformer` | `2.4652 ± 0.0019` (`2.4703`) | `2.5117 ± 0.0021` (`2.5080`) | `+0.0465` | `i3_hrwn` wins both cells |
| `i3_hrwn` (`BEST` @ 250) | **`2.4520 ± 0.0015`** (`2.4560`) | **`2.4864 ± 0.0003`** (`2.4924`) | `+0.0344` | **`+0.0132` (`AdamW`), `+0.0253` (`Muon+JEP`)** |

---

## 5. `EXP-INV-05A`: Multi-Scale Iso-Parameter Law (`8.65M -> 20.55M -> 47.64M`, `AdamW+CE`, `250` Steps)

| Tier | `b0_transformer` (`compBPB` / `ood2x` / `tps`) | `i3_hrwn` (`compBPB` / `ood2x` / `tps`) | BPB Advantage (`ID` / `2x OOD`) |
|---|---:|---:|---:|
| `scale_8m` | `2.4886 ± 0.0006` / `2.4916` / `37,871` | **`2.4875 ± 0.0035`** / **`2.4910`** / `18,658` | `+0.0011` / `+0.0006` |
| `discovery_18m` | `2.4652 ± 0.0019` / `2.4703` / `18,200` | **`2.4520 ± 0.0015`** / **`2.4560`** / `8,555` | **`+0.0132` / `+0.0143`** |
| `scale_42m` | `2.4668 ± 0.0056` / `2.4736` / `9,669` | **`2.4551 ± 0.0006`** / **`2.4586`** / `5,876` | **`+0.0117` / `+0.0150`** |

---

## 6. `EXP-INV-06`: Kaggle Tesla T4 `4x` Horizon Crossover & Cross-Hardware Audit (`CUDA 13.0`)

### 6.1 Cross-Hardware CUDA (`float16` / `fp32`) Invariant Verification
Every candidate passed `numel() == census`, `causality_max_err = 0.0`, `grads_finite = True`, and `duality_max_err < 2e-6` on NVIDIA Tesla T4 (`b0: 9.54e-7`, `i1_phmn: 1.88e-6`, `i3_hrwn: 9.54e-7`, `g3_hmwr: 9.54e-7`).

### 6.2 `EXP-INV-06A`: `4x` Token Horizon Learning Curves (`1,000` Steps = `4,096,000` tok/seed, `discovery_18m`, Seeds `[42, 137]`)

| Arm (`Opt`) | Step `250` | Step `500` | Step `750` | Final Step `1,000` (`compBPB`) | **`ood2x` (`512`)** | `web` / `textbook` / `math` / `code` | CUDA Speed / Peak VRAM |
|---|---:|---:|---:|---:|---:|---|---:|
| `b0_transformer_adamw` | `2.4139` | `2.2470` | `2.1571` | **`2.1188 ± 0.0050`** | `2.1526 ± 0.0039` *(`+0.0338` cliff!)* | `1.8578 / 1.6866 / 2.4499 / 2.4810` | `27,324 tok/s` / `2,555 MiB` |
| `i3_hrwn_adamw` | `2.4389` | `2.2908` | `2.2065` | `2.1654 ± 0.0043` | `2.1710 ± 0.0038` | `1.8926 / 1.7247 / 2.5077 / 2.5367` | `27,011 tok/s` / `3,306 MiB` |
| **`g3_hmwr_adamw`** | `2.4430` | `2.2535` | `2.1692` | **`2.1342 ± 0.0053`** | **`2.1322 ± 0.0057`** *(Beats B0!)* | `1.8694 / 1.7017 / 2.4638 / 2.5020` | `12,843 tok/s` / `4,362 MiB` |
| `g3_hmwr_muon_ce` | `2.5050` | `2.2678` | `2.1786` | `2.1457 ± 0.0043` | `2.1445 ± 0.0044` | `1.8793 / 1.7107 / 2.4800 / 2.5128` | `10,609 tok/s` / `4,331 MiB` |
