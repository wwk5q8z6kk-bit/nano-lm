# PREREG LAB-ARCH-02 — matched-budget token-mixer & depth comparison (Track A)

*Registered 2026-11-16 before any paid compute. Amendments are appended, never edited in place.*

## Question
At matched non-embedding parameters (~29.4M ± 1%, 46M total incl. tied 32k embedding), which token
mixer / depth scheme reaches the lowest held-out loss, and does the ranking survive matching training
FLOPs instead of tokens?

## Arms (lab/arch/arms.py)
T0_dense, T1_vres, T2_swa, H1_gla, H2_delta, H3_delta_vres, P1_gla, P2_delta, R1_loop, R2_delta_loop.
Plus LAB-OPT-01: T0_dense and best-arm with Muon.

## Fixed protocol
- Data: LAB-DATA-01 v1 corpus (sha256 in manifest), identical sequential order for every arm, single epoch, no repeats.
- seq 1024, batch 131,072 tokens, AdamW (0.9, 0.95) lr 3e-3 wd 0.1, WSD (2% warmup, 20% linear decay), grad-clip 1.0, bf16 autocast.
- Budget: **600M tokens** per arm (~13 tok/total-param, ~20 tok/non-emb-param) in `--match tokens` mode;
  looped arms (R1/R2) additionally in `--match flops` mode (tokens scaled so FLOPs = T0's).
- Seeds: 2 (0, 1) for every arm; a 3rd seed for any pair whose difference is < 2× the seed spread.
- Metrics: per-domain val BPB (web/textbook/math/code, 128×1024 tokens each, document-disjoint shards,
  13-gram filtered), composite BPB (0.4/0.2/0.2/0.2), copy-probe loss & accuracy, context-utilisation
  (loss at positions 0–64 vs last 256), train tok/s, peak memory, analytic FLOPs/token (lab/arch/flops.py),
  decode-state bytes/token.

## Hypotheses & decision rules
- H-A1: a delta-rule hybrid (H2/H3) matches T0 composite BPB within 1% **and** cuts global-attention KV by ≥ 60%.
- H-A2: R1_loop beats T0 at matched tokens (expected, more compute); **the scientific claim requires it to
  beat T0 at matched FLOPs**.
- H-A3: value residual lowers composite BPB by ≥ 1% in both T and H families.
- A "win" requires: ≥ 1% composite-BPB improvement, sign consistent across both seeds, no domain regressing > 1%.
  Results under 1% are reported as "no detectable difference at this scale".
- Copy-probe: pure recurrent arms are expected to lose; quantified, not used to reject hybrids.

## Compute plan & cost (measured pilot first)
- **Throughput was not measured before this registration; the earlier estimate ("10–20 min per arm at
  50–150M for 1–3B tokens on one H100") was wrong.** Analytic: T0 = 0.309 GFLOP/token → 600M tokens =
  1.9e17 FLOPs. At an assumed 25–35% MFU: H100 (989 TF bf16) ≈ 0.8–1.1M tok/s → ~10–13 min;
  RTX 4090 (~165 TF) ≈ 130–190k tok/s → ~55–75 min. 150M models at 3B tokens: ≈ 3–4 h on an H100.
- **Pilot (P0)**: one community RTX 4090 ($0.34/h) and/or H100 ($2.69/h), ≤ 20 min each, measuring
  tok/s for T0, H2 (pure-PyTorch and fla kernels), R1 with/without torch.compile. **Cap: $3.**
- Full grid: 10 arms × 2 seeds + 2 FLOP-matched + 2 Muon ≈ 24 runs. Budget cap **$30** for this experiment.
- Stopping conditions: kill any run with non-finite loss or loss > T0's at the same step by > 0.5 nats after
  20% of training (record as divergence, retry once at lr/2); terminate pods immediately when the queue is
  empty; stop the grid if the measured cost projection exceeds $30.

## Expected value
Decides which mixer family Track A scales (LAB-SCALE-01) and which mechanism is worth attempting to
transfer into Track B (e.g. delta-rule conversion of a pretrained model for long-context efficiency).
