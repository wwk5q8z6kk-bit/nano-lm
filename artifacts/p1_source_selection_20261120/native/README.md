# Native Nano unseen-value copying

A supervised source-position and copy-gate objective was tested on newly generated, public synthetic slot-state transcripts. All nine scheduled Vertex fits and the three same-seed local sensitivity fits completed.

## Held-out results

Every row uses the same 96 held-out encounters, 183 present fields and 201 absent fields. Counts within a cell follow model seeds 1481, 3527 and 6029. Repeated evaluations share the same test encounters.

| Objective | Present exact / 183 | Present exact, mean | Present omitted | Present wrong nonempty or malformed | False fill / 201 absent | All exact / 384 |
|---|---:|---:|---:|---:|---:|---:|
| Full-sequence LM | 0 / 0 / 0 | 0.0% | 87 / 41 / 100 | 96 / 142 / 83 | 15 / 45 / 9 | 186 / 156 / 192 |
| Answer-only LM | 33 / 8 / 16 | 10.4% | 6 / 14 / 6 | 144 / 161 / 161 | 22 / 19 / 12 | 212 / 190 / 205 |
| Answer-only + source alignment | 110 / 110 / 77 | 54.1% | 22 / 12 / 6 | 51 / 61 / 100 | 39 / 44 / 58 | 272 / 267 / 220 |

Against answer-only training, source alignment changed mean present exact recovery by +43.7 percentage points. The paired encounter bootstrap interval is [39.1, 48.3] points, conditional on these three model seeds and one synthetic generator.

## Evidence and errors

| Held-out subset | Cases per seed | Full correct | Answer-only correct | Aligned correct |
|---|---:|---:|---:|---:|
| MEDICATION present | 38 | 0 / 0 / 0 | 0 / 0 / 0 | 16 / 22 / 17 |
| DOSE present | 51 | 0 / 0 / 0 | 20 / 8 / 16 | 50 / 47 / 42 |
| ALLERGY present | 44 | 0 / 0 / 0 | 0 / 0 / 0 | 23 / 14 / 9 |
| VITAL_BP present | 50 | 0 / 0 / 0 | 13 / 0 / 0 | 21 / 27 / 9 |
| Final retraction returned ABSTAIN | 52 | 45 / 30 / 50 | 48 / 52 / 50 | 41 / 30 / 13 |
| Present with multiple distinct prior values | 92 | 0 / 0 / 0 | 16 / 4 / 4 | 64 / 64 / 40 |

The source error audit distinguishes copying a previous value, copying from another field, other quoted text, absent values and malformed text. An explicit ABSTAIN is the exact output for an absent field. Empty or malformed predictions count as errors.

## Design and reproducibility

The unchanged native hybrid model has 65,524 trainable parameters. Each arm received the same sealed train data, schedule and optimizer budget of 240 steps with 24 examples per step. Starting weights, batch schedules and sampled token counts matched across objectives within each Vertex seed. All inference was greedy for up to 16 tokens from the transcript and field query.

Training used 240 encounters and 24 whole values. Development and test used 48 and 96 encounters with 16 distinct whole values each. Individual token fragments and the generator grammar are shared. Whole values and transcripts are disjoint across splits and from the 840 archived records checked. The earlier 0/345 archival result uses a different test set. The fresh full-sequence arm is the matched comparison. The source-aligned arm changes supervision and training computation. These measurements cover the generated task and give no estimate of performance on clinical conversations.

Checkpoint hashes, state tensors, sampled batch invariants and every stored dev/test prediction were independently checked. Greedy single-row CPU replay matched 144/144 selected field outputs across nine Vertex checkpoints. Seed 1481 also ran locally using PyTorch 2.11.0, compared with Vertex 2.4.0+cpu. Exact backend output agreement was 275/180/244 of 384 fields respectively. The local and Vertex environments shared the batch schedule. Identical initialization across backends was observed for 0/3 objectives. This supplementary environment comparison reuses seed 1481.

Vertex custom job 9019587804438462464 reached JOB_STATE_SUCCEEDED at 2026-11-20T15:11:42Z. The worker ran in us-central1 on one N2D-standard-4 CPU machine. Worker start to terminal was 14.1 minutes, estimated at $0.047. Including provisioning from submission gives $0.089. These are list-price estimates before credits or tax. The actual invoice and credit application were unavailable at verification.

The split seed material and training code are pinned in `input-sha256.txt`. `verification.json` provides all per-seed counts and source hashes, `error-audit.json` supplies categories, examples and paired resampling, and `cloud/cost-and-terminal.json` records the resource outcome.

## Repeating the training

Use Python with NumPy and CPU PyTorch. The cloud worker used PyTorch 2.4.0+cpu. Run these commands in this directory. The generator reproduces the original train, development and test inputs, including their row order. The six source modules in `research/` are the dependency snapshot used for the experiment.

```sh
python3 -B make_data.py
shasum -a 256 -c input-sha256.txt
python3 -B launch_matrix.py /path/to/new/results
```

The launcher writes fresh checkpoints and predictions for all nine fits. The archived scalar replay and per-seed verification records describe the original Vertex checkpoints. Local runtime versions can affect initialization and outputs. To check the portable records, run `shasum -a 256 -c MANIFEST.sha256` here.
