# Dataset authority record — LoRA cross-model control

**2026-08-06.** Written **before** any pod is provisioned, per the standing
directive: *"Do not launch a paid training pod unless the training script can
access a complete, verified, authorized dataset."*

Purpose of the run: fine-tune `Llama-3.2-3B-Instruct` on Nano's own training
partition so the cross-model control in `RESULT_CROSSMODEL_CONTROL.md` §4 holds
task-training constant and varies only vocabulary exposure and scale. This is a
**control experiment**, not a Nano training run; no Nano checkpoint is produced
and no H-cycle gate is touched.

## Identification

| field | value |
|---|---|
| dataset name | Nano replay-mixture state/span dataset, **fit partition** |
| version / generator | `nano.replay-mixture-dataset.v1` |
| generator sha256 | `5f036831ac7385c49177d9b724f6939049873179654072bc19dc178144e2af40` |
| locality | **local, already on disk** — not downloaded, not privately hosted |
| exact source path | `artifacts/nano_h5/data/fit.jsonl` |
| license / permitted use | project-internal **synthetic** output of this repo's own generator; no third-party licence attaches. Contains no real patient data, no scraped text, no private user content. |
| file count | 3 (`fit.jsonl`, `calibration.jsonl`, `manifest.json`) |
| total size | 9,900,478 bytes |
| fit.jsonl | 9,134,638 B · sha256 `79f7581efbd989f48a474bdd2079e7ca163c2a2682bb48fb8398bd99febe22fb` |
| calibration.jsonl | 734,627 B · sha256 `ce0562ccb44ee83963eace0d873773addaee8e49f29499a6b720b16335930e70` |
| manifest.json | 31,213 B · sha256 `2569e7b27b53ef741fc92d1545ca323367155d960e77ff0e6852b32db0d32f31` |
| records | fit **11,200** examples · calibration **800** examples |
| tokenizer | `sft/tokenizer.json` sha256 `bae49648bfcc4904c50e2f006ee184bd26e74454ee170663e30a8e71640ce3c9` — **Nano's tokenizer, used for Nano only.** The LoRA run uses the Llama-3.2 tokenizer shipped with the base model; Nano's is recorded here because the data was generated against it. |

## Splits

- **fit** (11,200) — the only partition used for LoRA training.
- **calibration** (800) — **not used** in this run.
- **development** (`artifacts/nano_h6/kaggle/dataset-dev`, 1,000 records,
  sha256 `9c893d8e64110287b433d567e0e9abb42c611ecba33b40de192741324d37e290`) — **evaluation only**, never
  trained on. This is the partition the surface arms are measured over.

## Deduplication and contamination controls

- Manifest `overlap_audit.all_hard_intersections_zero` = **True**.
- The development partition declares its own isolation from training:
  `{"answer_templates_disjoint": true, "denial_phrases_disjoint": true, "fresh_v0_read_by_generator": false, "open_value_lexicons_disjoint": true, "question_templates_disjoint": true, "transcripts_disjoint": true, "uncertainty_phrases_disjoint": true, "worlds_disjoint": true}`.
  Every lexical pool that decides an epistemic state is disjoint from training —
  which is precisely the property under test.
- **Contamination risk specific to this run:** none introduced. The base model
  (Llama-3.2-3B) was pretrained on public web text and may have seen phrasings
  resembling these, but it cannot have seen these generated documents, which are
  synthetic and unpublished. This is a confound on absolute accuracy, not a
  train/test leak.
- **The one leak that would invalidate the result:** training on any part of the
  development partition, or on the vendored negspacy/medspacy inventories that
  the held-out arms are built from. Neither is touched. The external lexicons
  remain `evaluation-only` as recorded in `data/external/*/`.

## Transfer

| field | value |
|---|---|
| method | `runpodctl send` / `scp` of a single tar of `fit.jsonl` + `manifest.json` |
| payload size | ~9.2 MB uncompressed, ~2 MB gzipped |
| expected duration | seconds |
| expected transfer cost | \$0 (ingress) |
| destination path | `/workspace/nano_lora/data/` on the pod |
| storage required | < 1 GB data + ~7 GB base model weights |
| post-transfer verification | `sha256sum` on the pod compared against the hashes in this file; the training script aborts on mismatch |

## Cost envelope

Cheapest suitable on-demand GPU observed 2026-08-06: **RTX A5000 24 GB at
\$0.16/hr**. A 3B LoRA over 11,200 short examples is well under an hour.
**Budget: \$2, hard stop.** If the run exceeds it, kill the pod and report.

## Authorization gate

- [x] Dataset is complete (3 files, all present, all hashed above).
- [x] Dataset is verified (hashes recorded before provisioning).
- [x] Dataset is authorized (this repo's own synthetic output; no private data;
      no third-party licence; consistent with the standing rule to train only on
      open/own data and never on private data).
- [x] Evaluation partition is excluded from training and named explicitly.
- [x] Budget and kill condition stated.

**Gate satisfied.** Provisioning is permitted for this control run only.

