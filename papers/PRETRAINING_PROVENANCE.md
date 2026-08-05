# Pretraining data provenance — retroactive record

**Status:** Honest-gap record. Written 2026-08-04 as part of the training-data
authority work (see `papers/TRAINING_DATA_REGISTRY.md`). Nothing here changes any
manuscript; this file states exactly what is and is not reconstructable about the
two historical pretraining acquisitions.

## 1. What is attested (and by what)

**Acquisition A — nano 3.15M anchor (2026-07-15).** Source: `pretrain/AUDIT.md`
(prose audit) and `pretrain/train.log` (run log).

- Corpus: `HuggingFaceFW/fineweb`, config `sample-10BT`, **streamed** — ~12,000
  docs / 36.9M chars fetched in ~8s.
- Filtering: Gopher/C4 heuristics, 96% kept. Dedup: exact SHA-1 + MinHash
  (5-gram, 112 hashes, 14×8 bands), 0 duplicates found (FineWeb is pre-deduped).
  Decontamination: 13-gram + canaries method specified, executed as a documented
  no-op (no eval suite existed at nano scale).
- Tokenizer: byte-level BPE, V=4096, trained in-run; sharded to uint16 binary;
  **10.96M tokens** total.
- Training consumption: 4,000 steps / **32.8M tokens (~3.1 epochs)** / 20.4 min
  on Apple Silicon MPS; loss 8.35 → 3.70 train / 3.96 val; checkpoints at
  1k/2k/3k/4k.

**Acquisition B — 10M scale anchor (2026-07-17).** Source: `scale/AUDIT.md`.

- Corpus: FineWeb `sample-10BT` streamed again, **~240k docs**, ~200M tokens
  (D≈20N), one epoch-ish, reusing the nano 4096-BPE tokenizer.

**Downstream dependency.** `papers/paper1_draft.md` §Model-families cites both:
"pretrained on **32.8M** FineWeb tokens over ~3.1 epochs of a 10.96M-token
shard" and "**~200M** tokens, D≈20N". The `PAPER_ALPHA_CORRECTION_NOTE`
(2026-07-31, since consolidated) corrected the nano budget from ~200M to 32.8M —
that correction stands.

## 2. What is missing (and therefore not reconstructable)

Neither acquisition recorded:

- the dataset **revision/commit SHA** of `HuggingFaceFW/fineweb` at fetch time;
- the **stream seed or document ordering** (both runs *streamed*; "12,000 docs"
  and "~240k docs" are not reproducible without the iteration order);
- any **shard file list or per-file/corpus hash** — `pretrain/train.py` loads
  `shard_000.npy`, and no `*.npy`/`*.bin` shard exists anywhere in the repo;
- a **license record** (FineWeb is ODC-BY 1.0; never recorded at the time);
- an acquisition **script** — the pipeline exists only as prose in the audits.

## 3. Consequence (claim scoping)

The two pretraining runs are **historically attested but not re-runnable**. The
run logs and audits support the token-budget and loss-trajectory claims as a
record of *what was done*; they do not constitute a reproducible data pipeline,
and no byte-identical re-creation of either pretraining corpus is possible.
Paper α's architecture/budget statements remain accurate as historical record.
Any future claim that requires re-creating an anchor's pretraining corpus is
out of reach and must not be made.

The frozen v0.1 release checkpoints (the two anchors) are unaffected: they are
content-addressed artifacts, and every H-cycle result binds to checkpoint
hashes, not to the vanished pretraining shards.

## 4. Rule going forward

All future corpus acquisition goes through `nano_ai/pretraining/`
(`sources.py` pinned revisions + per-file sha256 inventories; `prepare.py`
manifest-verified staging with dedup/contamination digests; `dataset.py`
refuses unverified manifests). No streamed-and-discarded corpora. The
TinyStories smoke set prepared 2026-08-04 is the first acquisition under this
rule; `fineweb-edu` remains proposal-only pending its blocking gates (see
`papers/DATA_LICENSES.md` and `nano_ai/pretraining/sources.py`).
