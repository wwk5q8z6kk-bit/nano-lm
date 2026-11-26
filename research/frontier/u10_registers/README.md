# Does wider recurrent state improve next-token prediction?

This two-seed experiment compares three roughly five-million-parameter language models trained from scratch. M21 and M22 come from the Antigravity unified-loop architecture branch. Both combine a rotation-based recurrent mixer, concept-slot memory and a sparse causal copy pointer. M22 also learns how to combine residuals across layers. Its wide variant increases each head from four to eleven rotation registers and reduces the feed-forward width from 460 to 357 to keep the parameter count nearly fixed.

The prior synthetic probe increased numerical rank by widening the mixer, which left open whether it could use that capacity on text. Association and copying can vary across sequence mixers, motivating separate language-model and recall measurements. The protocol was fixed before the first update. [Study protocol](protocol.md) · [Zoology: Measuring and Improving Recall in Efficient Language Models](https://arxiv.org/abs/2312.04927)

## Measurement

Each model receives the same 8,000 training windows per seed and 1,024,000 next-token predictions. Two seeds give six cells, 6,144,000 training tokens total. Inputs come from an 800M-token historical training stream. Development validation uses separate source shards and 64 fixed windows per domain across three context lengths. At 512 tokens the table uses 16 windows per domain. BPB divides mean token cross-entropy by the original per-domain average bytes/token and log(2). A lower value indicates better prediction. The centered mixer rank is the exponential entropy of squared singular values after time-centering and row normalization. It is measured per head at the fourth layer and averaged.

## Results

M21 had the lowest four-domain mean validation BPB on both seeds. The wider M22 improved over the original M22 by 0.00004 and 0.00387 BPB, while trailing M21 by 0.00532 and 0.00350 BPB. **The prespecified decision is to hold the wider model.** Its centered rank on code reached 5.79 and 7.36 against a target of 12. Its copy-256 accuracies were 96.81% and 81.54% against a target of 95% on each seed.

| Model | Seed 0 mean BPB | Seed 1 mean BPB | Web | Textbooks | Math | Code |
|---|---:|---:|---:|---:|---:|---:|
| M21 | 1.83329 | 1.83366 | 1.84967 | 1.77812 | 2.10777 | 1.59834 |
| M22 | 1.83865 | 1.84103 | 1.85630 | 1.78886 | 2.11110 | 1.60309 |
| Wide M22 | 1.83860 | 1.83716 | 1.85513 | 1.78338 | 2.10745 | 1.60557 |

The four domain columns average the two seeds at context length 512. CPU replay of all six checkpoints independently recomputed all 72 domain-by-length scores to within 0.000006 BPB. All fits ran locally at zero cloud cost. Their timing reflects a shared host. [Full scores and decision](results/summary.json) · [CPU checkpoint replay](results/cpu/)

One 32-token sequence from a sampled training window matched a seed-0 math validation window. Removing the entire window preserves the order of all three arms. The scan found zero 128-token matches. Two seeds, roughly one million training tokens per fit, domain-average byte conversions and reuse of historical validation windows limit extrapolation to other corpora, training budgets or long-context systems. [Overlap counts](results/sampled-overlap.json) · [Sensitivity analysis](results/overlap-sensitivity.json)

## Reproduce

From this directory, check the original decision without external data, packages or credentials.

```sh
python3 score.py --check
```

For a training replay, use a Python environment with NumPy and PyTorch, and provide a local copy of the corpus. The reference software was Python 3.14.7, NumPy 2.5.1 and PyTorch 2.11.0 on little-endian ARM64 using MPS. The runner verifies the tokenizer, configuration and all token streams before training. A new output directory stores metrics and local CPU-loadable checkpoint tensors. Different backends can differ numerically.

```sh
python run.py --data "$CORPUS" --device mps --output replay
python score.py --results replay
python replay_checkpoint.py --data "$CORPUS" --receipt replay/seed-0-m21.json \
  --weights replay/seed-0-m21.pt --out cpu-check.json
python audit_overlap.py --data "$CORPUS"
```

The original run used the model module with SHA-256 `5f480eb97343571940967b771645ce74a852a4c5d94e9153267916c76721c744` from local unified-loop revision `90f3e18802e4896f7d67a2680e023d45a6f4184c`, descended from `d178771125351e8be6dc02d076ac1950f0ef6803`. The run protocol and executable were captured at `f663fd9be110929a6a5aae21d3ec794eab434671`. Only the used M21/M22 modules are included here. Source comparison checks all initialized parameter tensors and compares forward results and gradients on a seeded two-row test. Changing future tokens and the other batch row leaves the checked prefix unchanged. Users with the original module can run `python verify_source.py path/to/zoo.py`. The public runner allows corpus reconstruction time to vary. The recorded training script checksums identify the original versions. [Source-equivalence checks](results/source-equivalence.json)

## Corpus and source terms

The corpus has 62% web, 13% textbooks, 13% mathematics and 12% code training tokens. Validation has one million tokens in each domain. The builder uses shard 0 for training and shard 1 for validation in each pinned source. It rejects training documents with a matching subsampled 13-word phrase from validation, then concatenates and interleaves token blocks. Near-duplicate documents and short shared expressions can survive. The manifest records exact revisions, filenames, SHA-256 checksums, domain tallies and the original tokenizer identity. [Corpus manifest](corpus.json)

- Web uses the 10BT sample of FineWeb-Edu, released under ODC-By. [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- Textbooks use the Cosmopedia-v2 component of SmolLM-Corpus, labeled ODC-By. [SmolLM-Corpus dataset card](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)
- Mathematics uses the 4+ subset of FineMath, labeled ODC-By. [FineMath dataset card](https://huggingface.co/datasets/HuggingFaceTB/finemath)
- CodeParrot labels its dataset Apache-2.0 and includes repositories with a range of underlying licenses. The historical builder read the code column without filtering repository licenses. Individual source terms apply. This capsule publishes source identities, algorithms and aggregate measurements. [GitHub Code Clean dataset card](https://huggingface.co/datasets/codeparrot/github-code-clean)

For local reconstruction from the public sources, obtain access under the applicable source terms and install `pyarrow`, `huggingface_hub` and `tokenizers` (tested versions 25.0.0, 1.26.0 and 0.22.2). The reconstruction script derives from the repository's MIT-licensed builder at `3314170a8c9ca4e19445496fc27d40a631ac0063`. Give it an empty output directory. It verifies the pinned upstream parquet digests and checks the final token streams. Cached-only operation is the default. Add `--download` to fetch missing pinned objects directly, or `--tokenizer /path/to/tokenizer_32k.json` to supply an original tokenizer. Exact replay requires the digest in the manifest.

```sh
python rebuild_corpus.py --cache "$HF_HUB_CACHE" --out "$CORPUS" --download
```

For this release, fresh construction from the source shards reproduced the tokenizer byte for byte and all four one-million-token validation streams. Training used the archived 800M-token stream after a SHA-256 check. The full training corpus rebuild is available in the replay script. [Reconstruction checks](results/corpus-reconstruction.json)
