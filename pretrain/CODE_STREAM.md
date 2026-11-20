# Preparing the native code stream

The original mixed-corpus builder selected the code column without inspecting each record’s license label. This builder screens that field, reserves validation repositories before accepting training records, and preserves enough information to reproduce every token. It leaves the historical data and checkpoints available for their original comparisons.

The November 20 build used two shards of [github-code-clean at the pinned revision](https://huggingface.co/datasets/codeparrot/github-code-clean/tree/c48d40f9e70f0196f8236901ee35807f7d6c44c0/data). It used the archived 32K NanoLM tokenizer to compare native architectures under the same vocabulary. The [aggregate manifest](code_stream_manifest.json) includes the input and output digests, accepted labels, counts and sampling thresholds.

| Split | Documents | Repositories | Tokens |
|---|---:|---:|---:|
| Train | 7,041 | 6,582 | 12,000,300 |
| Validation | 335 | 329 | 516,368 |

The upstream files group records by license. The reader samples source rows by a deterministic hash before filtering to spread each split across its shard. It accepts complete documents with an exact allowlisted row label, excludes duplicate file contents and removes training files from validation repositories. Stream lengths can exceed the target by the last complete document. Local indexes map each token range to its original shard row, repository, path, label and source hash.

These are repository-disjoint splits for new experiments. Historical training and tokenizer construction used the same upstream collection. Exact source hashes do not detect similar files or copied fragments, and an upstream repository label does not verify individual file notices. A file-level rights and notice review is still needed before distributing source content. Only aggregate counts and digests are included here.

## Replay

Use Python 3.11 or later, NumPy, PyArrow and Hugging Face Tokenizers. Set `DATA` to a directory containing the two pinned Parquet files and `TOKENIZER` to the tokenizer JSON. The published manifest identifies the archived tokenizer by SHA-256. A different tokenizer produces different artifacts. Use a fresh output directory.

```sh
python pretrain/prepare_code_stream.py --data "$DATA" --tokenizer "$TOKENIZER" --output "$OUT"
python pretrain/verify_code_stream.py --data "$DATA" --tokenizer "$TOKENIZER" --output "$OUT"
python -m pytest -q pretrain/test_code_stream.py
```

The verifier re-encodes all 7,376 records for the recorded run, checks each token and its original source, and tests that the accepted repositories and exact file contents do not cross splits.
