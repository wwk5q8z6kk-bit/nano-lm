"""Replay source rows and verify the token streams and their split boundaries."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from prepare_code_stream import ALLOWED, DATASET, REVISION, SHARDS, sha256


def verify(data, tokenizer_path, output):
    manifest = json.loads((output / "manifest.json").read_text())
    if (manifest["dataset"], manifest["revision"], manifest["tokenizer_sha256"]) != (
        DATASET, REVISION, sha256(tokenizer_path)
    ):
        raise ValueError("Input identity mismatch")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos = tokenizer.token_to_id("<|eos|>")
    digests = set()
    repos = {}
    report = {}
    for split, (name, source_hash) in SHARDS.items():
        metrics = manifest["splits"][split]
        source = data / name
        bin_path = output / f"{split}_code.bin"
        index = output / f"{split}_code.jsonl"
        if (sha256(source), sha256(bin_path), sha256(index)) != (
            source_hash, metrics["tokens_sha256"], metrics["records_sha256"]
        ):
            raise ValueError(f"{split} artifact digest mismatch")
        tokens = np.memmap(bin_path, mode="r", dtype="<u2")
        offset = 0
        rows = 0
        repos[split] = set()
        row_number = 0
        columns = ["repo_name", "path", "code", "license"]
        with index.open() as handle:
            records = (json.loads(line) for line in handle)
            expected = next(records, None)
            for batch in pq.ParquetFile(source).iter_batches(columns=columns, batch_size=128):
                for row in batch.to_pylist():
                    if expected is None:
                        break
                    if row_number == expected["row"]:
                        record = expected
                        label = row["license"]
                        repo = row["repo_name"].strip().casefold()
                        digest = hashlib.sha256(row["code"].encode()).hexdigest()
                        if label not in ALLOWED or digest in digests or (
                            record["license"], record["repo"], record["path"], record["source_sha256"]
                        ) != (label, repo, row["path"], digest):
                            raise ValueError(f"Invalid or duplicate source in {split} row {row_number}")
                        ids = tokenizer.encode(row["code"], add_special_tokens=False).ids + [eos]
                        if (record["token_offset"], record["tokens"]) != (offset, len(ids)) or not np.array_equal(
                            tokens[offset:offset + len(ids)], ids
                        ):
                            raise ValueError(f"Token mismatch in {split} row {row_number}")
                        digests.add(digest)
                        repos[split].add(repo)
                        offset += len(ids)
                        rows += 1
                        expected = next(records, None)
                    row_number += 1
                if expected is None:
                    break
        if expected is not None or (offset, rows, len(repos[split])) != (
            metrics["tokens"], metrics["accepted"], metrics["repositories"]
        ) or offset != len(tokens) or offset < metrics["target"]:
            raise ValueError(f"Incomplete {split} replay")
        report[split] = dict(replayed_documents=rows, replayed_tokens=offset, repositories=len(repos[split]))
    if repos["train"] & repos["val"]:
        raise ValueError("Repository appears in both splits")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["data", "tokenizer", "output"]:
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.data, args.tokenizer, args.output), indent=2))
