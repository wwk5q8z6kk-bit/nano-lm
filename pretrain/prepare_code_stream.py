"""Build repository-disjoint native-LM streams from a pinned code dataset."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

DATASET = "codeparrot/github-code-clean"
REVISION = "c48d40f9e70f0196f8236901ee35807f7d6c44c0"
SHARDS = {
    "val": ("train-00001-of-00880.parquet", "2e1b2e8bc5823fd68a80a036f1a5db894aa89bd491992bda2dfd96c41187025f"),
    "train": ("train-00000-of-00880.parquet", "c05e4cad93aac3f978a2385f8c055a10d247b107afd8bf9cb8c762f8403bdc0f"),
}
ALLOWED = frozenset({"0bsd", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "cc0-1.0", "isc", "mit"})


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_split(shard, tokenizer, eos, split, target, directory, val_repos, seen):
    counts = Counter()
    licenses = Counter()
    skipped_licenses = Counter()
    repos = set()
    bin_path = directory / f"{split}_code.bin"
    rows_path = directory / f"{split}_code.jsonl"
    with bin_path.open("xb") as stream, rows_path.open("x", encoding="utf-8") as rows:
        parquet = pq.ParquetFile(shard)
        # The upstream shards group records by license. Spread candidates across
        # the shard using their source row numbers before any content filtering.
        threshold = min(2**64, (target * 2**64) // max(1, parquet.metadata.num_rows * 475))
        for batch in parquet.iter_batches(batch_size=128, columns=["repo_name", "path", "license", "code"]):
            for row in batch.to_pylist():
                row_index = counts["scanned"]
                counts["scanned"] += 1
                rank = hashlib.blake2b(f"nanolm-v1/{split}/{row_index}".encode(), digest_size=8).digest()
                if int.from_bytes(rank, "big") >= threshold:
                    counts["sampling_skipped"] += 1
                    continue
                label = row["license"]
                if label not in ALLOWED:
                    skipped_licenses[str(label)] += 1
                    continue
                repo = (row["repo_name"] or "").strip().casefold()
                if not repo or not row["path"]:
                    counts["missing_source"] += 1
                    continue
                if split == "train" and repo in val_repos:
                    counts["validation_repo"] += 1
                    continue
                code = row["code"]
                if not code or not 200 < len(code) < 100_000:
                    counts["length"] += 1
                    continue
                digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
                if digest in seen:
                    counts["duplicate_content"] += 1
                    continue
                ids = tokenizer.encode(code, add_special_tokens=False).ids
                tokens = np.asarray([*ids, eos], dtype="<u2")
                offset = counts["tokens"]
                stream.write(tokens.tobytes())
                rows.write(json.dumps(dict(row=row_index, repo=repo, path=row["path"], source_sha256=digest,
                                           license=label, token_offset=offset, tokens=len(tokens))) + "\n")
                counts["tokens"] += len(tokens)
                counts["accepted"] += 1
                seen.add(digest)
                repos.add(repo)
                licenses[label] += 1
                if counts["tokens"] >= target:
                    break
            if counts["tokens"] >= target:
                break
    if counts["tokens"] < target:
        raise ValueError(f"{split} source exhausted at {counts['tokens']} tokens, target {target}")
    return repos, dict(target=target, sampling_threshold_u64=threshold, **counts, repositories=len(repos), accepted_by_label=dict(licenses),
                       rejected_by_label=dict(skipped_licenses), tokens_sha256=sha256(bin_path),
                       records_sha256=sha256(rows_path))


def build(data, tokenizer_path, output, train_tokens, val_tokens):
    if min(train_tokens, val_tokens) <= 0:
        raise ValueError("Both token targets must be positive")
    if output.exists():
        raise FileExistsError(output)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos = tokenizer.token_to_id("<|eos|>")
    if eos is None or max(tokenizer.get_vocab().values()) > np.iinfo(np.uint16).max:
        raise ValueError("Tokenizer must supply <|eos|> and fit in uint16")
    manifest = dict(dataset=DATASET, revision=REVISION, allowed_row_labels=sorted(ALLOWED),
                    tokenizer_sha256=sha256(tokenizer_path), eos_id=eos, dtype="uint16-le", splits={})
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        staging = Path(temp) / "streams"
        staging.mkdir()
        seen = set()
        val_repos = set()
        for split, (name, digest) in SHARDS.items():
            path = data / name
            if sha256(path) != digest:
                raise ValueError(f"Unexpected SHA-256 for {name}")
            target = val_tokens if split == "val" else train_tokens
            repos, metrics = build_split(path, tokenizer, eos, split, target, staging, val_repos, seen)
            manifest["splits"][split] = dict(shard=name, shard_sha256=digest, **metrics)
            if split == "val":
                val_repos = repos
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        staging.rename(output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="directory with both pinned Parquet files")
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-tokens", type=int, default=12_000_000)
    parser.add_argument("--val-tokens", type=int, default=500_000)
    args = parser.parse_args()
    print(json.dumps(build(args.data, args.tokenizer, args.output, args.train_tokens, args.val_tokens), indent=2))
