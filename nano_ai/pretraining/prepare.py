"""Prepare deterministic uint16 token shards with a cryptographic manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .sources import source_record


MANIFEST_SCHEMA = "nano-pretraining-dataset-manifest/v1"
DEFAULT_TOKENIZER = Path(__file__).resolve().parents[2] / "pretrain" / "tokenizer.json"
EOD_TOKEN_ID = 0


class DatasetPreparationError(RuntimeError):
    """Raised when a corpus cannot satisfy Nano's data authority gate."""


class TextTokenizer(Protocol):
    def encode(self, text: str) -> Any: ...


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _token_ids(tokenizer: TextTokenizer, text: str) -> list[int]:
    encoded = tokenizer.encode(text)
    ids = encoded.ids if hasattr(encoded, "ids") else encoded
    result = [int(token_id) for token_id in ids]
    if any(token_id < 0 or token_id > np.iinfo(np.uint16).max for token_id in result):
        raise DatasetPreparationError("tokenizer produced an ID outside uint16 range")
    return result


class _ShardWriter:
    def __init__(self, directory: Path, split: str, shard_token_limit: int) -> None:
        self.directory = directory
        self.split = split
        self.shard_token_limit = shard_token_limit
        self.buffer: list[int] = []
        self.files: list[dict[str, Any]] = []

    def add(self, token_ids: list[int]) -> None:
        cursor = 0
        while cursor < len(token_ids):
            remaining = self.shard_token_limit - len(self.buffer)
            self.buffer.extend(token_ids[cursor : cursor + remaining])
            cursor += remaining
            if len(self.buffer) == self.shard_token_limit:
                self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        name = f"{self.split}-{len(self.files):05d}.npy"
        destination = self.directory / name
        partial = destination.with_suffix(".npy.partial")
        with partial.open("wb") as handle:
            np.save(handle, np.asarray(self.buffer, dtype=np.uint16), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, destination)
        self.files.append(
            {
                "path": name,
                "bytes": destination.stat().st_size,
                "tokens": len(self.buffer),
                "sha256": _sha256_file(destination),
                "dtype": "uint16",
            }
        )
        self.buffer.clear()


def _prepare_split(
    documents: Iterable[str],
    tokenizer: TextTokenizer,
    writer: _ShardWriter,
    token_budget: int,
    seen_hashes: set[str],
    excluded_hashes: set[str],
) -> dict[str, int]:
    stats = {
        "documents_seen": 0,
        "documents_written": 0,
        "empty_or_invalid_documents": 0,
        "exact_duplicates_removed": 0,
        "contamination_matches_removed": 0,
        "tokens_written": 0,
    }
    for value in documents:
        if stats["tokens_written"] >= token_budget:
            break
        stats["documents_seen"] += 1
        if not isinstance(value, str) or not value.strip():
            stats["empty_or_invalid_documents"] += 1
            continue
        text_bytes = value.encode("utf-8")
        document_hash = hashlib.sha256(text_bytes).hexdigest()
        if document_hash in excluded_hashes:
            stats["contamination_matches_removed"] += 1
            continue
        if document_hash in seen_hashes:
            stats["exact_duplicates_removed"] += 1
            continue
        ids = _token_ids(tokenizer, value) + [EOD_TOKEN_ID]
        remaining = token_budget - stats["tokens_written"]
        ids = ids[:remaining]
        if not ids:
            break
        writer.add(ids)
        seen_hashes.add(document_hash)
        stats["documents_written"] += 1
        stats["tokens_written"] += len(ids)
    writer.flush()
    if stats["tokens_written"] != token_budget:
        raise DatasetPreparationError(
            f"{writer.split} source exhausted at {stats['tokens_written']:,} "
            f"tokens before the {token_budget:,}-token budget"
        )
    return stats


def _read_exclusion_hashes(path: Path | None) -> tuple[set[str], dict[str, Any]]:
    if path is None:
        return set(), {"path": None, "sha256": None, "entries": 0}
    entries = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    invalid = [entry for entry in entries if len(entry) != 64 or entry.lower() != entry]
    if invalid or any(character not in "0123456789abcdef" for entry in entries for character in entry):
        raise DatasetPreparationError("contamination digest file contains a noncanonical SHA-256")
    return entries, {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "entries": len(entries),
    }


def prepare_dataset(
    *,
    dataset_name: str,
    train_documents: Iterable[str],
    validation_documents: Iterable[str],
    tokenizer: TextTokenizer,
    tokenizer_path: Path,
    output_directory: Path,
    train_token_budget: int,
    validation_token_budget: int,
    shard_token_limit: int = 10_000_000,
    contamination_hashes_path: Path | None = None,
) -> dict[str, Any]:
    """Prepare a complete dataset atomically, then verify every output byte."""

    source = source_record(dataset_name)
    if source["authorization_state"] != "approved_for_bounded_local_smoke":
        raise DatasetPreparationError(
            f"{dataset_name} is {source['authorization_state']}; preparation is blocked"
        )
    for label, value in (
        ("train_token_budget", train_token_budget),
        ("validation_token_budget", validation_token_budget),
        ("shard_token_limit", shard_token_limit),
    ):
        if value <= 0:
            raise ValueError(f"{label} must be positive")
    tokenizer_path = tokenizer_path.resolve()
    if not tokenizer_path.is_file():
        raise DatasetPreparationError(f"tokenizer not found: {tokenizer_path}")
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise DatasetPreparationError(
            f"output already exists; verify or choose a new versioned path: {output_directory}"
        )
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_directory.name}.", dir=output_directory.parent))
    try:
        excluded_hashes, exclusion_record = _read_exclusion_hashes(contamination_hashes_path)
        seen_hashes: set[str] = set()
        # Validation is processed first so exact overlap is always removed from train.
        validation_writer = _ShardWriter(staging, "validation", shard_token_limit)
        validation_stats = _prepare_split(
            validation_documents,
            tokenizer,
            validation_writer,
            validation_token_budget,
            seen_hashes,
            excluded_hashes,
        )
        train_writer = _ShardWriter(staging, "train", shard_token_limit)
        train_stats = _prepare_split(
            train_documents,
            tokenizer,
            train_writer,
            train_token_budget,
            seen_hashes,
            excluded_hashes,
        )
        files = validation_writer.files + train_writer.files
        manifest: dict[str, Any] = {
            "schema": MANIFEST_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "source": source,
            "tokenizer": {
                "path": str(tokenizer_path),
                "sha256": _sha256_file(tokenizer_path),
                "procedure": "existing Nano byte-level BPE; no text normalization; append EOD token 0 after each document",
                "vocabulary_size": 4096,
                "eod_token_id": EOD_TOKEN_ID,
                "output_dtype": "uint16",
            },
            "splits": {
                "train": {"token_budget": train_token_budget, **train_stats},
                "validation": {"token_budget": validation_token_budget, **validation_stats},
                "test": {"included": False, "reason": "evaluation artifacts are never training data"},
            },
            "controls": {
                "exact_deduplication": "SHA-256 of exact UTF-8 document bytes, across validation then train",
                "near_deduplication": "not_applied",
                "contamination_exclusion": exclusion_record,
                "private_data_included": False,
                "development_or_evaluation_data_included": False,
                "h6_artifacts_included": False,
            },
            "files": files,
            "file_count": len(files),
            "total_tokens": sum(file["tokens"] for file in files),
            "total_bytes": sum(file["bytes"] for file in files),
            "transfer": {
                "method": "RunPod network-volume S3-compatible API; upload manifest allowlist only",
                "destination_path": (
                    f"/workspace/nano-training-data/{dataset_name}/"
                    f"{source['source_revision']}/{output_directory.name}"
                ),
                "verification": "compare every remote byte size and SHA-256 to this manifest, then mmap each shard and sample a context window",
            },
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_bytes(_canonical_json(manifest))
        (staging / "manifest.sha256").write_text(
            f"{_sha256_file(manifest_path)}  manifest.json\n", encoding="utf-8"
        )
        os.replace(staging, output_directory)
        return verify_prepared_dataset(output_directory)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_prepared_dataset(directory: Path) -> dict[str, Any]:
    """Re-hash and structurally validate a prepared dataset directory."""

    directory = directory.resolve()
    manifest_path = directory / "manifest.json"
    digest_path = directory / "manifest.sha256"
    if not manifest_path.is_file() or not digest_path.is_file():
        raise DatasetPreparationError("manifest.json or manifest.sha256 is missing")
    expected_manifest_hash = digest_path.read_text(encoding="utf-8").split()[0]
    if _sha256_file(manifest_path) != expected_manifest_hash:
        raise DatasetPreparationError("manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") != "complete":
        raise DatasetPreparationError("unsupported or incomplete manifest")
    if manifest.get("file_count") != len(manifest.get("files", [])):
        raise DatasetPreparationError("manifest file count mismatch")
    total_tokens = 0
    total_bytes = 0
    splits_found: set[str] = set()
    for record in manifest["files"]:
        path = directory / record["path"]
        if not path.is_file():
            raise DatasetPreparationError(f"missing shard: {record['path']}")
        if path.stat().st_size != record["bytes"] or _sha256_file(path) != record["sha256"]:
            raise DatasetPreparationError(f"shard verification failed: {record['path']}")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.dtype != np.uint16 or values.ndim != 1 or len(values) != record["tokens"]:
            raise DatasetPreparationError(f"invalid shard structure: {record['path']}")
        split = record["path"].split("-", 1)[0]
        splits_found.add(split)
        total_tokens += len(values)
        total_bytes += path.stat().st_size
    if splits_found != {"train", "validation"}:
        raise DatasetPreparationError("both train and validation shards are required")
    if total_tokens != manifest["total_tokens"] or total_bytes != manifest["total_bytes"]:
        raise DatasetPreparationError("manifest totals do not match verified shards")
    return manifest


def _stream_hugging_face(source: dict[str, Any], split: str, cache_dir: Path) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DatasetPreparationError("install the project's datasets dependency first") from exc
    files = [record["path"] for record in source["source_files"] if record["split"] == split]
    dataset = load_dataset(
        source["dataset_name"],
        data_files={split: files},
        split=split,
        revision=source["source_revision"],
        streaming=True,
        cache_dir=str(cache_dir),
    )
    for row in dataset:
        yield row["text"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare-tinystories")
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    prepare_parser.add_argument("--cache-dir", type=Path, default=Path(".local-data/hf-cache"))
    prepare_parser.add_argument("--train-tokens", type=int, default=1_000_000)
    prepare_parser.add_argument("--validation-tokens", type=int, default=100_000)
    prepare_parser.add_argument("--shard-tokens", type=int, default=1_000_000)
    prepare_parser.add_argument("--contamination-hashes", type=Path)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("directory", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "verify":
        manifest = verify_prepared_dataset(args.directory)
    else:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise DatasetPreparationError("install the project's tokenizers dependency first") from exc
        source = source_record("tinystories")
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        manifest = prepare_dataset(
            dataset_name="tinystories",
            train_documents=_stream_hugging_face(source, "train", args.cache_dir),
            validation_documents=_stream_hugging_face(source, "validation", args.cache_dir),
            tokenizer=tokenizer,
            tokenizer_path=args.tokenizer,
            output_directory=args.output,
            train_token_budget=args.train_tokens,
            validation_token_budget=args.validation_tokens,
            shard_token_limit=args.shard_tokens,
            contamination_hashes_path=args.contamination_hashes,
        )
    print(json.dumps({
        "status": "verified",
        "dataset": manifest["source"]["dataset_name"],
        "file_count": manifest["file_count"],
        "total_tokens": manifest["total_tokens"],
        "total_bytes": manifest["total_bytes"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
