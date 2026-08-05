"""Manifest-gated token-window loader for Nano pretraining."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .prepare import verify_prepared_dataset


class ManifestTokenDataset:
    """Read random next-token windows only from verified split shards."""

    def __init__(self, directory: Path, split: str, sequence_length: int = 512) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.directory = directory.resolve()
        self.manifest = verify_prepared_dataset(self.directory)
        self.sequence_length = sequence_length
        records = [
            record for record in self.manifest["files"] if record["path"].startswith(f"{split}-")
        ]
        self.shards = [
            np.load(self.directory / record["path"], mmap_mode="r", allow_pickle=False)
            for record in records
            if record["tokens"] > sequence_length
        ]
        if not self.shards:
            raise ValueError(f"no {split} shard can provide a {sequence_length}-token window")
        self.window_counts = np.asarray(
            [len(shard) - sequence_length for shard in self.shards], dtype=np.int64
        )
        self.total_windows = int(self.window_counts.sum())

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        flat_index = int(rng.integers(self.total_windows))
        cumulative = 0
        for shard, count in zip(self.shards, self.window_counts, strict=True):
            if flat_index < cumulative + count:
                offset = flat_index - cumulative
                window = np.asarray(
                    shard[offset : offset + self.sequence_length + 1], dtype=np.int64
                )
                return window[:-1], window[1:]
            cumulative += int(count)
        raise AssertionError("sample index escaped verified shard bounds")
