"""Regression test for row-level licenses and train/validation leakage."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers

import prepare_code_stream as build
import verify_code_stream as replay


class CodeStreamTest(unittest.TestCase):
    def test_replays_only_accepted_disjoint_complete_documents(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            tokenizer = Tokenizer(models.WordLevel({"?": 0, "<|eos|>": 1, "a": 2, "b": 3}, unk_token="?"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer_path = root / "tokenizer.json"
            tokenizer.save(str(tokenizer_path))

            def row(repo, label, word):
                return dict(repo_name=repo, path="file.txt", license=label, code=(word + " ") * 4000)

            fixture = {
                "val": [row("org/holdout", "mit", "a")],
                "train": [row("org/other", "gpl-3.0", "b"), row("org/other", None, "b"),
                          row("ORG/HOLDOUT", "mit", "b"), row("org/fork", "isc", "a"),
                          row("org/train", "isc", "b")],
            }
            pins = {}
            for split, rows in fixture.items():
                path = root / f"{split}.parquet"
                pq.write_table(pa.Table.from_pylist(rows), path)
                pins[split] = (path.name, build.sha256(path))
            output = root / "output"
            with patch.dict(build.SHARDS, pins, clear=True):
                manifest = build.build(root, tokenizer_path, output, 4000, 4000)
                report = replay.verify(root, tokenizer_path, output)
                train = manifest["splits"]["train"]
                self.assertEqual(train["rejected_by_label"], {"None": 1, "gpl-3.0": 1})
                self.assertEqual((train["validation_repo"], train["duplicate_content"]), (1, 1))
                self.assertEqual(report["train"]["replayed_tokens"], 4001)
                with self.assertRaises(FileExistsError):
                    build.build(root, tokenizer_path, output, 4000, 4000)
                small = root / "small"
                tiny = build.build(root, tokenizer_path, small, 1, 1)
                self.assertEqual(tiny["splits"]["train"]["sampling_threshold_u64"], 2**64)
                self.assertEqual(replay.verify(root, tokenizer_path, small)["train"]["replayed_tokens"], 4001)
                with (output / "train_code.bin").open("ab") as stream:
                    stream.write(b"\x00\x00")
                with self.assertRaisesRegex(ValueError, "digest mismatch"):
                    replay.verify(root, tokenizer_path, output)
