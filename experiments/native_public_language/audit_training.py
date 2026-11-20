"""Recount source license labels and verify selected complete records in the archived code stream."""
import argparse
from collections import Counter, defaultdict
import json
import mmap
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer

from evaluate import EOS, aligned_match, verify_file
from corpus import HERE


def audit(data, shard):
    inputs = json.loads((HERE / "inputs.json").read_text())
    tok_path = data / "tokenizer_32k.json"
    verify_file(tok_path, inputs["tokenizer_sha256"])
    verify_file(data / "train_code.bin", "f67186454917c8f6fedd8424aee7a34ca2f3c0f75c1801dbd249631da8ab0078")
    verify_file(shard, "c05e4cad93aac3f978a2385f8c055a10d247b107afd8bf9cb8c762f8403bdc0f")
    tk = Tokenizer.from_file(str(tok_path))
    table = pq.ParquetFile(shard).read_row_group(0, columns=["code", "license"]).to_pydict()
    eligible = Counter()
    selections = defaultdict(list)
    for text, license_id in zip(table["code"], table["license"]):
        if text and 200 < len(text) < 100_000:
            eligible[license_id] += 1
            if len(selections[license_id]) < 6:
                selections[license_id].append(text)
    matches = {}
    with (data / "train_code.bin").open("rb") as raw, mmap.mmap(raw.fileno(), 0, access=mmap.ACCESS_READ) as stream:
        for license_id, texts in selections.items():
            hits = sum(aligned_match(stream, tk.encode(text).ids + [EOS]) for text in texts)
            matches[license_id] = dict(checked=len(texts), entire_source_and_eos_present=hits)
    return dict(shard=shard.name, first_row_group_rows=len(table["license"]),
                source_license_counts=dict(Counter(table["license"])),
                after_training_length_filter=dict(eligible), first_six_eligible_per_license=matches)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--shard", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(audit(args.data, args.shard), indent=2, sort_keys=True))
