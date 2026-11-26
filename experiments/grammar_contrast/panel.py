"""Load and verify the fixed BLiMP panel before scoring."""
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_panel(data, pin):
    rows = []
    files = {}
    for path in sorted(data.glob("*.jsonl")):
        raw = path.read_bytes()
        files[path.name] = digest(raw)
        items = [json.loads(line) for line in raw.splitlines()]
        uid = path.stem
        if (len(items) != 1000 or len({item["pairID"] for item in items}) != 1000
                or any(item["UID"] != uid for item in items)):
            raise ValueError(f"Invalid BLiMP file {path}")
        def rank(item):
            return digest(f'nano-grammar-20261120/{uid}/{item["pairID"]}'.encode())
        for item in sorted(items, key=rank)[:32]:
            rows.append([uid, item["pairID"], item["linguistics_term"],
                         item["sentence_good"], item["sentence_bad"]])
    fingerprint = dict(files=files, pairs=len(rows), panel_sha256=digest(json.dumps(rows).encode()))
    if len(files) != 67 or len(rows) != 2144 or fingerprint != json.loads(pin.read_text()):
        raise ValueError("BLiMP source or selected pairs differ from the pinned panel")
    return rows
