"""Save step metrics streamed by finish_native_pod for the operator-selected run."""
import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

from pull_native_weights import checkpoint_run_dir


def save_metrics(stream, run_id, checkpoint_dir):
    target = checkpoint_run_dir(checkpoint_dir, run_id)
    records = {}
    name = None
    lines = []

    def finish_record():
        if name is None:
            return
        data = json.loads("".join(lines))
        if data["config"]["run_id"] != run_id:
            raise ValueError("unexpected run ID in step metrics")
        records[name] = data

    for line in stream:
        if line.startswith("===FILE=== "):
            finish_record()
            name = line.removeprefix("===FILE=== ").rstrip("\r\n")
            if not re.fullmatch(r"step_[0-9]{6,12}\.json", name) or name in records:
                raise ValueError("invalid or duplicate native step filename")
            lines = []
        elif name is not None:
            lines.append(line)
        elif line.strip():
            raise ValueError("unexpected data before step header")
    finish_record()
    if not records:
        raise ValueError("no step metrics received")
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target) as stage:
        for name, data in records.items():
            (Path(stage) / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        for name in records:
            (Path(stage) / name).replace(target / name)
            print("wrote", target / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    save_metrics(sys.stdin, args.run_id, Path("artifacts/native_checkpoints"))
