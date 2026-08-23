#!/usr/bin/env python3
"""Pull native checkpoint weights from pod via offset-chunked SSH (uses runpod_pod_ssh.sh)."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MIN_WEIGHT_BYTES = 1_000_000
CHUNK_BYTES = 4 * 1024 * 1024
SSH = ROOT / "scripts" / "runpod_pod_ssh.sh"


def pod_ssh(pod_id: str, remote_cmd: str) -> str:
    proc = subprocess.run(
        ["bash", str(SSH), pod_id, remote_cmd],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ssh failed rc={proc.returncode}: {proc.stderr[-400:]}")
    return proc.stdout.strip()


def remote_size(pod_id: str, remote_path: str) -> int:
    out = pod_ssh(pod_id, f"stat -c%s {remote_path}")
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            return int(line)
    raise RuntimeError(f"could not stat remote file: {out!r}")


def pull_chunk(pod_id: str, remote_path: str, offset: int, length: int) -> bytes:
    cmd = (
        f"python3 -c \"import base64,sys; p={remote_path!r}; "
        f"f=open(p,'rb'); f.seek({offset}); d=f.read({length}); "
        f"print(base64.b64encode(d).decode())\""
    )
    out = pod_ssh(pod_id, cmd)
    b64 = "".join(out.split())
    return base64.b64decode(b64)


def verify_weights(run_id: str, checkpoint_dir: Path = Path("artifacts/native_checkpoints")) -> tuple[bool, str]:
    path = checkpoint_dir / run_id / "latest.pt"
    if not path.is_file():
        return False, f"missing {path}"
    size = path.stat().st_size
    if size < MIN_WEIGHT_BYTES:
        return False, f"too small {size} bytes at {path}"
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not payload.get("model_state"):
            return False, f"invalid checkpoint payload at {path}"
    except Exception as exc:
        return False, f"corrupt checkpoint at {path}: {exc}"
    return True, str(path)


def pull_weights(pod_id: str, run_id: str, *, checkpoint_dir: Path = Path("artifacts/native_checkpoints")) -> Path:
    remote_path = f"/workspace/nano-lm/artifacts/native_checkpoints/{run_id}/latest.pt"
    total = remote_size(pod_id, remote_path)
    if total < MIN_WEIGHT_BYTES:
        raise RuntimeError(f"remote file too small: {total}")
    out = checkpoint_dir / run_id / "latest.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file() and verify_weights(run_id, checkpoint_dir)[0]:
        return out
    if out.is_file():
        out.unlink()
    offset = 0
    with out.open("wb") as handle:
        while offset < total:
            length = min(CHUNK_BYTES, total - offset)
            data = pull_chunk(pod_id, remote_path, offset, length)
            if len(data) != length:
                raise RuntimeError(f"chunk size mismatch at {offset}: got {len(data)} expected {length}")
            handle.write(data)
            offset += length
            print(f"pulled {offset}/{total}", file=sys.stderr, flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull native checkpoint weights from pod")
    parser.add_argument("pod_id")
    parser.add_argument("run_id")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/native_checkpoints"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        ok, msg = verify_weights(args.run_id, args.checkpoint_dir)
        print(json.dumps({"ok": ok, "path": msg}))
        return 0 if ok else 1
    out = pull_weights(args.pod_id, args.run_id, checkpoint_dir=args.checkpoint_dir)
    ok, msg = verify_weights(args.run_id, args.checkpoint_dir)
    print(json.dumps({"pulled": str(out), "bytes": out.stat().st_size, "verified": ok, "verify_msg": msg}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
