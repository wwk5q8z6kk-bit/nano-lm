#!/usr/bin/env python3
"""Pull native checkpoint weights from a RunPod pod via chunked base64 over SSH."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MIN_WEIGHT_BYTES = 1_000_000
CHUNK_RE = re.compile(r"^WEIGHT_CHUNK_BEGIN (\d+) (\d+)$")
DONE_RE = re.compile(r"^WEIGHT_DONE (\d+)$")


def pod_host_id(pod_id: str) -> str:
    import json as _json

    cfg_path = os.path.expanduser("~/.runpod/config.toml")
    if not os.path.isfile(cfg_path):
        return ""
    cfg = open(cfg_path).read()
    m = re.search(r"apikey\s*=\s*'([^']+)'", cfg)
    if not m:
        return ""
    api = m.group(1)
    query = _json.dumps(
        {"query": f'query {{ pod(input: {{podId: "{pod_id}"}}) {{ machine {{ podHostId }} }} }}'}
    )
    out = subprocess.check_output(
        [
            "curl",
            "-s",
            "-X",
            "POST",
            "https://api.runpod.io/graphql",
            "-H",
            "Content-Type: application/json",
            "-H",
            f"Authorization: {api}",
            "-d",
            query,
        ],
        text=True,
    )
    pod = (_json.loads(out).get("data") or {}).get("pod") or {}
    return pod.get("machine", {}).get("podHostId", "") or ""


def remote_pull_cmd(run_id: str) -> str:
    remote_path = f"/workspace/nano-lm/artifacts/native_checkpoints/{run_id}/latest.pt"
    return f"""python3 <<'PY'
import base64, os, sys
path = {remote_path!r}
if not os.path.isfile(path):
    print('WEIGHT_MISSING', path, file=sys.stderr)
    sys.exit(2)
size = os.path.getsize(path)
chunk = 3 * 1024 * 1024
with open(path, 'rb') as f:
    offset = 0
    while True:
        data = f.read(chunk)
        if not data:
            break
        print(f'WEIGHT_CHUNK_BEGIN {{offset}} {{len(data)}}')
        print(base64.b64encode(data).decode())
        print('WEIGHT_CHUNK_END')
        offset += len(data)
print(f'WEIGHT_DONE {{size}}')
PY"""


def ssh_output(pod_id: str, remote_cmd: str) -> str:
    ssh_key = os.environ.get("RUNPOD_SSH_KEY", os.path.expanduser("~/.runpod/ssh/runpodctl-ssh-key"))
    host_id = pod_host_id(pod_id)
    if not host_id:
        raise RuntimeError(f"no podHostId for {pod_id}")
    if not os.path.isfile(ssh_key):
        raise RuntimeError(f"missing SSH key at {ssh_key}")
    proc = subprocess.run(
        [
            "ssh",
            "-tt",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=60",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-i",
            ssh_key,
            f"{host_id}@ssh.runpod.io",
            f"bash -lc {json.dumps(remote_cmd)}; exit",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ssh failed rc={proc.returncode}: {proc.stderr[-500:]}")
    return proc.stdout


def parse_chunks(stdout: str) -> bytes:
    lines = stdout.splitlines()
    chunks: dict[int, bytes] = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = CHUNK_RE.match(line)
        if m:
            offset = int(m.group(1))
            length = int(m.group(2))
            i += 1
            payload_lines: list[str] = []
            while i < len(lines) and lines[i].strip() != "WEIGHT_CHUNK_END":
                payload_lines.append(lines[i].strip())
                i += 1
            data = base64.b64decode("".join(payload_lines))
            if len(data) != length:
                raise RuntimeError(f"chunk length mismatch at offset {offset}")
            chunks[offset] = data
        i += 1
    if not chunks:
        raise RuntimeError("no weight chunks received")
    ordered = b"".join(chunks[k] for k in sorted(chunks))
    return ordered


def verify_weights(run_id: str, checkpoint_dir: Path = Path("artifacts/native_checkpoints")) -> tuple[bool, str]:
    path = checkpoint_dir / run_id / "latest.pt"
    if not path.is_file():
        return False, f"missing {path}"
    size = path.stat().st_size
    if size < MIN_WEIGHT_BYTES:
        return False, f"too small {size} bytes at {path}"
    return True, str(path)


def pull_weights(
    pod_id: str,
    run_id: str,
    *,
    checkpoint_dir: Path = Path("artifacts/native_checkpoints"),
) -> Path:
    stdout = ssh_output(pod_id, remote_pull_cmd(run_id))
    data = parse_chunks(stdout)
    out = checkpoint_dir / run_id / "latest.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
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
    print(
        json.dumps(
            {
                "pulled": str(out),
                "bytes": out.stat().st_size,
                "verified": ok,
                "verify_msg": msg,
            }
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
