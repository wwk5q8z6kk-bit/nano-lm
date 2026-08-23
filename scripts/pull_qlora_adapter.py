#!/usr/bin/env python3
"""Pull QLoRA adapter directory from RunPod pod via SCP (tar) or SSH fallback."""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pull_native_weights import CHUNK_BYTES, resolve_direct_scp

SSH = ROOT / "scripts" / "runpod_pod_ssh.sh"
REMOTE_DIR = "/workspace/qlora_canary_adapter"
REMOTE_TAR = "/workspace/qlora_canary_adapter.tar.gz"
MIN_ADAPTER_BYTES = 1_000_000
MAX_PULL_ATTEMPTS = 5


def pod_ssh(pod_id: str, remote_cmd: str) -> str:
    proc = subprocess.run(
        ["bash", str(SSH), pod_id, remote_cmd],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ssh failed rc={proc.returncode}: {proc.stderr[-500:]}")
    return proc.stdout.strip()


def remote_adapter_ready(pod_id: str) -> tuple[bool, int]:
    cmd = (
        f"test -f {REMOTE_DIR}/adapter_config.json && "
        f"du -sb {REMOTE_DIR} 2>/dev/null | awk '{{print $1}}' || echo 0"
    )
    out = pod_ssh(pod_id, cmd)
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            size = int(line)
            return size >= MIN_ADAPTER_BYTES, size
    return False, 0


def verify_adapter(local_dir: Path) -> tuple[bool, str, int]:
    if not local_dir.is_dir():
        return False, f"missing dir {local_dir}", 0
    config = local_dir / "adapter_config.json"
    if not config.is_file():
        return False, f"missing {config}", 0
    weights = list(local_dir.glob("adapter_model*.safetensors")) + list(local_dir.glob("adapter_model*.bin"))
    if not weights:
        return False, "missing adapter weight file", 0
    total = sum(p.stat().st_size for p in local_dir.rglob("*") if p.is_file())
    if total < MIN_ADAPTER_BYTES:
        return False, f"adapter too small ({total} bytes)", total
    try:
        from safetensors import safe_open

        st = next((p for p in weights if p.suffix == ".safetensors"), None)
        if st:
            with safe_open(str(st), framework="pt") as handle:
                keys = list(handle.keys())
            if not keys:
                return False, "empty safetensors", total
    except Exception as exc:
        return False, f"safetensors check failed: {exc}", total
    return True, str(local_dir), total


def pull_via_scp_tar(pod_id: str, local_dir: Path) -> bool:
    endpoint = resolve_direct_scp(pod_id)
    if endpoint is None:
        return False
    host, port, ssh_key = endpoint
    ready, remote_bytes = remote_adapter_ready(pod_id)
    if not ready:
        print(f"remote adapter not ready ({remote_bytes} bytes)", file=sys.stderr)
        return False
    pod_ssh(pod_id, f"tar -czf {REMOTE_TAR} -C /workspace qlora_canary_adapter")
    local_tar = local_dir.parent / "qlora_canary_adapter.tar.gz"
    if local_tar.is_file():
        local_tar.unlink()
    proc = subprocess.run(
        [
            "scp",
            "-i",
            str(ssh_key),
            "-P",
            str(port),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=60",
            "-o",
            "StrictHostKeyChecking=accept-new",
            f"root@{host}:{REMOTE_TAR}",
            str(local_tar),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(f"scp failed: {proc.stderr[-400:]}", file=sys.stderr)
        return False
    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path=local_dir.parent)
    ok, msg, total = verify_adapter(local_dir)
    if not ok:
        print(f"scp verify failed: {msg}", file=sys.stderr)
        return False
    print(f"scp pull ok host={host} port={port} bytes={total}", file=sys.stderr)
    return True


def pull_via_ssh_tar_chunks(pod_id: str, local_dir: Path) -> None:
    ready, remote_bytes = remote_adapter_ready(pod_id)
    if not ready:
        raise RuntimeError(f"remote adapter not ready ({remote_bytes} bytes)")
    pod_ssh(pod_id, f"tar -czf {REMOTE_TAR} -C /workspace qlora_canary_adapter")
    size = int(pod_ssh(pod_id, f"stat -c%s {REMOTE_TAR}"))
    local_tar = local_dir.parent / "qlora_canary_adapter.tar.gz"
    if local_tar.is_file():
        local_tar.unlink()
    offset = 0
    with local_tar.open("wb") as handle:
        while offset < size:
            length = min(CHUNK_BYTES, size - offset)
            cmd = (
                f"python3 -c \"import base64; f=open('{REMOTE_TAR}','rb'); "
                f"f.seek({offset}); print(base64.b64encode(f.read({length})).decode())\""
            )
            out = pod_ssh(pod_id, cmd)
            data = base64.b64decode("".join(out.split()))
            handle.write(data)
            offset += length
            print(f"pulled {offset}/{size}", file=sys.stderr, flush=True)
    if local_dir.exists():
        shutil.rmtree(local_dir)
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path=local_dir.parent)


def pull_adapter(pod_id: str, local_dir: Path) -> tuple[bool, str, int]:
    ok, msg, total = verify_adapter(local_dir)
    if ok:
        return True, msg, total

    for attempt in range(1, MAX_PULL_ATTEMPTS + 1):
        print(f"pull attempt {attempt}/{MAX_PULL_ATTEMPTS}", file=sys.stderr)
        if pull_via_scp_tar(pod_id, local_dir):
            return verify_adapter(local_dir)
        time.sleep(10)

    print("scp unavailable or failed; falling back to ssh base64 chunks", file=sys.stderr)
    pull_via_ssh_tar_chunks(pod_id, local_dir)
    return verify_adapter(local_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pod_id")
    parser.add_argument("--dest", type=Path, default=Path("artifacts/campaign/student_qlora_canary_adapter"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    dest = args.dest
    if args.verify_only:
        ok, msg, total = verify_adapter(dest)
        print(json.dumps({"ok": ok, "path": msg, "bytes": total, "verified": ok}))
        return 0 if ok else 1

    ok, msg, total = pull_adapter(args.pod_id, dest)
    print(json.dumps({"ok": ok, "path": msg, "bytes": total, "verified": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
