#!/usr/bin/env python3
"""Pull QLoRA adapter directory from RunPod pod via SCP (tar) or SSH fallback."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pull_native_weights import resolve_direct_scp

SSH = ROOT / "scripts" / "runpod_pod_ssh.sh"
REMOTE_DIR = "/workspace/qlora_canary_adapter"
REMOTE_TAR = "/workspace/qlora_canary_adapter.tar.gz"
REQUIRED = ("adapter_config.json",)


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
    if total < 1_000_000:
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
        return False
    if local_dir.exists():
        import shutil

        shutil.rmtree(local_dir)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path=local_dir.parent)
    return local_dir.is_dir()


def pull_via_ssh_tar_chunks(pod_id: str, local_dir: Path) -> None:
    pod_ssh(pod_id, f"tar -czf {REMOTE_TAR} -C /workspace qlora_canary_adapter")
    size = int(pod_ssh(pod_id, f"stat -c%s {REMOTE_TAR}"))
    chunk = 3 * 1024 * 1024
    local_tar = local_dir.parent / "qlora_canary_adapter.tar.gz"
    if local_tar.is_file():
        local_tar.unlink()
    offset = 0
    with local_tar.open("wb") as handle:
        while offset < size:
            length = min(chunk, size - offset)
            cmd = (
                f"python3 -c \"import base64; f=open('{REMOTE_TAR}','rb'); "
                f"f.seek({offset}); print(base64.b64encode(f.read({length})).decode())\""
            )
            out = pod_ssh(pod_id, cmd)
            data = base64.b64decode("".join(out.split()))
            handle.write(data)
            offset += length
    if local_dir.exists():
        import shutil

        shutil.rmtree(local_dir)
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path=local_dir.parent)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pod_id")
    parser.add_argument("--dest", type=Path, default=Path("artifacts/campaign/student_qlora_canary_adapter"))
    args = parser.parse_args()
    dest = args.dest
    if not pull_via_scp_tar(args.pod_id, dest):
        pull_via_ssh_tar_chunks(args.pod_id, dest)
    ok, msg, total = verify_adapter(dest)
    print(json.dumps({"ok": ok, "path": msg, "bytes": total, "verified": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
