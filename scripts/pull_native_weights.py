#!/usr/bin/env python3
"""Pull native checkpoint weights from a RunPod pod.

Preferred path: direct TCP SCP when runpodctl exposes host:port (typically ~1 min
for ~1.2GB). Fallback: offset-chunked base64 over scripts/runpod_pod_ssh.sh proxy.
"""

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
CHUNK_BYTES = 4 * 1024 * 1024
SSH = ROOT / "scripts" / "runpod_pod_ssh.sh"
DEFAULT_SSH_KEY = Path.home() / ".runpod/ssh/runpodctl-ssh-key"
_SSH_CMD_RE = re.compile(r"@([\d.]+)\s+-p\s+(\d+)")


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


def resolve_direct_scp(pod_id: str) -> tuple[str, int, Path] | None:
    """Return (host, port, ssh_key) for direct TCP SCP when exposed."""
    proc = subprocess.run(
        ["runpodctl", "pod", "get", pod_id, "-o", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    pod = json.loads(proc.stdout)
    ssh_key = Path(os.environ.get("RUNPOD_SSH_KEY", str(DEFAULT_SSH_KEY)))

    host = pod.get("publicIp")
    machine = pod.get("machine") or {}
    if not host:
        host = machine.get("publicIp")

    port: int | None = None
    mappings = pod.get("portMappings") or {}
    if isinstance(mappings, dict):
        raw_port = mappings.get("22") or mappings.get(22)
        if raw_port is not None:
            port = int(raw_port)

    if port is None:
        for entry in (pod.get("runtime") or {}).get("ports") or []:
            if entry.get("privatePort") == 22:
                host = entry.get("ip") or host
                public_port = entry.get("publicPort")
                if public_port is not None:
                    port = int(public_port)
                break

    if host is None or port is None:
        ssh_cmd = (pod.get("ssh") or {}).get("ssh_command") or pod.get("sshCmd") or ""
        match = _SSH_CMD_RE.search(ssh_cmd)
        if match:
            host = match.group(1)
            port = int(match.group(2))

    if host and port and ssh_key.is_file():
        return str(host), port, ssh_key
    return None


def pull_via_scp(
    pod_id: str,
    run_id: str,
    *,
    checkpoint_dir: Path = Path("artifacts/native_checkpoints"),
) -> Path | None:
    endpoint = resolve_direct_scp(pod_id)
    if endpoint is None:
        return None
    host, port, ssh_key = endpoint
    remote_path = f"/workspace/nano-lm/artifacts/native_checkpoints/{run_id}/latest.pt"
    out = checkpoint_dir / run_id / "latest.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():
        out.unlink()
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
            "ConnectTimeout=45",
            "-o",
            "StrictHostKeyChecking=accept-new",
            f"root@{host}:{remote_path}",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        if out.is_file():
            out.unlink()
        print(f"scp failed: {proc.stderr[-400:]}", file=sys.stderr)
        return None
    ok, msg = verify_weights(run_id, checkpoint_dir)
    if not ok:
        if out.is_file():
            out.unlink()
        print(f"scp verify failed: {msg}", file=sys.stderr)
        return None
    print(f"scp pull ok host={host} port={port} bytes={out.stat().st_size}", file=sys.stderr)
    return out


def pull_via_ssh_chunks(
    pod_id: str,
    run_id: str,
    *,
    checkpoint_dir: Path = Path("artifacts/native_checkpoints"),
) -> Path:
    remote_path = f"/workspace/nano-lm/artifacts/native_checkpoints/{run_id}/latest.pt"
    total = remote_size(pod_id, remote_path)
    if total < MIN_WEIGHT_BYTES:
        raise RuntimeError(f"remote file too small: {total}")
    out = checkpoint_dir / run_id / "latest.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
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


def pull_weights(pod_id: str, run_id: str, *, checkpoint_dir: Path = Path("artifacts/native_checkpoints")) -> Path:
    ok, msg = verify_weights(run_id, checkpoint_dir)
    if ok:
        return Path(msg)

    scp_path = pull_via_scp(pod_id, run_id, checkpoint_dir=checkpoint_dir)
    if scp_path is not None:
        return scp_path

    print("scp unavailable or failed; falling back to ssh base64 chunks", file=sys.stderr)
    out = pull_via_ssh_chunks(pod_id, run_id, checkpoint_dir=checkpoint_dir)
    ok, msg = verify_weights(run_id, checkpoint_dir)
    if not ok:
        raise RuntimeError(msg)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull native checkpoint weights from pod")
    parser.add_argument("pod_id")
    parser.add_argument("run_id")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/native_checkpoints"))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--method", choices=["auto", "scp", "ssh"], default="auto")
    args = parser.parse_args()
    if args.verify_only:
        ok, msg = verify_weights(args.run_id, args.checkpoint_dir)
        print(json.dumps({"ok": ok, "path": msg}))
        return 0 if ok else 1

    if args.method == "scp":
        out = pull_via_scp(args.pod_id, args.run_id, checkpoint_dir=args.checkpoint_dir)
        if out is None:
            print(json.dumps({"error": "scp pull failed"}))
            return 1
    elif args.method == "ssh":
        out = pull_via_ssh_chunks(args.pod_id, args.run_id, checkpoint_dir=args.checkpoint_dir)
    else:
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
