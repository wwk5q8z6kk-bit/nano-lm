#!/usr/bin/env python3
"""Poll a RunPod pod and terminate it the moment work stops.

RunPod is a container orchestrator: a backgrounded (`nohup ... &`) training
script that dies leaves the pod alive with no foreground process, so it sits at
0% utilisation and keeps billing. That is documented RunPod behaviour, not a
bug — the orchestration has to notice.

This watchdog terminates on ANY of:
  * the done-marker appears (success),
  * no training process AND no done-marker (the script died),
  * GPU utilisation stays at 0% for `--idle-strikes` consecutive polls,
  * the wall-clock deadline passes.

Usage:
    python3 scripts/pod_watchdog.py POD_ID --marker NATIVE30_REVALIDATION_DONE \
        --log /workspace/reval30.log --max-minutes 150
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SSH = ROOT / "scripts" / "runpod_pod_ssh.sh"


def _ssh(pod: str, cmd: str, timeout: int = 120) -> str:
    try:
        proc = subprocess.run(
            ["bash", str(SSH), pod, cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.stdout
    except subprocess.TimeoutExpired:
        return ""


def parse_probe(text: str) -> dict | None:
    """Parse WD_KEY=value lines. Returns None when the probe is unusable.

    Returning None rather than zeros matters: a failed SSH must never look like
    'the training process is gone', which is what terminates the pod.
    """
    found: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("WD_") and "=" in line:
            key, _, value = line.partition("=")
            found[key] = value.strip()
    if not {"WD_DONE", "WD_PROCS"} <= set(found):
        return None
    try:
        done_count = int(found["WD_DONE"] or 0)
        procs = int(found["WD_PROCS"] or 0)
    except ValueError:
        return None
    util_raw = found.get("WD_UTIL", "")
    try:
        util = int(util_raw) if util_raw.strip().isdigit() else -1
    except ValueError:
        util = -1
    return {"done": done_count > 0, "procs": procs, "util": util}


def _pod_alive(pod: str) -> bool:
    proc = subprocess.run(
        ["runpodctl", "pod", "list"], capture_output=True, text=True, check=False
    )
    try:
        return any(p.get("id") == pod for p in json.loads(proc.stdout))
    except Exception:
        return True  # assume alive rather than terminate on a parse error


def fetch_results(pod: str, remote_dir: str, local_dir: Path) -> dict:
    """Pull results off the pod BEFORE it is terminated.

    The first version of this watchdog terminated on the done-marker without
    retrieving anything, so a completed nine-arm run — nine trained models and
    eighteen eval files — was deleted with the pod. Terminating is only safe
    once the output is somewhere else.

    Streams a base64 tar over the existing SSH helper: the payload is a handful
    of small JSON files, and this avoids depending on scp being reachable.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    listing = _ssh(pod, f"ls -1 {remote_dir}/*.json 2>/dev/null | wc -l")
    expected = next((int(x) for x in listing.split() if x.strip().isdigit()), 0)

    b64 = _ssh(
        pod,
        f"tar -czf - -C {remote_dir} . 2>/dev/null | base64 -w0 2>/dev/null "
        f"|| tar -czf - -C {remote_dir} . 2>/dev/null | base64",
        timeout=300,
    )
    payload = "".join(b64.split())
    if not payload:
        return {"fetched": False, "reason": "empty archive from pod", "expected_files": expected}

    import base64 as _b64
    import io
    import tarfile

    try:
        raw = _b64.b64decode(payload)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
            tf.extractall(local_dir, filter="data")
    except Exception as exc:
        return {"fetched": False, "reason": f"{type(exc).__name__}: {exc}", "expected_files": expected}

    got = len(list(local_dir.glob("*.json")))
    return {
        "fetched": got > 0,
        "local_dir": str(local_dir),
        "expected_files": expected,
        "retrieved_files": got,
        "complete": got >= expected > 0,
    }


def terminate(pod: str, reason: str) -> dict:
    proc = subprocess.run(
        ["runpodctl", "pod", "delete", pod], capture_output=True, text=True, check=False
    )
    return {
        "terminated": proc.returncode == 0,
        "reason": reason,
        "at": datetime.now(UTC).isoformat(),
        "output": (proc.stdout or proc.stderr)[-200:],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Terminate a pod when work stops")
    ap.add_argument("pod")
    ap.add_argument("--marker", required=True, help="done-marker string in the log")
    ap.add_argument("--log", required=True)
    ap.add_argument("--process", default="train_native", help="pgrep pattern for live work")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--max-minutes", type=int, default=150)
    ap.add_argument("--idle-strikes", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/campaign/pod_watchdog_log.json")
    args = ap.parse_args()

    deadline = time.time() + args.max_minutes * 60
    idle = 0
    events: list[dict] = []

    while True:
        if time.time() > deadline:
            result = terminate(args.pod, f"deadline {args.max_minutes}min exceeded")
            events.append(result)
            break
        if not _pod_alive(args.pod):
            events.append({"note": "pod already gone", "at": datetime.now(UTC).isoformat()})
            break

        # LABELLED output. Positional parsing killed a healthy run: `grep -c`
        # prints "0" AND exits 1, so `|| echo 0` emitted a duplicate line, the
        # process count was read from the wrong index, and the watchdog concluded
        # the script had died while the GPU was at 97%.
        probe = _ssh(
            args.pod,
            f"echo \"WD_DONE=$(grep -c '{args.marker}' {args.log} 2>/dev/null | head -1)\"; "
            f"echo \"WD_PROCS=$(pgrep -c -f '{args.process}' 2>/dev/null | head -1)\"; "
            "echo \"WD_UTIL=$(nvidia-smi --query-gpu=utilization.gpu "
            '--format=csv,noheader,nounits 2>/dev/null | head -1)"',
        )
        fields = parse_probe(probe)
        if fields is None:
            # An unreadable probe is NOT evidence of death — SSH hiccups happen.
            events.append({"at": datetime.now(UTC).isoformat(), "note": "probe unreadable; skipping"})
            time.sleep(args.interval)
            continue
        done, procs, util = fields["done"], fields["procs"], fields["util"]

        snapshot = {
            "at": datetime.now(UTC).isoformat(),
            "done_marker": done,
            "processes": procs,
            "gpu_util": util,
        }
        events.append(snapshot)
        print(json.dumps(snapshot), flush=True)

        if done:
            events.append(terminate(args.pod, "done-marker present: work complete"))
            break
        if procs == 0 and probe:
            # No training process and no done-marker: the backgrounded script died.
            events.append(terminate(args.pod, "no training process and no done-marker — script died"))
            break
        idle = idle + 1 if util == 0 else 0
        if idle >= args.idle_strikes:
            events.append(
                terminate(args.pod, f"GPU idle for {idle} consecutive polls")
            )
            break

        time.sleep(args.interval)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"pod": args.pod, "events": events}, indent=2) + "\n")
    print(json.dumps(events[-1], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
