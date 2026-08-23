#!/usr/bin/env python3
"""Student QLoRA 50-step pod canary — raw pod with log tail + SCP adapter pull."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pull_native_weights import resolve_direct_scp

MANIFEST = ROOT / "artifacts/campaign/manifests/student_qlora_canary_v1.json"
OUT_PATH = ROOT / "artifacts/campaign/student_qlora_canary_results.json"
FAIL_LOG_PATH = ROOT / "artifacts/campaign/student_qlora_canary_last.log"
ADAPTER_DIR = ROOT / "artifacts/campaign/student_qlora_canary_adapter"
CHECKPOINT = ROOT / "artifacts/campaign/checkpoint_v3.json"
LEDGER = ROOT / "artifacts/campaign/spend.json"
SSH = ROOT / "scripts/runpod_pod_ssh.sh"
TRAIN_SCRIPT_GIST = (
    "https://gist.githubusercontent.com/wwk5q8z6kk-bit/"
    "cee97a251dcf9848227e5b7a67b77531/raw/student_qlora_minimal_train.py"
)
PULL_SCRIPT = ROOT / "scripts/pull_qlora_adapter.py"

BASE_MODEL = "Qwen/Qwen2.5-32B-Instruct"
DATASET_URL = (
    "https://gist.githubusercontent.com/wwk5q8z6kk-bit/"
    "f3d11aa2463908190976f388b7b5afc1/raw/p1_distill_train_axolotl_v1.json"
)
# Prefer cheaper GPUs that fit 32B QLoRA (~20-28GB VRAM); try in order at launch.
GPU_PREFERENCES: tuple[tuple[str, float], ...] = (
    ("NVIDIA L40S", 0.99),
    ("NVIDIA A100 80GB PCIe", 1.39),
    ("NVIDIA A100-SXM4-80GB", 1.59),
)
DATA_CENTER = os.environ.get("RUNPOD_DATA_CENTER_IDS", "US-KS-2")
TEMPLATE = "runpod-torch-v240"
CONTAINER_DISK_GB = 80
REMOTE_LOG = "/workspace/qlora_canary.log"
REMOTE_ADAPTER = "/workspace/qlora_canary_adapter"
MILESTONE_STEPS = (1, 10, 25, 50)
STEP_LOSS_RE = re.compile(r"STEP_LOSS step=(\d+) loss=([\d.eE+-]+)")
MILESTONE_RE = re.compile(r"MILESTONE step=(\d+) loss=([\d.eE+-]+)")


def _wallet() -> float:
    proc = subprocess.run(["runpodctl", "user", "-o", "json"], capture_output=True, text=True, check=False)
    return float(json.loads(proc.stdout).get("clientBalance", 0)) if proc.returncode == 0 else 0.0


def _spend_gate(amount: float) -> None:
    subprocess.run(
        ["python3", str(ROOT / "scripts/campaign_spend.py"), "--ledger", str(LEDGER), "gate", "--amount", str(amount)],
        check=True,
    )


def _commit_spend(amount: float, pod_id: str, rate: float, gpu_id: str, attempt: int) -> None:
    subprocess.run(
        [
            "python3",
            str(ROOT / "scripts/campaign_spend.py"),
            "--ledger",
            str(LEDGER),
            "commit",
            "--lane",
            "student_qlora",
            "--description",
            f"Student QLoRA pod canary attempt-{attempt}",
            "--amount",
            str(amount),
            "--gpu",
            gpu_id,
            "--rate-hr",
            str(rate),
            "--pod-id",
            pod_id,
        ],
        check=True,
    )


def _ssh(pod_id: str, cmd: str, *, retries: int = 3) -> str:
    last_err = ""
    for attempt in range(retries):
        endpoint = resolve_direct_scp(pod_id)
        if endpoint is not None:
            host, port, ssh_key = endpoint
            proc = subprocess.run(
                [
                    "ssh",
                    "-i",
                    str(ssh_key),
                    "-p",
                    str(port),
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=30",
                    "-o",
                    "StrictHostKeyChecking=accept-new",
                    f"root@{host}",
                    cmd,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                return proc.stdout.strip()
            last_err = (proc.stderr or proc.stdout or f"rc={proc.returncode}").strip()[-800:]
        else:
            proc = subprocess.run(["bash", str(SSH), pod_id, cmd], capture_output=True, text=True, check=False)
            if proc.returncode == 0:
                return proc.stdout.strip()
            last_err = (proc.stderr or proc.stdout or f"rc={proc.returncode}").strip()[-800:]
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"ssh failed: {last_err}")


def _launch_pod(gpu_id: str) -> dict[str, Any]:
    terminate = subprocess.run(
        ["date", "-u", "-v+75M", "+%Y-%m-%dT%H:%M:%SZ"],
        capture_output=True,
        text=True,
        check=False,
    )
    if terminate.returncode != 0:
        terminate = subprocess.run(
            ["date", "-u", "-d", "+75 minutes", "+%Y-%m-%dT%H:%M:%SZ"],
            capture_output=True,
            text=True,
            check=True,
        )
    proc = subprocess.run(
        [
            "runpodctl",
            "pod",
            "create",
            "--name",
            f"qlora-canary-{datetime.now(UTC).strftime('%m%d%H%M')}",
            "--template-id",
            TEMPLATE,
            "--gpu-id",
            gpu_id,
            "--data-center-ids",
            DATA_CENTER,
            "--cloud-type",
            "SECURE",
            "--container-disk-in-gb",
            str(CONTAINER_DISK_GB),
            "--terminate-after",
            terminate.stdout.strip(),
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip()
        if any(x in msg.lower() for x in ("stock", "available", "capacity", "no available")):
            return {"ok": False, "awaiting_gpu": True, "error": msg, "gpu_id": gpu_id}
        return {"ok": False, "error": msg, "gpu_id": gpu_id}
    data = json.loads(proc.stdout)
    return {
        "ok": True,
        "pod_id": data["id"],
        "rate": float(data.get("costPerHr", 1.39)),
        "gpu_id": gpu_id,
    }


def _launch_pod_preferred() -> dict[str, Any]:
    last: dict[str, Any] = {"ok": False, "error": "no gpu preferences configured"}
    for gpu_id, default_rate in GPU_PREFERENCES:
        launch = _launch_pod(gpu_id)
        if launch.get("ok"):
            launch.setdefault("rate", default_rate)
            return launch
        last = launch
        if not launch.get("awaiting_gpu"):
            continue
    return last


def _cuda_gate(pod_id: str) -> bool:
    proc = subprocess.run(["bash", str(ROOT / "scripts/native_cuda_gate.sh"), pod_id], check=False)
    return proc.returncode == 0


def _cuda_gate_with_retries(
    *, max_attempts: int = 4, est_cost: float, attempt: int
) -> tuple[str | None, float, str | None, str | None]:
    """Launch pod(s) until CUDA gate passes — mirrors native100 extended pattern."""
    last_error: str | None = None
    selected_gpu: str | None = None
    for gate_attempt in range(1, max_attempts + 1):
        launch = _launch_pod_preferred()
        if not launch.get("ok"):
            last_error = launch.get("error", "launch failed")
            if launch.get("awaiting_gpu"):
                return None, 0.0, f"AWAITING_GPU: {last_error}", launch.get("gpu_id")
            time.sleep(15)
            continue
        pod_id = str(launch["pod_id"])
        rate = float(launch.get("rate", 1.39))
        selected_gpu = str(launch.get("gpu_id", GPU_PREFERENCES[1][0]))
        _commit_spend(est_cost, pod_id, rate, selected_gpu, attempt)
        time.sleep(50 + 10 * (gate_attempt - 1))
        if _cuda_gate(pod_id):
            return pod_id, rate, None, selected_gpu
        last_error = f"cuda gate failed attempt {gate_attempt}/{max_attempts}"
        # native_cuda_gate.sh deletes the pod on failure
    return None, 0.0, last_error, selected_gpu


def _upload_train_script(pod_id: str) -> None:
    local = ROOT / "scripts/student_qlora_minimal_train.py"
    remote = "/workspace/nano-lm/scripts/student_qlora_minimal_train.py"
    endpoint = resolve_direct_scp(pod_id)
    if endpoint is None:
        raise RuntimeError("direct scp unavailable for train script upload")
    host, port, ssh_key = endpoint
    _ssh(pod_id, "mkdir -p /workspace/nano-lm/scripts")
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
            "ConnectTimeout=30",
            "-o",
            "StrictHostKeyChecking=accept-new",
            str(local),
            f"root@{host}:{remote}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"scp train script failed: {proc.stderr[-500:]}")


def _bootstrap(pod_id: str) -> None:
    _upload_train_script(pod_id)
    cmd = (
        "set -e; export HF_HOME=/workspace/hf_cache TRANSFORMERS_CACHE=/workspace/hf_cache; "
        "pip install -q 'peft==0.12.0' 'bitsandbytes==0.43.3' 'accelerate==0.34.2' "
        "'datasets==2.21.0' 'transformers==4.44.2' safetensors sentencepiece; "
        f"nohup python3 /workspace/nano-lm/scripts/student_qlora_minimal_train.py "
        f"--dataset-url '{DATASET_URL}' --output-dir {REMOTE_ADAPTER} "
        f"--base-model '{BASE_MODEL}' --max-steps 50 --log-file {REMOTE_LOG} "
        f"> {REMOTE_LOG} 2>&1 & echo TRAIN_PID=$!"
    )
    out = _ssh(pod_id, cmd)
    if "TRAIN_PID=" not in out:
        raise RuntimeError(f"bootstrap did not start trainer: {out!r}")


def _parse_log(text: str) -> dict[str, Any]:
    curve: list[dict[str, Any]] = []
    milestones: dict[str, float | None] = {f"step_{s}": None for s in MILESTONE_STEPS}
    model_loaded = "MODEL_LOADED" in text
    qlora_init = "QLORA_INIT_OK" in text
    adapter_saved = "ADAPTER_SAVED" in text
    train_done = "TRAIN_DONE" in text
    train_fail = "TRAIN_FAIL:" in text
    fatal = (
        "MODEL_LOAD_FAIL" in text
        or "CUDA out of memory" in text
        or train_fail
        or ("Traceback" in text and "TRAIN_START" in text and not train_done and "STEP_LOSS" not in text)
    )

    for match in STEP_LOSS_RE.finditer(text):
        step = int(match.group(1))
        loss = float(match.group(2))
        curve.append({"step": step, "loss": loss})
        if step in MILESTONE_STEPS:
            milestones[f"step_{step}"] = loss
    for match in MILESTONE_RE.finditer(text):
        step = int(match.group(1))
        milestones[f"step_{step}"] = float(match.group(2))

    final_loss = milestones.get("step_50")
    if final_loss is None and curve:
        final_loss = curve[-1]["loss"]
    step_1 = milestones.get("step_1")
    if step_1 is None and curve:
        step_1 = curve[0]["loss"]
    loss_decreased = bool(step_1 is not None and final_loss is not None and final_loss < step_1)
    return {
        "loss_curve": curve,
        "loss_milestones": milestones,
        "final_loss": final_loss,
        "loss_decreased": loss_decreased,
        "model_loaded": model_loaded and qlora_init,
        "adapter_saved": adapter_saved,
        "train_done": train_done,
        "fatal_error": fatal,
    }


def _save_remote_log(pod_id: str) -> str | None:
    try:
        full = _ssh(pod_id, f"cat {REMOTE_LOG} 2>/dev/null || true")
        if full.strip():
            FAIL_LOG_PATH.write_text(full)
            return str(FAIL_LOG_PATH)
    except RuntimeError:
        pass
    return None


def _poll_training(pod_id: str, *, timeout_s: int = 5400) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_text = ""
    while time.time() < deadline:
        try:
            tail = _ssh(pod_id, f"tail -80 {REMOTE_LOG} 2>/dev/null || echo LOG_MISSING")
        except RuntimeError:
            time.sleep(20)
            continue
        last_text = tail
        parsed = _parse_log(tail)
        if parsed["train_done"] or parsed["fatal_error"]:
            full = _ssh(pod_id, f"cat {REMOTE_LOG} 2>/dev/null || true")
            return _parse_log(full)
        if "TRAIN_START" in tail:
            time.sleep(45)
            continue
        time.sleep(25)
    return _parse_log(last_text)


def _terminate_pod(pod_id: str) -> bool:
    proc = subprocess.run(["runpodctl", "pod", "delete", pod_id], capture_output=True, text=True, check=False)
    return proc.returncode == 0


def _pull_adapter(pod_id: str) -> dict[str, Any]:
    proc = subprocess.run(
        ["python3", str(PULL_SCRIPT), pod_id, "--dest", str(ADAPTER_DIR)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr[-500:] or proc.stdout}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"ok": False, "error": proc.stdout[-500:]}


def run_pod_canary(*, est_cost: float = 2.0, attempt: int = 5) -> dict[str, Any]:
    wallet_before = _wallet()
    _spend_gate(est_cost)
    run_id = f"student_qlora_pod_canary_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    result: dict[str, Any] = {
        "schema": "nano.campaign.student_qlora_canary.v3",
        "timestamp": datetime.now(UTC).isoformat(),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "run_id": run_id,
        "attempt": attempt,
        "surface": "raw_pod",
        "capture_strategy": "direct_tcp_ssh_log_tail_scp_adapter",
        "container_disk_gb": CONTAINER_DISK_GB,
        "gpu_preferences": [g for g, _ in GPU_PREFERENCES],
        "base_model": BASE_MODEL,
        "dataset_revision": "p1_distill_train_v1",
        "dataset_url": DATASET_URL,
        "max_steps": 50,
        "wallet_before_usd": round(wallet_before, 4),
        "status": "RUNNING",
    }
    pod_id = None
    try:
        pod_id, rate, cuda_err, selected_gpu = _cuda_gate_with_retries(
            max_attempts=4, est_cost=est_cost, attempt=attempt
        )
        if pod_id is None:
            status = "AWAITING_GPU" if cuda_err and cuda_err.startswith("AWAITING_GPU") else "CUDA_FAIL"
            result.update({"status": status, "error": cuda_err or "cuda gate failed", "pod_deleted": True})
            if selected_gpu:
                result["gpu"] = selected_gpu
            return result
        result["pod_id"] = pod_id
        result["gpu"] = selected_gpu or GPU_PREFERENCES[1][0]
        result["datacenter"] = DATA_CENTER
        result["container_disk_gb"] = CONTAINER_DISK_GB
        _bootstrap(pod_id)
        parsed = _poll_training(pod_id)
        pull = _pull_adapter(pod_id)
        adapter_ok = bool(pull.get("ok"))
        adapter_bytes = int(pull.get("bytes", 0))
        adapter_path = str(ADAPTER_DIR) if adapter_ok else None
        passed = (
            parsed["model_loaded"]
            and parsed["final_loss"] is not None
            and parsed["final_loss"] == parsed["final_loss"]
            and 0 < float(parsed["final_loss"]) < 1e6
            and parsed["loss_decreased"]
            and adapter_ok
            and parsed["train_done"]
            and not parsed["fatal_error"]
        )
        result.update(
            {
                "loss_curve": parsed["loss_curve"],
                "loss_milestones": parsed["loss_milestones"],
                "final_loss": parsed["final_loss"],
                "loss_decreased": parsed["loss_decreased"],
                "model_loaded": parsed["model_loaded"],
                "finite_loss": parsed["final_loss"] is not None,
                "adapter_saved": adapter_ok,
                "adapter_location": adapter_path,
                "adapter_bytes": adapter_bytes,
                "train_done": parsed["train_done"],
                "pass": passed,
                "status": "PASS" if passed else "FAIL",
                "round1_adaptation_gate": "UNLOCKED_PENDING_LAUNCH" if passed else "BLOCKED",
            }
        )
        if not passed:
            reasons = []
            if not parsed["model_loaded"]:
                reasons.append("model_not_loaded")
            if not parsed["train_done"]:
                reasons.append("train_not_done")
            if not parsed["loss_decreased"]:
                reasons.append("loss_not_decreased")
            if not adapter_ok:
                reasons.append("adapter_pull_failed")
            result["error"] = ",".join(reasons) or "pass_criteria_unmet"
            log_path = _save_remote_log(pod_id)
            if log_path:
                result["failure_log"] = log_path
        return result
    except Exception as exc:
        result.update({"status": "FAIL", "error": str(exc), "pass": False, "round1_adaptation_gate": "BLOCKED"})
        return result
    finally:
        if pod_id:
            result["pod_deleted"] = _terminate_pod(pod_id)
        result["wallet_after_usd"] = round(_wallet(), 4)
        result["wallet_delta_usd"] = round(result["wallet_after_usd"] - wallet_before, 4)
        OUT_PATH.write_text(json.dumps(result, indent=2) + "\n")
        if MANIFEST.is_file():
            manifest = json.loads(MANIFEST.read_text())
            manifest.update(
                {
                    "status": result.get("status", "COMPLETE"),
                    "attempt": attempt,
                    "surface": "raw_pod",
                    "capture_strategy": "direct_tcp_ssh_log_tail_scp_adapter",
                    "container_disk_gb": CONTAINER_DISK_GB,
                    "pod_id": pod_id,
                }
            )
            MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        if CHECKPOINT.is_file() and result.get("pass"):
            ckpt = json.loads(CHECKPOINT.read_text())
            ckpt.setdefault("student_a", {})["qlora_canary_status"] = "PASS"
            ckpt["student_a"]["qlora_canary_attempt"] = attempt
            ckpt["student_a"]["round1_adaptation_gate"] = "UNLOCKED_PENDING_LAUNCH"
            ckpt["next_dollar"] = "Bounded Round-1 QLoRA adaptation (canary PASS)"
            CHECKPOINT.write_text(json.dumps(ckpt, indent=2) + "\n")
        elif CHECKPOINT.is_file():
            ckpt = json.loads(CHECKPOINT.read_text())
            ckpt.setdefault("student_a", {})["qlora_canary_status"] = result.get("status", "FAIL")
            ckpt["student_a"]["qlora_canary_attempt"] = attempt
            CHECKPOINT.write_text(json.dumps(ckpt, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Student QLoRA pod canary")
    parser.add_argument("--est-cost", type=float, default=2.0)
    parser.add_argument("--attempt", type=int, default=5)
    args = parser.parse_args()
    payload = run_pod_canary(est_cost=args.est_cost, attempt=args.attempt)
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
