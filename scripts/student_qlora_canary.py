#!/usr/bin/env python3
"""Student-A QLoRA 50-step compatibility canary via Axolotl Hub serverless."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.serverless_inference import _resolve_api_key, endpoint_native_urls

MANIFEST = ROOT / "artifacts/campaign/manifests/student_qlora_canary_v1.json"
OUT_PATH = ROOT / "artifacts/campaign/student_qlora_canary_results.json"
LEDGER = ROOT / "artifacts/campaign/spend.json"
HUB_AXOLOTL = "cma1ofy3e000008l7he7c764k"
BASE_MODEL = "Qwen/Qwen2.5-32B-Instruct"
GPU_ID = "NVIDIA A100 80GB PCIe"
DATASET_URL = (
    "https://gist.githubusercontent.com/wwk5q8z6kk-bit/"
    "f3d11aa2463908190976f388b7b5afc1/raw/p1_distill_train_axolotl_v1.json"
)


def _resolve_hf_token() -> str:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value
    token_path = Path.home() / ".cache/huggingface/token"
    if token_path.is_file():
        return token_path.read_text().strip()
    return ""


def _wallet() -> float:
    proc = subprocess.run(["runpodctl", "user", "-o", "json"], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return 0.0
    return float(json.loads(proc.stdout).get("clientBalance", 0))


def _spend_gate(amount: float) -> None:
    subprocess.run(
        ["python3", str(ROOT / "scripts/campaign_spend.py"), "--ledger", str(LEDGER), "gate", "--amount", str(amount)],
        check=True,
    )


def _commit_spend(amount: float, endpoint_id: str) -> None:
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
            "Student QLoRA 50-step Axolotl canary",
            "--amount",
            str(amount),
            "--gpu",
            GPU_ID,
            "--rate-hr",
            "1.39",
            "--pod-id",
            endpoint_id,
        ],
        check=True,
    )


def _deploy_endpoint(name: str) -> dict[str, Any]:
    proc = subprocess.run(
        [
            "runpodctl",
            "serverless",
            "create",
            "--hub-id",
            HUB_AXOLOTL,
            "--gpu-id",
            GPU_ID,
            "--model-reference",
            f"https://huggingface.co/{BASE_MODEL}:main",
            "--name",
            name,
            "--workers-min",
            "0",
            "--workers-max",
            "1",
            "--idle-timeout",
            "300",
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip()
        if any(x in msg.lower() for x in ("stock", "available", "capacity")):
            return {"ok": False, "error": msg, "awaiting_gpu": True}
        return {"ok": False, "error": msg}
    data = json.loads(proc.stdout)
    return {"ok": True, "endpoint_id": data.get("id"), "cost_per_hr": data.get("costPerHr")}


def _delete_endpoint(endpoint_id: str) -> dict[str, Any]:
    proc = subprocess.run(
        ["runpodctl", "serverless", "delete", endpoint_id, "-o", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {"deleted": False, "error": proc.stderr.strip() or proc.stdout.strip()}
    return {"deleted": True, "endpoint_id": endpoint_id}


def _build_axolotl_args(*, max_steps: int) -> dict[str, Any]:
    return {
        "base_model": BASE_MODEL,
        "model_type": "AutoModelForCausalLM",
        "tokenizer_type": "AutoTokenizer",
        "trust_remote_code": True,
        "load_in_4bit": True,
        "adapter": "qlora",
        "datasets": [
            {
                "path": DATASET_URL,
                "type": "alpaca",
            }
        ],
        "dataset_prepared_path": "last_run_prepared",
        "val_set_size": 0.02,
        "output_dir": "./outputs/student-qlora-canary",
        "sequence_len": 2048,
        "sample_packing": False,
        "pad_to_sequence_len": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_linear": True,
        "gradient_accumulation_steps": 4,
        "micro_batch_size": 1,
        "num_epochs": 1,
        "max_steps": max_steps,
        "optimizer": "adamw_bnb_8bit",
        "lr_scheduler": "cosine",
        "learning_rate": 0.0002,
        "warmup_steps": 5,
        "logging_steps": 1,
        "saves_per_epoch": 1,
        "evals_per_epoch": 1,
        "gradient_checkpointing": True,
        "bf16": "auto",
        "flash_attention": True,
        "weight_decay": 0.0,
    }


def _build_payload(*, max_steps: int, run_id: str) -> dict[str, Any]:
    return {
        "input": {
            "user_id": "nano-lm",
            "model_id": "student-a-qlora",
            "run_id": run_id,
            "credentials": {
                "wandb_api_key": os.environ.get("WANDB_API_KEY", ""),
                "hf_token": _resolve_hf_token(),
            },
            "args": _build_axolotl_args(max_steps=max_steps),
        }
    }


def _extract_loss_curve(output: Any) -> list[dict[str, Any]]:
    curve: list[dict[str, Any]] = []
    if not isinstance(output, dict):
        return curve
    for key in ("loss_curve", "train_loss", "logs", "metrics"):
        value = output.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and "loss" in item:
                    curve.append(item)
    history = output.get("history")
    if isinstance(history, list):
        for item in history:
            if isinstance(item, dict) and "loss" in item:
                curve.append(item)
    return curve


def _final_loss(output: Any) -> float | None:
    curve = _extract_loss_curve(output)
    if curve:
        try:
            return float(curve[-1].get("loss"))
        except (TypeError, ValueError):
            pass
    if isinstance(output, dict):
        for key in ("final_loss", "loss", "train_loss"):
            if key in output:
                try:
                    return float(output[key])
                except (TypeError, ValueError):
                    continue
    return None


def _adapter_saved(output: Any) -> bool:
    if not isinstance(output, dict):
        return False
    for key in ("adapter_saved", "adapter_path", "output_dir", "artifact_path"):
        if output.get(key):
            return True
    artifacts = output.get("artifacts")
    if isinstance(artifacts, dict):
        return any(artifacts.values())
    text = json.dumps(output).lower()
    return "adapter_model" in text or "lora" in text and "saved" in text


def _submit_and_poll(endpoint_id: str, payload: dict[str, Any], *, timeout_s: int = 5400) -> dict[str, Any]:
    api_key = _resolve_api_key(None)
    urls = endpoint_native_urls(endpoint_id)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=60.0) as client:
        submit = client.post(urls["run"], headers=headers, json=payload)
        submit.raise_for_status()
        job_id = str(submit.json().get("id") or "")
        if not job_id:
            raise RuntimeError(f"missing job id: {submit.text}")
        status_url = f"{urls['run'].rsplit('/', 1)[0]}/status/{job_id}"
        deadline = time.time() + timeout_s
        last_status = ""
        while time.time() < deadline:
            status_resp = client.get(status_url, headers=headers)
            status_resp.raise_for_status()
            data = status_resp.json()
            last_status = str(data.get("status") or "")
            if last_status == "COMPLETED":
                return {
                    "status": "COMPLETED",
                    "output": data.get("output"),
                    "raw_status": data,
                    "job_id": job_id,
                }
            if last_status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
                return {
                    "status": last_status,
                    "error": data.get("error") or data.get("output"),
                    "raw_status": data,
                    "job_id": job_id,
                }
            time.sleep(15)
    return {"status": "TIMEOUT", "error": f"poll exceeded {timeout_s}s", "job_id": job_id}


def run_canary(*, max_steps: int = 50, est_cost: float = 2.5) -> dict[str, Any]:
    wallet_before = _wallet()
    _spend_gate(est_cost)
    run_id = f"student_qlora_canary_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    endpoint_name = f"student-qlora-canary-{uuid.uuid4().hex[:8]}"
    result: dict[str, Any] = {
        "schema": "nano.campaign.student_qlora_canary.v1",
        "timestamp": datetime.now(UTC).isoformat(),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "run_id": run_id,
        "base_model": BASE_MODEL,
        "dataset_revision": "p1_distill_train_v1",
        "dataset_url": DATASET_URL,
        "max_steps": max_steps,
        "wallet_before_usd": round(wallet_before, 4),
        "status": "RUNNING",
    }
    endpoint_id = None
    try:
        deploy = _deploy_endpoint(endpoint_name)
        if not deploy.get("ok"):
            result.update(
                {
                    "status": "AWAITING_GPU" if deploy.get("awaiting_gpu") else "DEPLOY_FAILED",
                    "error": deploy.get("error"),
                    "endpoint_deleted": False,
                }
            )
            return result
        endpoint_id = str(deploy["endpoint_id"])
        result["endpoint_id"] = endpoint_id
        _commit_spend(est_cost, endpoint_id)
        payload = _build_payload(max_steps=max_steps, run_id=run_id)
        job = _submit_and_poll(endpoint_id, payload)
        result["job_id"] = job.get("job_id")
        result["job_status"] = job.get("status")
        result["raw_status"] = job.get("raw_status")
        output = job.get("output")
        if output is None and isinstance(job.get("raw_status"), dict):
            output = job["raw_status"].get("output")
        if isinstance(output, dict) and "output" in output and len(output) == 1:
            output = output["output"]
        result["raw_output"] = output
        final_loss = _final_loss(output)
        loss_curve = _extract_loss_curve(output)
        adapter_saved = _adapter_saved(output)
        finite_loss = final_loss is not None and final_loss == final_loss and final_loss < 1e6
        passed = job.get("status") == "COMPLETED" and finite_loss and adapter_saved
        result.update(
            {
                "final_loss": final_loss,
                "loss_curve": loss_curve,
                "adapter_saved": adapter_saved,
                "finite_loss": finite_loss,
                "pass": passed,
                "status": "PASS" if passed else ("FAIL" if job.get("status") == "COMPLETED" else str(job.get("status"))),
                "error": job.get("error"),
            }
        )
        if passed:
            result["round1_adaptation_gate"] = "UNLOCKED"
        return result
    finally:
        if endpoint_id:
            result["endpoint_cleanup"] = _delete_endpoint(endpoint_id)
            result["endpoint_deleted"] = bool(result["endpoint_cleanup"].get("deleted"))
        result["wallet_after_usd"] = round(_wallet(), 4)
        result["wallet_delta_usd"] = round(result["wallet_after_usd"] - wallet_before, 4)
        OUT_PATH.write_text(json.dumps(result, indent=2) + "\n")
        if MANIFEST.is_file():
            manifest = json.loads(MANIFEST.read_text())
            manifest["status"] = result.get("status", "COMPLETE")
            manifest["endpoint_id"] = endpoint_id
            MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Student QLoRA Axolotl canary")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--est-cost", type=float, default=2.5)
    args = parser.parse_args()
    payload = run_canary(max_steps=args.max_steps, est_cost=args.est_cost)
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
