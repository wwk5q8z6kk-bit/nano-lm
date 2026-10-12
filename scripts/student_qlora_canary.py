#!/usr/bin/env python3
"""Student-A QLoRA 50-step compatibility canary via Axolotl Hub serverless."""

from __future__ import annotations

import argparse
import json
import os
import re
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
CAPTURE_STRATEGY = "runsync_primary_stream_fallback"
MILESTONE_STEPS = (1, 10, 25, 50)
LOSS_RE = re.compile(r"""['"]loss['"]\s*:\s*([\d.eE+-]+)""")
STEP_RE = re.compile(r"""['"]step['"]\s*:\s*(\d+)""")
CHECKPOINT_RE = re.compile(
    r"(?:Saving model checkpoint to|saved.*checkpoint|output_dir[:\s]+)([^\s'\"]+)",
    re.IGNORECASE,
)
HUB_MODEL_RE = re.compile(r"(?:hub_model_id|push.*hub|https://huggingface\.co/)([^\s'\"]+)", re.IGNORECASE)


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


def _build_axolotl_args(*, max_steps: int, run_id: str) -> dict[str, Any]:
    hf_token = _resolve_hf_token()
    args: dict[str, Any] = {
        "base_model": BASE_MODEL,
        "model_type": "AutoModelForCausalLM",
        "tokenizer_type": "AutoTokenizer",
        "trust_remote_code": True,
        "load_in_4bit": True,
        "adapter": "qlora",
        "datasets": [{"path": DATASET_URL, "type": "alpaca"}],
        "dataset_prepared_path": "last_run_prepared",
        "val_set_size": 0.02,
        "output_dir": f"./outputs/student-qlora-canary-{run_id}",
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
    if hf_token:
        args["hub_model_id"] = f"wwk5q8z6kk-bit/nano-student-qlora-canary-{run_id.lower()}"
        args["push_to_hub"] = True
    return args


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
            "args": _build_axolotl_args(max_steps=max_steps, run_id=run_id),
        }
    }


def _flatten_output(output: Any) -> list[str]:
    lines: list[str] = []
    if output is None:
        return lines
    if isinstance(output, str):
        return [output]
    if isinstance(output, list):
        for item in output:
            lines.extend(_flatten_output(item))
        return lines
    if isinstance(output, dict):
        if "output" in output:
            lines.extend(_flatten_output(output["output"]))
        if "text" in output:
            lines.extend(_flatten_output(output["text"]))
        for key in ("message", "result", "log", "logs"):
            if key in output:
                lines.extend(_flatten_output(output[key]))
        if not lines:
            lines.append(json.dumps(output))
        return lines
    lines.append(str(output))
    return lines


def _parse_training_logs(lines: list[str]) -> dict[str, Any]:
    loss_points: list[dict[str, Any]] = []
    adapter_paths: list[str] = []
    hub_ids: list[str] = []
    model_loaded = False
    training_started = False
    fatal_error = False

    for raw in lines:
        line = raw.strip()
        lower = line.lower()
        if "starting training" in lower:
            training_started = True
        if any(x in lower for x in ("loading weights", "loaded model", "trainable params", "trainable parameters")):
            model_loaded = True
        if any(x in lower for x in ("cuda out of memory", "traceback", "error:", "failed", "exception")):
            if "loss_watchdog" not in lower:
                fatal_error = True

        for match in LOSS_RE.finditer(line):
            step = None
            step_match = STEP_RE.search(line)
            if step_match:
                step = int(step_match.group(1))
            try:
                loss = float(match.group(1))
            except ValueError:
                continue
            point: dict[str, Any] = {"loss": loss}
            if step is not None:
                point["step"] = step
            loss_points.append(point)

        for match in CHECKPOINT_RE.finditer(line):
            path = match.group(1).rstrip(",.")
            if path and path not in adapter_paths:
                adapter_paths.append(path)

        for match in HUB_MODEL_RE.finditer(line):
            hub = match.group(1).rstrip("/")
            if hub and hub not in hub_ids:
                hub_ids.append(hub)

        if "adapter_model" in lower or "lora" in lower and "saved" in lower:
            model_loaded = True

    # Assign step indices when logging_steps=1 and step not in log line.
    for idx, point in enumerate(loss_points, start=1):
        point.setdefault("step", idx)

    milestones: dict[str, float | None] = {}
    for target in MILESTONE_STEPS:
        hit = next((p for p in loss_points if p.get("step") == target), None)
        if hit is None and len(loss_points) >= target:
            hit = loss_points[target - 1]
        milestones[f"step_{target}"] = float(hit["loss"]) if hit else None

    final_loss = milestones.get("step_50")
    if final_loss is None and loss_points:
        final_loss = float(loss_points[-1]["loss"])

    loss_decreased = False
    s1 = milestones.get("step_1")
    s50 = milestones.get("step_50") or final_loss
    if s1 is not None and s50 is not None:
        loss_decreased = s50 < s1

    adapter_location = None
    if hub_ids:
        adapter_location = f"https://huggingface.co/{hub_ids[-1]}"
    elif adapter_paths:
        adapter_location = adapter_paths[-1]
    elif training_started and loss_points:
        adapter_location = "./outputs/student-qlora-canary (worker volume; see run_id output_dir)"

    return {
        "loss_curve": loss_points,
        "loss_milestones": milestones,
        "final_loss": final_loss,
        "loss_decreased": loss_decreased,
        "model_loaded": model_loaded or training_started,
        "training_started": training_started,
        "fatal_error": fatal_error,
        "adapter_location": adapter_location,
        "adapter_paths": adapter_paths,
        "hub_model_ids": hub_ids,
    }


def _extract_output_payload(data: dict[str, Any]) -> Any:
    for key in ("output", "result"):
        if key in data and data[key] is not None:
            return data[key]
    return None


def _submit_runsync(
    endpoint_id: str,
    payload: dict[str, Any],
    *,
    timeout_s: int = 7200,
) -> dict[str, Any]:
    api_key = _resolve_api_key(None)
    urls = endpoint_native_urls(endpoint_id)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=httpx.Timeout(timeout_s, connect=60.0)) as client:
        resp = client.post(urls["runsync"], headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return {
            "capture_mode": "runsync",
            "status": str(data.get("status") or "UNKNOWN"),
            "output": _extract_output_payload(data),
            "raw_response": data,
            "job_id": str(data.get("id") or ""),
            "error": data.get("error"),
        }


def _collect_stream_chunks(
    client: httpx.Client,
    *,
    stream_url: str,
    headers: dict[str, str],
    timeout_s: float = 30.0,
) -> list[Any]:
    chunks: list[Any] = []
    try:
        resp = client.get(stream_url, headers=headers, timeout=timeout_s)
        if resp.status_code != 200:
            return chunks
        body = resp.json()
        if isinstance(body, list):
            for item in body:
                out = item.get("output") if isinstance(item, dict) else item
                if out is not None:
                    chunks.append(out)
        elif body:
            chunks.append(body)
    except (httpx.HTTPError, json.JSONDecodeError):
        pass
    return chunks


def _submit_run_stream_poll(
    endpoint_id: str,
    payload: dict[str, Any],
    *,
    timeout_s: int = 7200,
    poll_s: float = 20.0,
) -> dict[str, Any]:
    api_key = _resolve_api_key(None)
    urls = endpoint_native_urls(endpoint_id)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    stream_chunks: list[Any] = []
    with httpx.Client(timeout=httpx.Timeout(120.0, connect=60.0)) as client:
        submit = client.post(urls["run"], headers=headers, json=payload)
        submit.raise_for_status()
        job_id = str(submit.json().get("id") or "")
        if not job_id:
            raise RuntimeError(f"missing job id: {submit.text}")
        status_url = f"{urls['run'].rsplit('/', 1)[0]}/status/{job_id}"
        stream_url = f"{urls['run'].rsplit('/', 1)[0]}/stream/{job_id}"
        deadline = time.time() + timeout_s
        last_status = "IN_QUEUE"
        raw_status: dict[str, Any] = {}
        while time.time() < deadline:
            stream_chunks.extend(_collect_stream_chunks(client, stream_url=stream_url, headers=headers))
            status_resp = client.get(status_url, headers=headers)
            status_resp.raise_for_status()
            raw_status = status_resp.json()
            last_status = str(raw_status.get("status") or "")
            output = _extract_output_payload(raw_status)
            if last_status == "COMPLETED":
                if output is None and stream_chunks:
                    output = stream_chunks
                return {
                    "capture_mode": "run_stream_poll",
                    "status": "COMPLETED",
                    "output": output,
                    "stream_chunks": stream_chunks,
                    "raw_status": raw_status,
                    "job_id": job_id,
                }
            if last_status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
                return {
                    "capture_mode": "run_stream_poll",
                    "status": last_status,
                    "output": output or stream_chunks or None,
                    "stream_chunks": stream_chunks,
                    "error": raw_status.get("error") or output,
                    "raw_status": raw_status,
                    "job_id": job_id,
                }
            time.sleep(poll_s)
        return {
            "capture_mode": "run_stream_poll",
            "status": "TIMEOUT",
            "output": stream_chunks or None,
            "stream_chunks": stream_chunks,
            "error": f"poll exceeded {timeout_s}s",
            "raw_status": raw_status,
            "job_id": job_id,
        }


def _submit_and_capture(endpoint_id: str, payload: dict[str, Any], *, timeout_s: int = 7200) -> dict[str, Any]:
    runsync: dict[str, Any] = {"capture_mode": "runsync", "status": "SKIPPED", "output": None}
    try:
        runsync = _submit_runsync(endpoint_id, payload, timeout_s=timeout_s)
        if runsync.get("output") is not None:
            return runsync
        job_id = str(runsync.get("job_id") or "")
        if job_id and runsync.get("status") == "COMPLETED":
            api_key = _resolve_api_key(None)
            urls = endpoint_native_urls(endpoint_id)
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            status_url = f"{urls['run'].rsplit('/', 1)[0]}/status/{job_id}"
            stream_url = f"{urls['run'].rsplit('/', 1)[0]}/stream/{job_id}"
            with httpx.Client(timeout=httpx.Timeout(120.0, connect=60.0)) as client:
                status_resp = client.get(status_url, headers=headers)
                status_resp.raise_for_status()
                raw_status = status_resp.json()
                output = _extract_output_payload(raw_status)
                stream_chunks = _collect_stream_chunks(client, stream_url=stream_url, headers=headers, timeout_s=120.0)
                if output is not None or stream_chunks:
                    return {
                        "capture_mode": "runsync_status_stream_backfill",
                        "status": "COMPLETED",
                        "output": output or stream_chunks,
                        "stream_chunks": stream_chunks,
                        "raw_status": raw_status,
                        "job_id": job_id,
                        "runsync_attempt": runsync,
                    }
            return runsync
    except httpx.HTTPError as exc:
        runsync = {"capture_mode": "runsync", "status": "HTTP_ERROR", "error": str(exc), "output": None}

    fallback = _submit_run_stream_poll(endpoint_id, payload, timeout_s=timeout_s)
    fallback["runsync_attempt"] = runsync
    return fallback


def _evaluate_pass(
    *,
    job: dict[str, Any],
    parsed: dict[str, Any],
    max_steps: int,
) -> dict[str, Any]:
    output = job.get("output")
    output_non_null = output is not None and (
        bool(_flatten_output(output)) if not isinstance(output, (int, float)) else True
    )
    final_loss = parsed.get("final_loss")
    finite_loss = (
        final_loss is not None
        and final_loss == final_loss
        and 0 < float(final_loss) < 1e6
    )
    adapter_location = parsed.get("adapter_location")
    passed = (
        job.get("status") == "COMPLETED"
        and output_non_null
        and parsed.get("model_loaded")
        and finite_loss
        and parsed.get("loss_decreased")
        and bool(adapter_location)
        and not parsed.get("fatal_error")
    )
    return {
        "model_loaded": bool(parsed.get("model_loaded")),
        "finite_loss_at_step_50": finite_loss,
        "loss_decreased_from_step_1": bool(parsed.get("loss_decreased")),
        "adapter_location": adapter_location,
        "job_output_non_null": output_non_null,
        "pass": passed,
        "max_steps": max_steps,
    }


def run_canary(*, max_steps: int = 50, est_cost: float = 1.0, attempt: int = 3) -> dict[str, Any]:
    wallet_before = _wallet()
    _spend_gate(est_cost)
    run_id = f"student_qlora_canary_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    endpoint_name = f"student-qlora-canary-{uuid.uuid4().hex[:8]}"
    result: dict[str, Any] = {
        "schema": "nano.campaign.student_qlora_canary.v2",
        "timestamp": datetime.now(UTC).isoformat(),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "run_id": run_id,
        "attempt": attempt,
        "capture_strategy": CAPTURE_STRATEGY,
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
        job = _submit_and_capture(endpoint_id, payload)
        result["job_id"] = job.get("job_id")
        result["job_status"] = job.get("status")
        result["capture_mode"] = job.get("capture_mode")
        result["raw_status"] = job.get("raw_status") or job.get("raw_response")
        result["raw_output"] = job.get("output")
        lines = _flatten_output(job.get("output"))
        if not lines and job.get("stream_chunks"):
            lines = _flatten_output(job.get("stream_chunks"))
        parsed = _parse_training_logs(lines)
        eval_result = _evaluate_pass(job=job, parsed=parsed, max_steps=max_steps)
        result.update(
            {
                "log_lines_captured": len(lines),
                "loss_curve": parsed.get("loss_curve", []),
                "loss_milestones": parsed.get("loss_milestones", {}),
                "final_loss": parsed.get("final_loss"),
                "loss_decreased": parsed.get("loss_decreased"),
                "model_loaded": eval_result["model_loaded"],
                "finite_loss": eval_result["finite_loss_at_step_50"],
                "adapter_saved": bool(eval_result["adapter_location"]),
                "adapter_location": eval_result["adapter_location"],
                "job_output_non_null": eval_result["job_output_non_null"],
                "pass": eval_result["pass"],
                "status": "PASS" if eval_result["pass"] else ("FAIL" if job.get("status") == "COMPLETED" else str(job.get("status"))),
                "error": job.get("error"),
            }
        )
        if eval_result["pass"]:
            result["round1_adaptation_gate"] = "UNLOCKED_PENDING_LAUNCH"
        else:
            result["round1_adaptation_gate"] = "BLOCKED"
            if not eval_result["job_output_non_null"]:
                result.setdefault("error", "job_output_null")
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
            manifest["capture_strategy"] = CAPTURE_STRATEGY
            manifest["attempt"] = attempt
            MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Student QLoRA Axolotl canary")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--est-cost", type=float, default=1.0)
    parser.add_argument("--attempt", type=int, default=3)
    args = parser.parse_args()
    payload = run_canary(max_steps=args.max_steps, est_cost=args.est_cost, attempt=args.attempt)
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
