"""Native Nano checkpoint save/load."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nanoscribe.native.config import NativeTrainConfig, checkpoint_path


def save_checkpoint(
    model: Any,
    cfg: NativeTrainConfig,
    *,
    step: int,
    optimizer: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    import torch

    out = checkpoint_path(cfg, step)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "nano.native.checkpoint.v1",
        "timestamp": datetime.now(UTC).isoformat(),
        "step": step,
        "config": cfg.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "extra": extra or {},
    }
    torch.save(payload, out)
    latest = checkpoint_path(cfg)
    torch.save(payload, latest)
    meta = out.with_suffix(".json")
    meta_payload = {
        "schema": payload["schema"],
        "timestamp": payload["timestamp"],
        "step": payload["step"],
        "config": payload["config"],
        "extra": payload["extra"],
    }
    meta.write_text(json.dumps(meta_payload, indent=2) + "\n")
    return out


def load_checkpoint(
    cfg: NativeTrainConfig,
    model: Any,
    *,
    step: int | None = None,
    optimizer: Any | None = None,
) -> dict[str, Any]:
    import torch

    path = checkpoint_path(cfg, step)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state"):
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload
