"""CUDA GPU vs PyTorch arch preflight — block sm_100 pods on sm_90-only wheels."""

from __future__ import annotations

import sys
from typing import Any


def _max_sm_from_arch_list(arch_list: list[str]) -> int:
    best = 0
    for name in arch_list:
        if not name.startswith("sm_"):
            continue
        try:
            best = max(best, int(name.split("_", 1)[1]))
        except ValueError:
            continue
    return best


def device_sm_version(major: int, minor: int) -> int:
    """Map torch.cuda.get_device_capability() to sm_XX integer (e.g. 90, 100)."""
    return major * 10 + minor


def check_cuda_torch_compatible() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        return {"ok": False, "reason": f"torch not installed: {exc}"}

    if not torch.cuda.is_available():
        return {"ok": False, "reason": "CUDA not available"}

    major, minor = torch.cuda.get_device_capability()
    device_sm = device_sm_version(major, minor)
    arch_list = list(torch.cuda.get_arch_list())
    max_supported = _max_sm_from_arch_list(arch_list)
    name = torch.cuda.get_device_name(0)

    ok = device_sm <= max_supported
    return {
        "ok": ok,
        "device_name": name,
        "device_capability": [major, minor],
        "device_sm": device_sm,
        "torch_arch_list": arch_list,
        "max_supported_sm": max_supported,
        "reason": (
            None
            if ok
            else (
                f"GPU sm_{device_sm} ({name}) exceeds PyTorch build max sm_{max_supported}; "
                "use H100/A100 or PyTorch with Blackwell (sm_100) support"
            )
        ),
    }


def assert_cuda_torch_compatible() -> None:
    result = check_cuda_torch_compatible()
    if not result["ok"]:
        raise RuntimeError(result.get("reason") or "CUDA preflight failed")


def block_b200_without_sm100(gpu_id: str, image_or_template: str) -> None:
    """Raise before pod create when B200 would use legacy PyTorch images."""
    gpu_upper = gpu_id.upper()
    if "B200" not in gpu_upper:
        return
    legacy_markers = ("2.4.0", "2.2.0", "2.1.0", "cuda12.4", "runpod-torch-v240")
    blob = image_or_template.lower()
    if any(m.lower() in blob for m in legacy_markers):
        raise ValueError(
            f"Refusing B200 launch with sm_90-only surface {image_or_template!r}; "
            "use H100/A100 or runpod/pytorch cu128 / 2.8+ with sm_100 preflight"
        )


def main() -> int:
    result = check_cuda_torch_compatible()
    import json

    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
