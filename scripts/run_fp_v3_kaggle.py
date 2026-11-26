#!/usr/bin/env python3
"""Free Kaggle GPU ($0.00) launcher & artifact collector for NANOLM FP-v3 (Cycle 4).

Enforces strict $0.00 RunPod spend by running on Kaggle Free ``NvidiaTeslaT4`` GPU,
packing {nanolm_holonomy, nanolm_frontier, nanolm_posttrain, nano, fabric} read-only,
running CUDA pytest first, executing ``cycle4_first_principles.py``, and stripping
non-JSON files when pulling artifacts into ``artifacts/nanolm_fp_v3/``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

KERNEL_ID = "hassaneljesr/nanolm-fp-v3"
KERNEL_TITLE = "nanolm-fp-v3"
STAGE_DIR = Path("/tmp/kaggle_nanolm_fp_v3")
PACKAGES = ("nanolm_holonomy", "nanolm_frontier", "nanolm_posttrain", "nano", "fabric")

REMOTE_RUNNER_TEMPLATE = r'''#!/usr/bin/env python3
"""Remote Kaggle GPU runner for NANOLM FP-v3 Cycle-4 Suite (EXP-FP1..EXP-FP4)."""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/tmp/hf_cache"
os.environ["TORCH_EXTENSIONS_DIR"] = "/tmp/torch_ext"
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

GIT_COMMIT = "__GIT_COMMIT__"
RUN_MODE = "__RUN_MODE__"
ARCHIVE_B64 = "__ARCHIVE_B64__"


def main() -> int:
    t0 = time.time()
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA not available on Kaggle kernel")
    cap = torch.cuda.get_device_capability(0)
    if cap < (7, 0):
        raise SystemExit(f"ERROR: CUDA SM {cap} < (7, 0) required")

    n_gpus = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
    vram_gbs = [round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2) for i in range(n_gpus)]
    try:
        import triton
        triton_ver = getattr(triton, "__version__", "unknown")
    except Exception as exc:
        triton_ver = f"unavailable:{exc}"

    print(
        f"[kaggle-fp3] torch={torch.__version__} triton={triton_ver} "
        f"gpus={gpu_names} vram_gb={vram_gbs} sm={cap}",
        flush=True,
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "transformers>=4.51.0",
            "accelerate",
            "sentencepiece",
            "protobuf",
            "pytest",
        ],
        check=True,
    )

    src_dir = Path("/tmp/nanolm_src")
    src_dir.mkdir(parents=True, exist_ok=True)
    raw = base64.b64decode(ARCHIVE_B64.encode("ascii"))
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        tf.extractall(src_dir, filter="data")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_dir)
    os.chdir(src_dir)
    sys.path.insert(0, str(src_dir))

    print("[kaggle-fp3] Running full Cycle-4 pytest suite on CUDA...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "nanolm_holonomy/tests/", "nanolm_posttrain/", "nanolm_frontier/"],
        cwd=str(src_dir),
        env=env,
        check=True,
    )

    out_dir = Path("/kaggle/working")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[kaggle-fp3] Running NANOLM FP-v3 suite (--mode {RUN_MODE})...", flush=True)
    subprocess.run(
        [sys.executable, "-u", "-m", "nanolm_holonomy.cycle4_first_principles", "--mode", RUN_MODE, "--out-dir", str(out_dir)],
        cwd=str(src_dir),
        env=env,
        check=True,
    )

    provenance = {
        "campaign": "nanolm_fp_v3",
        "backend": "kaggle_free_gpu",
        "machine_shape": "NvidiaTeslaT4",
        "gpu_count": n_gpus,
        "gpu_names": gpu_names,
        "gpu_vram_gb": vram_gbs,
        "cuda_capability": list(cap),
        "torch_version": torch.__version__,
        "triton_version": triton_ver,
        "source_commit": GIT_COMMIT,
        "run_mode": RUN_MODE,
        "wall_seconds": round(time.time() - t0, 2),
        "runpod_spend_usd": 0.0,
        "total_usd": 0.0,
    }
    (out_dir / "SPEND_AND_PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"[kaggle-fp3] COMPLETE wall_s={provenance['wall_seconds']} files={sorted(p.name for p in out_dir.glob('*.json'))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _find_kaggle_bin() -> str:
    found = shutil.which("kaggle")
    if found:
        return found
    candidate = Path.home() / ".local" / "bin" / "kaggle"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError("kaggle CLI binary not found in PATH or ~/.local/bin/kaggle")


def _pack_sources(worktree_dir: Path) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for pkg in PACKAGES:
            pkg_dir = worktree_dir / pkg
            if not pkg_dir.is_dir():
                raise FileNotFoundError(f"Missing package directory: {pkg_dir}")
            for path in sorted(pkg_dir.rglob("*")):
                if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
                    continue
                rel = path.relative_to(worktree_dir)
                tf.add(path, arcname=str(rel))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def stage_and_push(worktree_dir: Path, *, run_mode: str = "all") -> int:
    kaggle_bin = _find_kaggle_bin()
    STAGE_DIR.mkdir(parents=True, exist_ok=True)

    commit_proc = subprocess.run(
        ["git", "-C", str(worktree_dir), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    git_commit = commit_proc.stdout.strip()
    archive_b64 = _pack_sources(worktree_dir)

    script_src = (
        REMOTE_RUNNER_TEMPLATE
        .replace("__GIT_COMMIT__", git_commit)
        .replace("__RUN_MODE__", run_mode)
        .replace("__ARCHIVE_B64__", archive_b64)
    )
    (STAGE_DIR / "run_kaggle_fp_v3.py").write_text(script_src, encoding="utf-8")

    metadata = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": "run_kaggle_fp_v3.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu"],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (STAGE_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[stage] Packed {len(archive_b64)} b64 bytes (commit={git_commit}, mode={run_mode}) -> {STAGE_DIR}")

    proc = subprocess.run([kaggle_bin, "kernels", "push", "-p", str(STAGE_DIR)], check=False)
    return int(proc.returncode)


def check_status() -> int:
    kaggle_bin = _find_kaggle_bin()
    proc = subprocess.run([kaggle_bin, "kernels", "status", KERNEL_ID], check=False)
    return int(proc.returncode)


def pull_outputs(worktree_dir: Path) -> int:
    kaggle_bin = _find_kaggle_bin()
    out_dir = worktree_dir / "artifacts" / "nanolm_fp_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [kaggle_bin, "kernels", "output", KERNEL_ID, "-p", str(out_dir)],
        check=False,
    )
    if proc.returncode != 0:
        return int(proc.returncode)
    diag_dir = Path("/tmp/kaggle_fp_v3_diag")
    diag_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("*.log", "*.html"):
        for f in out_dir.glob(ext):
            shutil.copy2(f, diag_dir / f.name)
            f.unlink()
    print(f"[pull] Output JSON files in {out_dir}:")
    for p in sorted(out_dir.iterdir()):
        print(f"  - {p.name} ({p.stat().st_size} bytes)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage NANOLM FP-v3 Free Kaggle GPU kernel")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage-and-push", action="store_true")
    group.add_argument("--status", action="store_true")
    group.add_argument("--pull", action="store_true")
    parser.add_argument("--run-mode", choices=("all", "fp1", "fp2", "fp3", "fp4", "fp123", "fp23"), default="all")
    args = parser.parse_args()

    worktree_dir = Path(__file__).resolve().parent.parent
    if args.stage_and_push:
        return stage_and_push(worktree_dir, run_mode=args.run_mode)
    if args.status:
        return check_status()
    return pull_outputs(worktree_dir)


if __name__ == "__main__":
    raise SystemExit(main())
