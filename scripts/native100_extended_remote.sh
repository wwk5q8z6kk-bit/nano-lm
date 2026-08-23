#!/usr/bin/env bash
# Remote entrypoint for native100 extended — single primary run @ 200 steps.
set -euo pipefail
cd /workspace/nano-lm
export HF_HOME=/workspace/hf_cache
RUN_ID="${NATIVE_EXTENDED_RUN_ID:-native100_evidence_bottleneck_s1}"
python3 scripts/train_native_nano.py --export-train-json
if ! python3 -m nanoscribe.runpod_gpu_preflight; then
  echo "PREFLIGHT_WARN: preflight failed; checking nvidia-smi"
  nvidia-smi || { echo "PREFLIGHT_FATAL: no GPU"; exit 1; }
  echo "PREFLIGHT_OVERRIDE: nvidia-smi ok, continuing"
fi
echo "==> train ${RUN_ID} (max_steps=200)"
python3 scripts/train_native_nano.py --run-id "${RUN_ID}"
mkdir -p /workspace/campaign_native_checkpoints
python3 - <<'PY'
import shutil
from pathlib import Path
src = Path("artifacts/native_checkpoints")
dst = Path("/workspace/campaign_native_checkpoints")
dst.mkdir(parents=True, exist_ok=True)
for run_dir in src.glob("native100_*"):
    if run_dir.is_dir():
        shutil.copytree(run_dir, dst / run_dir.name, dirs_exist_ok=True)
print("ARCHIVE_OK", dst)
PY
echo NATIVE100_EXTENDED_DONE
