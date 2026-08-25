#!/usr/bin/env bash
# Remote entrypoint for native100 round2 — run inside pod after sync.
set -euo pipefail
cd /workspace/nano-lm
export HF_HOME=/workspace/hf_cache
python3 scripts/train_native_nano.py --export-train-json
python3 -m nanoscribe.runpod_gpu_preflight
for rid in native100_evidence_bottleneck_s0 native100_evidence_bottleneck_s1 native100_span_port_s0 native100_span_port_s1; do
  echo "==> train ${rid}"
  python3 scripts/train_native_nano.py --run-id "${rid}"
done
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
echo NATIVE100_DONE
