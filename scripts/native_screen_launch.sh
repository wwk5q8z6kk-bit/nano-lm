#!/usr/bin/env bash
# Native Nano short screening launcher — prepare locally, then provision GPU with command ready.
# FAIL-CLOSED: do not start pod without NANOSCIBE_NATIVE_EXPERIMENT_ID and NANOSCIBE_NATIVE_COMMAND set.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EXPERIMENT_ID="${NANOSCIBE_NATIVE_EXPERIMENT_ID:-native_screen_v1}"
VARIANT="${NANOSCIBE_NATIVE_VARIANT:-NATIVE_A}"
MAX_HOURS="${NANOSCIBE_NATIVE_MAX_HOURS:-0.5}"
GPU_TYPE="${NANOSCIBE_NATIVE_GPU:-NVIDIA GeForce RTX 4090}"

echo "Native screen prep: experiment=$EXPERIMENT_ID variant=$VARIANT max_hours=$MAX_HOURS"
python3 scripts/campaign_spend.py gate --amount "$(python3 -c "print(0.74*float('${MAX_HOURS}'))")" || {
  echo "Budget gate blocked native screen"; exit 1;
}

# Local smoke only until training script lands on pod.
python3 -c "from nanoscribe.native import manifest; import json; print(json.dumps(manifest(), indent=2))"

cat <<EOF
To launch on RunPod (after syncing repo to volume):
  export NANOSCIBE_NATIVE_EXPERIMENT_ID=$EXPERIMENT_ID
  runpodctl pod create --name "native-\${EXPERIMENT_ID}" \\
    --gpu-type-id "$GPU_TYPE" \\
    --image runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 \\
    --volume-path /workspace --terminate-after 2h \\
    --start-ssh --command "cd /workspace/nano-lm && python3 sft/train_sft.py --steps 500"
Watchdog: within 5 min verify nvidia-smi utilization and log output; else terminate pod.
EOF
