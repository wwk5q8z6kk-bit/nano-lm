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
rsync -a artifacts/native_checkpoints/ /workspace/campaign_native_checkpoints/
echo NATIVE100_DONE
