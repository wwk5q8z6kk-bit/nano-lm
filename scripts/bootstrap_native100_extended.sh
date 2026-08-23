#!/usr/bin/env bash
# Bootstrap native100 extended training on a running pod.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

POD_ID="${1:?pod id required}"
COMMIT_SHA="${WAVE1_COMMIT_SHA:-$(git rev-parse HEAD)}"
MANIFEST="artifacts/campaign/manifests/native100_extended_v1.json"

BOOT="set -e; cd /workspace/nano-lm 2>/dev/null || { cd /workspace && git clone https://github.com/wwk5q8z6kk-bit/nano-lm.git && cd nano-lm; }; git fetch origin frontier/accelerated-research-campaign-v2 && git checkout ${COMMIT_SHA}; pip install -q -r requirements.txt; export HF_HOME=/workspace/hf_cache; nohup bash scripts/native100_extended_remote.sh > /workspace/native100_extended.log 2>&1 & echo BOOTSTRAP_STARTED"

echo "==> Bootstrapping extended native100 on ${POD_ID} @ ${COMMIT_SHA}"
echo "==> Manifest: ${MANIFEST}"
bash scripts/runpod_pod_ssh.sh "${POD_ID}" "${BOOT}"
sleep 8
bash scripts/runpod_pod_ssh.sh "${POD_ID}" "tail -15 /workspace/native100_extended.log 2>/dev/null || echo log_pending; pgrep -af train_native || echo no_train; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
