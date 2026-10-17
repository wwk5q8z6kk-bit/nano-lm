#!/usr/bin/env bash
# Bootstrap native100 round2 on an already-running pod (default entrypoint, volume mounted).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

POD_ID="${1:?pod id required}"
COMMIT_SHA="${WAVE1_COMMIT_SHA:-2094214}"
MANIFEST="artifacts/campaign/manifests/native100_round2_promotion_v1.json"

BOOT="set -e; cd /workspace/nano-lm 2>/dev/null || { cd /workspace && git clone https://github.com/wwk5q8z6kk-bit/nano-lm.git && cd nano-lm; }; git fetch origin && git checkout ${COMMIT_SHA}; pip install -q -r requirements.txt; export HF_HOME=/workspace/hf_cache; command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync); nohup bash scripts/native_round2_remote.sh > /workspace/native100_train.log 2>&1 & echo BOOTSTRAP_STARTED"

echo "==> Bootstrapping native100 on pod ${POD_ID} @ ${COMMIT_SHA}"
echo "==> Manifest: ${MANIFEST}"
echo "==> After training completes, run: bash scripts/finish_native_pod.sh ${POD_ID} <run_id> (marker=NATIVE100_DONE)"
bash scripts/runpod_pod_ssh.sh "${POD_ID}" "${BOOT}"
sleep 5
bash scripts/runpod_pod_ssh.sh "${POD_ID}" "tail -20 /workspace/native100_train.log 2>/dev/null || echo log_pending; pgrep -af train_native || echo no_train_yet; nvidia-smi --query-gpu=utilization.gpu,name --format=csv,noheader"
