#!/usr/bin/env bash
# P1 RunPod smoke: provision 4090 pod, cache Qwen2.5-1.5B, run compact harness.
set -euo pipefail

REPO="${NANO_REPO:-/workspace/nano-lm}"
VOLUME_ID="${RUNPOD_VOLUME_ID:-04himzqxbm}"
DC="${RUNPOD_DC:-EU-RO-1}"
GPU="${RUNPOD_GPU:-NVIDIA GeForce RTX 4090}"
IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
POD_NAME="p1-three-track-smoke-$(date +%Y%m%d%H%M)"

echo "==> Creating pod ${POD_NAME} on ${GPU} with volume ${VOLUME_ID}"
POD_JSON=$(runpodctl pod create \
  --name "${POD_NAME}" \
  --image "${IMAGE}" \
  --gpu-id "${GPU}" \
  --cloud-type SECURE \
  --network-volume-id "${VOLUME_ID}" \
  --container-disk-in-gb 20 \
  --volume-mount-path /workspace \
  --ports "22/tcp")

POD_ID=$(echo "${POD_JSON}" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "==> Pod ID: ${POD_ID}"

echo "==> Waiting for pod to be RUNNING..."
for _ in $(seq 1 60); do
  STATUS=$(runpodctl pod get "${POD_ID}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('desiredStatus',''))")
  if [[ "${STATUS}" == "RUNNING" ]]; then break; fi
  sleep 10
done

SSH_CMD=$(runpodctl pod get "${POD_ID}" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(d.get('sshCmd',''))
")
echo "==> SSH: ${SSH_CMD}"

REMOTE_SCRIPT=$(cat <<'EOS'
set -euo pipefail
cd /workspace
if [[ ! -d nano-lm ]]; then
  git clone https://github.com/$(git -C /workspace/nano-lm remote get-url origin 2>/dev/null | sed -n 's#.*github.com[:/]\(.*\)\.git#\1#p' || echo 'OWNER/nano-lm') nano-lm || true
fi
cd nano-lm
pip install -q -r requirements.txt
export HF_HOME=/workspace/hf_cache
export NANOSCIBE_QWEN_WEIGHTS=Qwen/Qwen2.5-1.5B-Instruct
export NANOSCIBE_SMOKE_OUT=/workspace/p1_runs/compact_smoke.json
mkdir -p /workspace/p1_runs
python3 nanoscribe/smoke_qwen_baseline.py
python3 scripts/p1_harness.py --tracks compact --output /workspace/p1_runs/harness_compact.json --capture-raw
EOS
)

echo "==> Running remote smoke (manual SSH if auto fails):"
echo "${SSH_CMD} bash -lc $(printf %q "${REMOTE_SCRIPT}")"

echo "==> Terminate when done: runpodctl pod delete ${POD_ID}"
echo "${POD_ID}" > /tmp/p1_runpod_pod_id.txt
