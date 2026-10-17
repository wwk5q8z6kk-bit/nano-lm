#!/usr/bin/env bash
# P1 acceleration campaign launcher — prep-first, cost-gated RunPod lanes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

VOLUME_ID="${RUNPOD_VOLUME_ID:-04himzqxbm}"
DC="${RUNPOD_DC:-EU-RO-1}"
IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
LEDGER="${CAMPAIGN_LEDGER:-artifacts/campaign/spend.json}"
LANE="${1:-compact}"

if [[ "${LANE}" == "compact" ]]; then
  echo "ERROR: compact lane raw 4090 inference superseded by RunPod Serverless Qwen3.8-27B (owner: 82a04724). Use student lane or serverless branch." >&2
  exit 1
fi
EST_HOURS="${EST_HOURS:-0.5}"

# GPU pricing (secure cloud, USD/hr) — update from `runpodctl gpu list`.
GPU_RATE_4090=0.74
GPU_RATE_A100=1.59
GPU_RATE_L40S=0.99

case "${LANE}" in
  compact)
    GPU="NVIDIA GeForce RTX 4090"
    RATE="${GPU_RATE_4090}"
  ;;
  student)
    GPU="NVIDIA A100-SXM4-80GB"
    RATE="${GPU_RATE_A100}"
  ;;
  *)
    echo "Unknown lane: ${LANE} (use compact|student)" >&2
    exit 1
  ;;
esac

EST_COST=$(python3 -c "print(round(${RATE} * ${EST_HOURS}, 4))")
echo "==> Lane ${LANE}: ${GPU} @ \$${RATE}/hr × ${EST_HOURS}h ≈ \$${EST_COST}"

GATE=$(python3 scripts/campaign_spend.py --ledger "${LEDGER}" gate --amount "${EST_COST}")
ALLOWED=$(echo "${GATE}" | python3 -c "import sys,json; print(json.load(sys.stdin)['allowed'])")
if [[ "${ALLOWED}" != "True" ]]; then
  echo "BUDGET GATE BLOCKED: ${GATE}" >&2
  exit 2
fi

POD_NAME="p1-accel-${LANE}-$(date +%Y%m%d%H%M)"
echo "==> Creating pod ${POD_NAME}"
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

python3 scripts/campaign_spend.py --ledger "${LEDGER}" commit \
  --lane "${LANE}" \
  --description "RunPod ${LANE} baseline harness" \
  --amount "${EST_COST}" \
  --pod-id "${POD_ID}" \
  --gpu "${GPU}" \
  --rate-hr "${RATE}" > /dev/null

echo "==> Waiting for RUNNING..."
for _ in $(seq 1 60); do
  STATUS=$(runpodctl pod get "${POD_ID}" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('desiredStatus',''))" || echo "")
  if [[ "${STATUS}" == "RUNNING" ]]; then break; fi
  sleep 10
done

SSH_CMD=$(runpodctl pod get "${POD_ID}" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(d.get('ssh',{}).get('ssh_command') or d.get('sshCmd',''))
")
echo "==> SSH: ${SSH_CMD}"

TRACKS="compact"
WEIGHTS="Qwen/Qwen2.5-1.5B-Instruct"
if [[ "${LANE}" == "student" ]]; then
  TRACKS="student"
  WEIGHTS="Qwen/Qwen2.5-32B-Instruct"
fi

REMOTE_SCRIPT=$(cat <<EOS
set -euo pipefail
cd /workspace/nano-lm || { cd /workspace && git clone https://github.com/wwk5q8z6kk-bit/nano-lm.git && cd nano-lm; }
git fetch origin --no-tags
git checkout frontier/p1-acceleration-campaign-v0 || git checkout frontier/p1-qwen-baseline-smoke-v0
pip install -q -r requirements.txt
export HF_HOME=/workspace/hf_cache
export NANOSCIBE_QWEN_WEIGHTS=${WEIGHTS}
mkdir -p /workspace/p1_runs
python3 scripts/p1_harness.py --tracks ${TRACKS} --output /workspace/p1_runs/accel_${LANE}.json --capture-raw
python3 nanoscribe/smoke_qwen_baseline.py 2>/dev/null || true
EOS
)

echo "==> Run remotely:"
echo "${SSH_CMD} bash -lc $(printf %q "${REMOTE_SCRIPT}")"

echo "${POD_ID}" > "/tmp/p1_accel_${LANE}_pod_id.txt"
echo "==> When done: runpodctl pod delete ${POD_ID}"
echo "==> Then: python3 scripts/campaign_spend.py actual --lane ${LANE} --amount <actual>"
