#!/usr/bin/env bash
# Launch native100 span_port extended pod (US-KS-2 A100 SXM, CUDA gate, ~$2 cap).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

COMMIT_SHA="${WAVE1_COMMIT_SHA:-$(git rev-parse HEAD)}"
DATA_CENTER="${RUNPOD_DATA_CENTER_IDS:-US-KS-2}"
LEDGER="${CAMPAIGN_LEDGER:-artifacts/campaign/spend.json}"
EST_HOURS="${NATIVE_EXTENDED_HOURS:-0.15}"
RATE="${NATIVE_EXTENDED_RATE:-1.59}"
MANIFEST="artifacts/campaign/manifests/native100_span_port_extended_v1.json"
GPU_ID="${NATIVE_GPU_ID:-NVIDIA A100-SXM4-80GB}"
RUN_IDS="${NATIVE_EXTENDED_RUN_IDS:-native100_span_port_s0 native100_span_port_s1}"

python3 scripts/campaign_spend.py --ledger "${LEDGER}" gate --amount "$(python3 -c "print(round(${RATE}*${EST_HOURS},4))")" >/dev/null

CREATE_ARGS=(
  --name "native100-sp-$(date +%Y%m%d%H%M)"
  --template-id runpod-torch-v240
  --gpu-id "${GPU_ID}"
  --data-center-ids "${DATA_CENTER}"
  --cloud-type SECURE
  --container-disk-in-gb 40
  --terminate-after "$(date -u -v+90M '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date -u -d '+90 minutes' '+%Y-%m-%dT%H:%M:%SZ')"
)

MAX_CUDA_ATTEMPTS="${NATIVE_CUDA_ATTEMPTS:-4}"
POD_ID=""
for attempt in $(seq 1 "${MAX_CUDA_ATTEMPTS}"); do
  if ! POD_JSON=$(runpodctl pod create "${CREATE_ARGS[@]}" 2>&1); then
    echo "POD_CREATE_FAILED attempt=${attempt}: ${POD_JSON}" >&2
    if echo "${POD_JSON}" | grep -qiE "no available|out of stock|insufficient|not available"; then
      echo "AWAITING_GPU: no ${GPU_ID} in ${DATA_CENTER}" >&2
      exit 2
    fi
    sleep 15
    continue
  fi
  POD_ID=$(echo "${POD_JSON}" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
  RATE=$(echo "${POD_JSON}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('costPerHr', ${RATE}))")
  echo "cuda_gate_attempt=${attempt} pod=${POD_ID}" >&2
  sleep 50
  if bash scripts/native_cuda_gate.sh "${POD_ID}"; then
    break
  fi
  POD_ID=""
  if [[ "${attempt}" -eq "${MAX_CUDA_ATTEMPTS}" ]]; then
    echo "AWAITING_GPU: CUDA gate failed after ${MAX_CUDA_ATTEMPTS} attempts in ${DATA_CENTER}" >&2
    exit 2
  fi
done

EST_COST=$(python3 -c "print(round(float('${RATE}')*${EST_HOURS},4))")

python3 scripts/campaign_spend.py --ledger "${LEDGER}" commit \
  --lane native_a100 \
  --description "Native100 span_port extended s0+s1" \
  --amount "${EST_COST}" \
  --pod-id "${POD_ID}" \
  --gpu "${GPU_ID}" \
  --rate-hr "${RATE}" >/dev/null

export WAVE1_COMMIT_SHA="${COMMIT_SHA}"
export NATIVE_EXTENDED_RUN_IDS="${RUN_IDS}"
bash scripts/bootstrap_native100_span_port_extended.sh "${POD_ID}"

echo "native_span_port_pod=${POD_ID}"
echo "datacenter=${DATA_CENTER}"
echo "manifest=${MANIFEST}"
echo "commit_sha=${COMMIT_SHA}"
echo "run_ids=${RUN_IDS}"
echo "finish_cmd=bash scripts/finish_native_pod.sh ${POD_ID} native100_span_port_s0"
