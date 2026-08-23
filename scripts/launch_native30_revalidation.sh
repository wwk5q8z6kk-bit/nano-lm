#!/usr/bin/env bash
# Launch native100 extended pod (default entrypoint, bootstrap via SSH).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

COMMIT_SHA="${WAVE1_COMMIT_SHA:-$(git rev-parse HEAD)}"
DATA_CENTER="${RUNPOD_DATA_CENTER_IDS:-CA-MTL-3}"
VOLUME_ID="${RUNPOD_VOLUME_ID:-}"
LEDGER="${CAMPAIGN_LEDGER:-artifacts/campaign/spend.json}"
EST_HOURS="${NATIVE_REVAL_HOURS:-1.5}"
RATE="${NATIVE_REVAL_RATE:-0.27}"
MANIFEST="artifacts/campaign/manifests/native30_revalidation_wave1_v1.json"
# Right-sized from MEASURED footprint: a 30M arm peaked at 8.2GB of an A100's
# 80GB — a 9.8x over-provision at $1.39-1.59/hr. An RTX A5000 (24GB, $0.27/hr)
# still leaves 2.9x headroom for eval and fragmentation, cutting the 9-run screen
# from ~$2.23 to ~$0.38. Override with NATIVE_GPU_ID for larger arms.
GPU_ID="${NATIVE_GPU_ID:-NVIDIA RTX A5000}"

python3 scripts/campaign_spend.py --ledger "${LEDGER}" gate --amount "$(python3 -c "print(round(${RATE}*${EST_HOURS},4))")" >/dev/null

CREATE_ARGS=(
  --name "native30-reval-$(date +%Y%m%d%H%M)"
  --template-id runpod-torch-v240
  --gpu-id "${GPU_ID}"
  --data-center-ids "${DATA_CENTER}"
  --cloud-type SECURE
  --container-disk-in-gb 40
  --terminate-after "$(date -u -v+2H '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date -u -d '+2 hours' '+%Y-%m-%dT%H:%M:%SZ')"
)
if [[ -n "${VOLUME_ID}" ]]; then
  CREATE_ARGS+=(--network-volume-id "${VOLUME_ID}" --volume-mount-path /workspace)
fi

MAX_CUDA_ATTEMPTS="${NATIVE_CUDA_ATTEMPTS:-4}"
POD_ID=""
for attempt in $(seq 1 "${MAX_CUDA_ATTEMPTS}"); do
  POD_JSON=$(runpodctl pod create "${CREATE_ARGS[@]}")
  POD_ID=$(echo "${POD_JSON}" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
  RATE=$(echo "${POD_JSON}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('costPerHr', ${RATE}))")
  echo "cuda_gate_attempt=${attempt} pod=${POD_ID}" >&2
  sleep 50
  if bash scripts/native_cuda_gate.sh "${POD_ID}"; then
    break
  fi
  POD_ID=""
  if [[ "${attempt}" -eq "${MAX_CUDA_ATTEMPTS}" ]]; then
    echo "CUDA_GATE_FAILED: no pod with working torch.cuda after ${MAX_CUDA_ATTEMPTS} attempts" >&2
    exit 1
  fi
done

EST_COST=$(python3 -c "print(round(float('${RATE}')*${EST_HOURS},4))")

python3 scripts/campaign_spend.py --ledger "${LEDGER}" commit \
  --lane native_a100 \
  --description "Native30 revalidation wave1 (9 runs)" \
  --amount "${EST_COST}" \
  --pod-id "${POD_ID}" \
  --gpu "${GPU_ID}" \
  --rate-hr "${RATE}" >/dev/null

echo "native_extended_pod=${POD_ID}"
echo "datacenter=${DATA_CENTER}"
echo "manifest=${MANIFEST}"
echo "commit_sha=${COMMIT_SHA}"
echo "finish_cmd=bash scripts/finish_native_pod.sh ${POD_ID} native100_evidence_bottleneck_s1"
