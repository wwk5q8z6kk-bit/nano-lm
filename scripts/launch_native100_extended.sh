#!/usr/bin/env bash
# Launch native100 extended pod (default entrypoint, bootstrap via SSH).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

COMMIT_SHA="${WAVE1_COMMIT_SHA:-$(git rev-parse HEAD)}"
DATA_CENTER="${RUNPOD_DATA_CENTER_IDS:-CA-MTL-3}"
VOLUME_ID="${RUNPOD_VOLUME_ID:-}"
LEDGER="${CAMPAIGN_LEDGER:-artifacts/campaign/spend.json}"
EST_HOURS="${NATIVE_EXTENDED_HOURS:-0.75}"
RATE="${NATIVE_EXTENDED_RATE:-1.39}"
MANIFEST="artifacts/campaign/manifests/native100_extended_v1.json"

python3 scripts/campaign_spend.py --ledger "${LEDGER}" gate --amount "$(python3 -c "print(round(${RATE}*${EST_HOURS},4))")" >/dev/null

CREATE_ARGS=(
  --name "native100-ext-$(date +%Y%m%d%H%M)"
  --template-id runpod-torch-v240
  --gpu-id "NVIDIA A100 80GB PCIe"
  --data-center-ids "${DATA_CENTER}"
  --cloud-type SECURE
  --container-disk-in-gb 40
  --terminate-after "$(date -u -v+2H '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date -u -d '+2 hours' '+%Y-%m-%dT%H:%M:%SZ')"
)
if [[ -n "${VOLUME_ID}" ]]; then
  CREATE_ARGS+=(--network-volume-id "${VOLUME_ID}" --volume-mount-path /workspace)
fi

POD_JSON=$(runpodctl pod create "${CREATE_ARGS[@]}")
POD_ID=$(echo "${POD_JSON}" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
RATE=$(echo "${POD_JSON}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('costPerHr', ${RATE}))")
EST_COST=$(python3 -c "print(round(float('${RATE}')*${EST_HOURS},4))")

python3 scripts/campaign_spend.py --ledger "${LEDGER}" commit \
  --lane native_a100 \
  --description "Native100 extended evidence_bottleneck_s1" \
  --amount "${EST_COST}" \
  --pod-id "${POD_ID}" \
  --gpu "NVIDIA A100 80GB PCIe" \
  --rate-hr "${RATE}" >/dev/null

echo "native_extended_pod=${POD_ID}"
echo "datacenter=${DATA_CENTER}"
echo "manifest=${MANIFEST}"
echo "commit_sha=${COMMIT_SHA}"
