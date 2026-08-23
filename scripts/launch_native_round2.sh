#!/usr/bin/env bash
# Native Round 2 — 4× 100M promotion runs on single A100 (volume-backed archive).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

COMMIT_SHA="${WAVE1_COMMIT_SHA:-$(git rev-parse HEAD)}"
VOLUME_ID="${RUNPOD_VOLUME_ID:-04himzqxbm}"
DATA_CENTER_IDS="${RUNPOD_DATA_CENTER_IDS:-EU-RO-1}"
PYTORCH_TEMPLATE_ID="${RUNPOD_PYTORCH_TEMPLATE:-runpod-torch-v240}"
LEDGER="${CAMPAIGN_LEDGER:-artifacts/campaign/spend.json}"
EST_HOURS="${NATIVE_ROUND2_HOURS:-1.0}"
RATE="${NATIVE_ROUND2_RATE:-1.39}"

RUN_IDS=(
  native100_evidence_bottleneck_s0
  native100_evidence_bottleneck_s1
  native100_span_port_s0
  native100_span_port_s1
)

python3 scripts/campaign_spend.py --ledger "${LEDGER}" gate --amount "$(python3 -c "print(round(${RATE}*${EST_HOURS},4))")" >/dev/null

SYNC_CMD="cd /workspace/nano-lm 2>/dev/null || { cd /workspace && git clone https://github.com/wwk5q8z6kk-bit/nano-lm.git && cd nano-lm; }; git fetch origin && git checkout ${COMMIT_SHA}; pip install -q -r requirements.txt; export HF_HOME=/workspace/hf_cache"
PREFLIGHT_CMD="python3 -m nanoscribe.runpod_gpu_preflight"
TERMINATE_AFTER="${NATIVE_ROUND2_TERMINATE_AFTER:-$(date -u -v+3H '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date -u -d '+3 hours' '+%Y-%m-%dT%H:%M:%SZ')}"
NATIVE_ARCHIVE='mkdir -p /workspace/campaign_native_checkpoints && rsync -a artifacts/native_checkpoints/ /workspace/campaign_native_checkpoints/'
REMOTE="python3 scripts/train_native_nano.py --export-train-json; for rid in ${RUN_IDS[*]}; do python3 scripts/train_native_nano.py --run-id \"\$rid\" || exit 1; done; ${NATIVE_ARCHIVE}; echo NATIVE100_DONE"

TS=$(date +%Y%m%d%H%M)
POD_JSON=$(runpodctl pod create \
  --name "native100-round2-${TS}" \
  --template-id "${PYTORCH_TEMPLATE_ID}" \
  --gpu-id "NVIDIA A100 80GB PCIe" \
  --data-center-ids "${DATA_CENTER_IDS}" \
  --cloud-type SECURE \
  --network-volume-id "${VOLUME_ID}" \
  --container-disk-in-gb 40 \
  --volume-mount-path /workspace \
  --terminate-after "${TERMINATE_AFTER}" \
  --docker-args "bash -lc $(printf %q "${SYNC_CMD}; ${PREFLIGHT_CMD} || exit 1; ${REMOTE}")")

POD_ID=$(echo "${POD_JSON}" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
RATE=$(echo "${POD_JSON}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('costPerHr', ${RATE}))")
EST_COST=$(python3 -c "print(round(float('${RATE}')*${EST_HOURS},4))")

python3 scripts/campaign_spend.py --ledger "${LEDGER}" commit \
  --lane native_a100 \
  --description "Native round2 100M x4 promotion" \
  --amount "${EST_COST}" \
  --pod-id "${POD_ID}" \
  --gpu "NVIDIA A100 80GB PCIe" \
  --rate-hr "${RATE}" >/dev/null

echo "native_round2_pod=${POD_ID}"
echo "commit_sha=${COMMIT_SHA}"
echo "run_ids=${RUN_IDS[*]}"
