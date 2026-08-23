#!/usr/bin/env bash
# P1 serverless lane launcher — deploy or smoke-test RunPod Serverless endpoints.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

HUB_VLLM="${RUNPOD_VLLM_HUB_ID:-cm8h09d9n000008jvh2rqdsmb}"
MODEL="${SERVERLESS_MODEL:-Qwen/Qwen3.8-27B}"
GPU="${SERVERLESS_GPU:-NVIDIA A100 80GB PCIe}"
ENDPOINT_NAME="${SERVERLESS_ENDPOINT_NAME:-p1-qwen38-27b-strong-control}"
WORKERS_MIN="${WORKERS_MIN:-0}"
WORKERS_MAX="${WORKERS_MAX:-2}"
LEDGER="${CAMPAIGN_LEDGER:-artifacts/campaign/spend.json}"
ACTION="${1:-smoke}"

case "${ACTION}" in
  deploy)
    echo "==> Deploying ${MODEL} on ${GPU} (workers ${WORKERS_MIN}-${WORKERS_MAX})"
    runpodctl serverless create \
      --hub-id "${HUB_VLLM}" \
      --gpu-id "${GPU}" \
      --model-reference "https://huggingface.co/${MODEL}:main" \
      --name "${ENDPOINT_NAME}" \
      --workers-min "${WORKERS_MIN}" \
      --workers-max "${WORKERS_MAX}" \
      --idle-timeout 300 \
      --env "MODEL_NAME=${MODEL}" \
      --env TRUST_REMOTE_CODE=true \
      --env DTYPE=bfloat16
    ;;
  smoke)
    ENDPOINT_ID="${RUNPOD_SERVERLESS_ENDPOINT_ID:-tbnur4mac60i70}"
    echo "==> Smoke test endpoint ${ENDPOINT_ID}"
    python3 scripts/p1_serverless_smoke.py --endpoint "${ENDPOINT_ID}" --model "${MODEL}"
    ;;
  *)
    echo "Usage: $0 deploy|smoke" >&2
    exit 1
    ;;
esac
