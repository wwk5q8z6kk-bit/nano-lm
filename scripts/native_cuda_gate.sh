#!/usr/bin/env bash
# Quick CUDA availability check on a running pod — delete if torch.cuda unavailable.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
POD_ID="${1:?pod id required}"
OUT=$(bash scripts/runpod_pod_ssh.sh "${POD_ID}" \
  "python3 -c \"import torch, json; print(json.dumps({'cuda': torch.cuda.is_available(), 'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))\"" 2>&1)
echo "${OUT}"
if echo "${OUT}" | grep -q '"cuda": true'; then
  echo "CUDA_OK"
  exit 0
fi
echo "CUDA_FAIL: terminating pod ${POD_ID}" >&2
runpodctl pod delete "${POD_ID}" >/dev/null
exit 1
