#!/usr/bin/env bash
# Terminate a RunPod pod when GPU utilization stays below threshold after bootstrap.
# Usage: bash scripts/gpu_idle_watchdog.sh <pod_id> [bootstrap_sec] [idle_min] [util_threshold]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

POD_ID="${1:?pod id required}"
BOOTSTRAP_SEC="${2:-120}"
IDLE_MIN="${3:-5}"
UTIL_THRESHOLD="${4:-5}"
POLL_SEC="${GPU_IDLE_POLL_SEC:-60}"
SSH="${ROOT}/scripts/runpod_pod_ssh.sh"

echo "gpu_idle_watchdog pod=${POD_ID} bootstrap=${BOOTSTRAP_SEC}s idle=${IDLE_MIN}min threshold=${UTIL_THRESHOLD}%"
sleep "${BOOTSTRAP_SEC}"

idle_streak=0
required_streak=$((IDLE_MIN * 60 / POLL_SEC))
if [[ "${required_streak}" -lt 1 ]]; then
  required_streak=1
fi

while true; do
  OUT=$(bash "${SSH}" "${POD_ID}" \
    "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1" 2>&1 || true)
  UTIL=$(echo "${OUT}" | tr -dc '0-9' | head -c 3)
  if [[ -z "${UTIL}" ]]; then
    echo "watchdog poll: nvidia-smi unavailable (${OUT})"
    idle_streak=0
  elif [[ "${UTIL}" -lt "${UTIL_THRESHOLD}" ]]; then
    idle_streak=$((idle_streak + 1))
    echo "watchdog poll: util=${UTIL}% idle_streak=${idle_streak}/${required_streak}"
  else
    idle_streak=0
    echo "watchdog poll: util=${UTIL}% (active)"
  fi

  if [[ "${idle_streak}" -ge "${required_streak}" ]]; then
    echo "IDLE_BURN_ABORT pod=${POD_ID} util<${UTIL_THRESHOLD}% for ${IDLE_MIN}min — terminating"
    runpodctl pod delete "${POD_ID}" || true
    exit 2
  fi
  sleep "${POLL_SEC}"
done
