#!/usr/bin/env bash
# 5-minute GPU utilization fail-fast: terminate pod if util stays <5%.
#
# SSH (RunPod):
#   runpodctl ssh info <pod_id>  # direct: ssh -i ~/.runpod/ssh/runpodctl-ssh-key root@<ip> -p <port>
#   When direct TCP times out during startup, use the UI proxy:
#     ssh -i ~/.runpod/ssh/runpodctl-ssh-key <pod_id>-<ui_token>@ssh.runpod.io
#   Example (owner handoff): yglu6kr1ys0kw3-64410f1c@ssh.runpod.io — ui_token is NOT the public port.
#   Non-interactive checks: RUNPOD_SSH_PROXY_SUFFIX=<ui_token> scripts/runpod_pod_ssh.sh <pod_id> 'nvidia-smi'
#   ~/.ssh/id_ed25519 only works if that key is registered on the RunPod account / pod PUBLIC_KEY.
set -euo pipefail
POD_ID="${1:?pod id required}"
RATE_HR="${2:-0}"
LOG_DIR="${3:-artifacts/campaign/health_gates}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${POD_ID}_$(date +%Y%m%dT%H%M%SZ).log"
MAX_WAIT=300
INTERVAL=30
THRESH=5

echo "health_gate pod=$POD_ID max_wait=${MAX_WAIT}s threshold=${THRESH}%" | tee "$LOG"
elapsed=0
max_util=0
while [[ "$elapsed" -lt "$MAX_WAIT" ]]; do
  if ! runpodctl pod get "$POD_ID" -o json >>"$LOG" 2>&1; then
    echo "pod gone" | tee -a "$LOG"
    exit 0
  fi
  ssh_out=$(bash "$(dirname "$0")/runpod_pod_ssh.sh" "$POD_ID" \
    "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits" 2>>"$LOG" || true)
  util=$(echo "$ssh_out" | tr -d ' ' | head -1)
  if [[ "$util" =~ ^[0-9]+$ ]]; then
    max_util=$(( util > max_util ? util : max_util ))
    echo "t=${elapsed}s gpu_util=${util}% max=${max_util}%" | tee -a "$LOG"
    if [[ "$util" -ge "$THRESH" ]]; then
      echo "PASS gpu util >= ${THRESH}%" | tee -a "$LOG"
      exit 0
    fi
  else
    echo "t=${elapsed}s ssh/smi not ready" | tee -a "$LOG"
  fi
  sleep "$INTERVAL"
  elapsed=$((elapsed + INTERVAL))
done

echo "FAIL: max_util=${max_util}% < ${THRESH}% after ${MAX_WAIT}s — terminating $POD_ID" | tee -a "$LOG"
runpodctl pod delete "$POD_ID" -o json >>"$LOG" 2>&1 || true
if [[ "$RATE_HR" != "0" && "$RATE_HR" != "0.0" ]]; then
  hours=$(python3 -c "print(round(${MAX_WAIT}/3600.0, 4))")
  amt=$(python3 -c "print(round(float('${RATE_HR}')*${hours}, 4))")
  python3 scripts/campaign_spend.py actual --lane native_a100 --amount "$amt" 2>>"$LOG" || true
fi
exit 1
