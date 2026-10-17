#!/usr/bin/env bash
# Minimal RunPod soak test: template pod, sleep 3600, no git clone, no network volume.
# Polls uptimeSeconds + SSH + nvidia-smi at 2/5/10/15 minutes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GPU="${SOAK_GPU:-NVIDIA A100 80GB PCIe}"
DC="${SOAK_DATA_CENTER:-EU-RO-1}"
TEMPLATE="${SOAK_TEMPLATE:-runpod-torch-v240}"
TERMINATE_AFTER="${SOAK_TERMINATE_AFTER:-$(date -u -v+2H '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date -u -d '+2 hours' '+%Y-%m-%dT%H:%M:%SZ')}"
LOG_DIR="${SOAK_LOG_DIR:-artifacts/campaign/health_gates}"
mkdir -p "$LOG_DIR"

TS=$(date +%Y%m%d%H%M)
NAME="soak-minimal-${TS}"
LOG="$LOG_DIR/soak_${TS}.log"

echo "==> Creating soak pod ${NAME} gpu=${GPU} dc=${DC} template=${TEMPLATE}" | tee "$LOG"
POD_JSON=$(runpodctl pod create \
  --name "${NAME}" \
  --template-id "${TEMPLATE}" \
  --gpu-id "${GPU}" \
  --data-center-ids "${DC}" \
  --cloud-type SECURE \
  --container-disk-in-gb 20 \
  --terminate-after "${TERMINATE_AFTER}" \
  --docker-args "bash -lc 'sleep 3600'")

POD_ID=$(echo "${POD_JSON}" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "SOAK_POD_ID=${POD_ID}" | tee -a "$LOG"
echo "${POD_ID}" > /tmp/soak_pod_id.txt

poll_once() {
  local label="$1"
  local pod_json uptime ssh_ok smi_out
  pod_json=$(runpodctl pod get "${POD_ID}" -o json 2>>"$LOG" || echo '{}')
  uptime=$(echo "$pod_json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('uptimeSeconds', d.get('uptime', 'NA')))" 2>/dev/null || echo "NA")
  volume_gb=$(echo "$pod_json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('volumeInGb', 'NA'))" 2>/dev/null || echo "NA")
  desired=$(echo "$pod_json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('desiredStatus','NA'))" 2>/dev/null || echo "NA")
  dc=$(echo "$pod_json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('dataCenterId', d.get('machine',{}).get('dataCenterId','NA')))" 2>/dev/null || echo "NA")

  ssh_ok=FAIL
  smi_out=""
  if bash scripts/runpod_pod_ssh.sh "${POD_ID}" "hostname && nvidia-smi -L" >>"$LOG" 2>&1; then
    ssh_ok=PASS
    smi_out=$(bash scripts/runpod_pod_ssh.sh "${POD_ID}" "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader" 2>>"$LOG" || true)
  fi

  echo "[$label] t=${label} desired=${desired} uptimeSeconds=${uptime} volumeInGb=${volume_gb} dc=${dc} ssh=${ssh_ok} smi=${smi_out:-none}" | tee -a "$LOG"
}

prev=0
for target in 2 5 10 15; do
  sleep_sec=$(( (target - prev) * 60 ))
  if [[ "$sleep_sec" -gt 0 ]]; then
    echo "==> Sleeping ${sleep_sec}s until t=${target}min..." | tee -a "$LOG"
    sleep "$sleep_sec"
  fi
  prev=$target
  poll_once "${target}min"
done

echo "==> Final pod state:" | tee -a "$LOG"
runpodctl pod get "${POD_ID}" -o json | tee -a "$LOG"
echo "SOAK_DONE pod=${POD_ID} log=${LOG}"
