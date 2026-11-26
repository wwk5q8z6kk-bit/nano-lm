#!/bin/bash
# Resilient host-side launcher, incremental syncer, and auto-teardown watcher for Phase 5 H200 pod.
set -euo pipefail
POD=$1
LAB="${NANOLM_LAB_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LOCAL_DATA="$LAB/lab/baselines/results/phase4/phase3_data"
LOCAL_OUT="$LAB/lab/baselines/results/phase5"
mkdir -p "$LOCAL_OUT"

teardown_pod() {
  for r in 1 2 3 4 5; do
    runpodctl remove pod "$POD" && break || sleep 5
  done
}

INFO=""
for i in $(seq 60); do
  INFO=$(runpodctl ssh info "$POD" 2>/dev/null || true)
  echo "$INFO" | grep -q '"port"' && break
  sleep 10
done
IP=$(echo "$INFO" | python3 -c "import json,sys;print(json.load(sys.stdin)['ip'])")
PORT=$(echo "$INFO" | python3 -c "import json,sys;print(json.load(sys.stdin)['port'])")
KEY=$HOME/.runpod/ssh/runpodctl-ssh-key
SSH=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30 -i "$KEY" -p "$PORT" "root@$IP")
RS="ssh -o StrictHostKeyChecking=no -i $KEY -p $PORT"

echo "Waiting for SSH on H200 pod $POD ($IP:$PORT)..."
for i in $(seq 40); do
  "${SSH[@]}" true 2>/dev/null && break
  sleep 10
done

cd "$LAB"
echo "Uploading codebase and pre-verified Phase-3 curriculum to $POD..."
COPYFILE_DISABLE=1 tar --no-xattrs --exclude="__pycache__" --exclude="results" -czf - nanolm lab/__init__.py lab/system lab/trackb | \
  "${SSH[@]}" "mkdir -p /workspace/nl /workspace/phase3_data /workspace/results && tar --no-same-owner -xzf - -C /workspace/nl && test -s /workspace/nl/lab/trackb/run_phase5_h200.sh"

if [ -d "$LOCAL_DATA" ]; then
  rsync -rt --no-o --no-g -e "$RS" "$LOCAL_DATA/" "root@$IP:/workspace/phase3_data/" < /dev/null
fi

echo "Launching detached run_phase5_h200.sh on $POD at $(date -u +%FT%TZ)..."
"${SSH[@]}" "setsid bash /workspace/nl/lab/trackb/run_phase5_h200.sh > /workspace/results/phase5_master.log 2>&1 < /dev/null &" < /dev/null
sleep 8

LAST_LINES=0
while true; do
  if "${SSH[@]}" "true" 2>/dev/null < /dev/null; then
    CUR_LINES=$("${SSH[@]}" "wc -l < /workspace/results/phase5_master.log 2>/dev/null || echo 0" < /dev/null | tr -d '[:space:]')
    if [ "${CUR_LINES:-0}" -gt "$LAST_LINES" ]; then
      "${SSH[@]}" "tail -n +$((LAST_LINES + 1)) /workspace/results/phase5_master.log" < /dev/null || true
      LAST_LINES=$CUR_LINES
    fi
    rsync -a -e "$RS" "root@$IP:/workspace/results/" "$LOCAL_OUT/" < /dev/null || true
    if "${SSH[@]}" "grep -q 'PHASE5_ALL_COMPLETE' /workspace/results/phase5_master.log" 2>/dev/null < /dev/null; then
      echo "PHASE5_ALL_COMPLETE detected on $POD at $(date -u +%FT%TZ)!"
      break
    fi
    if ! "${SSH[@]}" "pgrep -f '[r]un_phase5_h200.sh' >/dev/null" 2>/dev/null < /dev/null; then
      echo "Remote run_phase5_h200.sh process exited on $POD at $(date -u +%FT%TZ)."
      break
    fi
  fi
  sleep 45
done

rsync -a -e "$RS" "root@$IP:/workspace/results/" "$LOCAL_OUT/" < /dev/null || true
echo "Tearing down H200 pod $POD at $(date -u +%FT%TZ)..."
teardown_pod
echo "PHASE5_POD_CLOSED $POD"
