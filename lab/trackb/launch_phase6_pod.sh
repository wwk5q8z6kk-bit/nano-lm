#!/bin/bash
# Resilient, SSH-blip-immune host launcher, incremental syncer, and auto-teardown watcher for Phase 6 H200 pod.
set -uo pipefail
POD=$1
LAB="${NANOLM_LAB_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LOCAL_DATA="$LAB/lab/baselines/results/phase4/phase3_data"
LOCAL_OUT="$LAB/lab/baselines/results/phase6"
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

echo "Waiting for SSH on Phase 6 H200 pod $POD ($IP:$PORT)..."
for i in $(seq 40); do
  "${SSH[@]}" true 2>/dev/null && break
  sleep 10
done

cd "$LAB"
echo "Uploading codebase and pre-verified Phase-3 curriculum to $POD..."
COPYFILE_DISABLE=1 tar --no-xattrs --exclude="__pycache__" --exclude="results" -czf - nanolm lab/__init__.py lab/system lab/trackb | \
  "${SSH[@]}" "mkdir -p /workspace/nl /workspace/phase3_data /workspace/results && tar --no-same-owner -xzf - -C /workspace/nl && test -s /workspace/nl/lab/trackb/run_phase6_h200.sh"

rsync -rt --no-o --no-g -e "$RS" "$LOCAL_DATA/" "root@$IP:/workspace/phase3_data/" < /dev/null

echo "Launching detached run_phase6_h200.sh on $POD at $(date -u +%FT%TZ)..."
"${SSH[@]}" "setsid bash /workspace/nl/lab/trackb/run_phase6_h200.sh > /workspace/results/phase6_master.log 2>&1 < /dev/null &" < /dev/null
sleep 10

while true; do
  if "${SSH[@]}" "true" 2>/dev/null < /dev/null; then
    "${SSH[@]}" "tail -n 5 /workspace/results/phase6_master.log 2>/dev/null" < /dev/null || true
    rsync -rt --no-o --no-g -e "$RS" "root@$IP:/workspace/results/" "$LOCAL_OUT/" < /dev/null || true
    if "${SSH[@]}" "grep -q 'PHASE6_ALL_COMPLETE' /workspace/results/phase6_master.log" 2>/dev/null < /dev/null; then
      echo "PHASE6_ALL_COMPLETE detected on $POD at $(date -u +%FT%TZ)!"
      break
    fi
    if ! "${SSH[@]}" "pgrep -f '[r]un_phase6_h200.sh' >/dev/null" 2>/dev/null < /dev/null; then
      echo "Remote run_phase6_h200.sh process exited on $POD at $(date -u +%FT%TZ)."
      break
    fi
  fi
  sleep 45
done

rsync -rt --no-o --no-g -e "$RS" "root@$IP:/workspace/results/" "$LOCAL_OUT/" < /dev/null || true
echo "Tearing down Phase 6 H200 pod $POD at $(date -u +%FT%TZ)..."
teardown_pod
echo "PHASE6_POD_CLOSED $POD"
