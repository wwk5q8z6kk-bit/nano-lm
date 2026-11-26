#!/bin/bash
# Resilient, SSH-blip-immune LAB-SCALE-03 runner (supports train_nextgen & train_arch04 via setsid).
# Usage: scale03_pod.sh <pod_id> <module: lab.arch.train_arch04 | lab.arch.train_nextgen> "<cli_args>"
set -uo pipefail
POD=$1
MOD=$2
CLI_ARGS=$3
LAB="${NANOLM_LAB_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
DATA="${NANOLM_DATA_DIR:-$LAB/artifacts/nanolm_lab/data/v1}"
OUT="${NANOLM_RUNS_DIR:-$LAB/artifacts/nanolm_lab/runs}"
EXPECTED_SHA="867beb0d8d4c3977acaaf97fb91da39be29fb9fb040d86ea0bfdf78c9de72ac4"

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

for i in $(seq 40); do
  "${SSH[@]}" true 2>/dev/null && break
  sleep 10
done

cd "$LAB"
COPYFILE_DISABLE=1 tar --no-xattrs --exclude="__pycache__" -czf - lab/__init__.py lab/arch | \
  "${SSH[@]}" "mkdir -p /workspace/nl /workspace/data /workspace/runs && tar --no-same-owner -xzf - -C /workspace/nl && test -s /workspace/nl/lab/arch/train_arch04.py"
"${SSH[@]}" "pip install -q --break-system-packages flash-linear-attention numpy >/dev/null 2>&1" < /dev/null

for f in manifest.json tokenizer_32k.json val_web.bin val_textbook.bin val_math.bin val_code.bin train.bin; do
  for t in 1 2 3 4 5; do
    rsync -rt --no-o --no-g --partial -e "$RS" "$DATA/$f" "root@$IP:/workspace/data/" < /dev/null && break
    sleep 5
  done
done

REMOTE_SHA=$("${SSH[@]}" "sha256sum /workspace/data/train.bin | awk '{print \$1}'" < /dev/null | tr -d '[:space:]')
echo "POD $POD train.bin sha256=$REMOTE_SHA"
if [ "$REMOTE_SHA" != "$EXPECTED_SHA" ]; then
  echo "FATAL: sha256 mismatch on $POD ($REMOTE_SHA != $EXPECTED_SHA)"
  teardown_pod
  exit 1
fi

echo "=== $(date -u +%FT%TZ) SCALE03 DETACHED LAUNCH ($POD): python -m $MOD $CLI_ARGS ==="
"${SSH[@]}" "setsid bash -c 'cd /workspace/nl && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NANOLM_FLA=1 python -m $MOD $CLI_ARGS --data /workspace/data --out /workspace/runs > /workspace/runs/train_master.log 2>&1 && echo SCALE03_JOB_FINISHED >> /workspace/runs/train_master.log' < /dev/null >/dev/null 2>&1 &" < /dev/null
sleep 10

while true; do
  if "${SSH[@]}" "true" 2>/dev/null < /dev/null; then
    "${SSH[@]}" "grep -E 'DONE|EVAL|NON-FINITE|Traceback|SCALE03_JOB_FINISHED|\"step\": [0-9]*00,' /workspace/runs/train_master.log 2>/dev/null | tail -n 6" < /dev/null || true
    rsync -rt --no-o --no-g -e "$RS" --include='*/' --include='result.json' --exclude='*' "root@$IP:/workspace/runs/" "$OUT/" < /dev/null || true
    if "${SSH[@]}" "grep -q 'SCALE03_JOB_FINISHED' /workspace/runs/train_master.log" 2>/dev/null < /dev/null; then
      echo "SCALE03_JOB_FINISHED on $POD at $(date -u +%FT%TZ)"
      break
    fi
    if ! "${SSH[@]}" "pgrep -f '[l]ab.arch.train' >/dev/null" 2>/dev/null < /dev/null; then
      echo "Remote training process exited on $POD at $(date -u +%FT%TZ)"
      "${SSH[@]}" "tail -n 25 /workspace/runs/train_master.log" < /dev/null || true
      break
    fi
  fi
  sleep 60
done

rsync -rt --no-o --no-g -e "$RS" --include='*/' --include='result.json' --exclude='*' "root@$IP:/workspace/runs/" "$OUT/" < /dev/null || true
echo "SCALE03_COMPLETE $POD at $(date -u +%FT%TZ)"
teardown_pod
