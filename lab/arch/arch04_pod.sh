#!/bin/bash
# Immutable LAB-ARCH-04 runner for X5_atrd_muon and X6_pkm_muon (DO NOT EDIT WHILE RUNNING).
set -euo pipefail
POD=$1
SPECS=$2
LAB=/Users/mac/Projects/nano-lm/.worktrees/nanolm-lab
DATA=/Volumes/Express1M2/nanolm-lab/data/v1
OUT=/Volumes/Express1M2/nanolm-lab/runs
EXPECTED_SHA="867beb0d8d4c3977acaaf97fb91da39be29fb9fb040d86ea0bfdf78c9de72ac4"

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

for i in $(seq 30); do
  "${SSH[@]}" true 2>/dev/null && break
  sleep 10
done

cd "$LAB"
COPYFILE_DISABLE=1 tar --no-xattrs -czf - lab/__init__.py lab/arch | "${SSH[@]}" "mkdir -p /workspace/nl && tar xzf - -C /workspace/nl && test -s /workspace/nl/lab/arch/train_arch04.py"
"${SSH[@]}" "pip install -q --break-system-packages flash-linear-attention numpy >/dev/null 2>&1; mkdir -p /workspace/data /workspace/runs" < /dev/null

for f in manifest.json tokenizer_32k.json val_web.bin val_textbook.bin val_math.bin val_code.bin train.bin; do
  for t in 1 2 3 4 5; do
    rsync -a --partial -e "$RS" "$DATA/$f" "root@$IP:/workspace/data/" < /dev/null && break
    sleep 5
  done
done

REMOTE_SHA=$("${SSH[@]}" "sha256sum /workspace/data/train.bin | awk '{print \$1}'" < /dev/null)
echo "POD $POD train.bin sha256=$REMOTE_SHA"
if [ "$REMOTE_SHA" != "$EXPECTED_SHA" ]; then
  echo "FATAL: sha256 mismatch on $POD ($REMOTE_SHA != $EXPECTED_SHA)"
  runpodctl remove pod "$POD" || true
  exit 1
fi

IFS=';' read -ra RUNS <<< "$SPECS"
for line in "${RUNS[@]}"; do
  [ -z "$line" ] && continue
  echo "=== $(date -u +%FT%TZ) ARCH04 START $line"
  "${SSH[@]}" "cd /workspace/nl && NANOLM_FLA=1 python -m lab.arch.train_arch04 $line --compile --data /workspace/data --out /workspace/runs 2>&1 | grep --line-buffered -v -i warn | grep --line-buffered -E 'DONE|EVAL|NON-FINITE|Error|error|Traceback|\"step\": [0-9]*00,' " < /dev/null
  rsync -a -e "$RS" --include='*/' --include='result.json' --exclude='*' "root@$IP:/workspace/runs/" "$OUT/" < /dev/null
  echo "=== $(date -u +%FT%TZ) ARCH04 END $line"
done

echo "ARCH04_COMPLETE $POD"
runpodctl remove pod "$POD" || true
