#!/bin/bash
# LAB-ARCH-02 grid driver (runs locally). usage: grid_pod.sh <pod_id> <queue_file>
# Uploads code + corpus, runs each queue line as `python -m lab.arch.train <line>` sequentially on the pod,
# pulls result.json files back after every run (so partial progress survives pod loss).
set -u
POD=$1; Q=$2
LAB="${NANOLM_LAB_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
DATA="${NANOLM_DATA_DIR:-$LAB/artifacts/nanolm_lab/data/v1}"
OUT="${NANOLM_RUNS_DIR:-$LAB/artifacts/nanolm_lab/runs}"
for i in $(seq 60); do INFO=$(runpodctl ssh info $POD 2>/dev/null); echo "$INFO" | grep -q '"port"' && break; sleep 10; done
IP=$(echo "$INFO" | python3 -c "import json,sys;print(json.load(sys.stdin)['ip'])"); PORT=$(echo "$INFO" | python3 -c "import json,sys;print(json.load(sys.stdin)['port'])")
KEY=$HOME/.runpod/ssh/runpodctl-ssh-key
SSH=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30 -i $KEY -p $PORT root@$IP)
RS="ssh -o StrictHostKeyChecking=no -i $KEY -p $PORT"
for i in $(seq 30); do "${SSH[@]}" true 2>/dev/null && break; sleep 10; done
cd $LAB
COPYFILE_DISABLE=1 tar --no-xattrs -czf - lab/__init__.py lab/arch | "${SSH[@]}" "mkdir -p /workspace/nl && tar xzf - -C /workspace/nl"
"${SSH[@]}" "pip install -q --break-system-packages flash-linear-attention numpy >/dev/null 2>&1; mkdir -p /workspace/data /workspace/runs"
for f in manifest.json tokenizer_32k.json val_web.bin val_textbook.bin val_math.bin val_code.bin train.bin; do
  for t in 1 2 3 4 5; do rsync -a --partial -e "$RS" $DATA/$f root@$IP:/workspace/data/ && break; sleep 5; done
done
"${SSH[@]}" "cd /workspace/data && sha256sum train.bin" | tee -a $OUT/../logs/grid_$POD.sha
while IFS= read -r line; do
  [ -z "$line" ] && continue
  echo "=== $(date -u +%FT%TZ) START $line"
  "${SSH[@]}" "cd /workspace/nl && NANOLM_FLA=1 python -m lab.arch.train $line --compile --micro 16 --data /workspace/data --out /workspace/runs 2>&1 | grep -v -i warn | grep -E 'DONE|EVAL|NON-FINITE|Error|error|\"step\": [0-9]*00,' " < /dev/null
  rsync -a -e "$RS" --include='*/' --include='result.json' --exclude='*' root@$IP:/workspace/runs/ $OUT/ < /dev/null
  echo "=== $(date -u +%FT%TZ) END $line"
done < $Q
echo "QUEUE COMPLETE $POD"
