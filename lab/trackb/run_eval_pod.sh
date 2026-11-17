#!/bin/bash
# usage: run_eval_pod.sh <pod_id> "<model1>:<tag1> <model2>:<tag2> ..."
POD=$1; LIST=$2
LAB=/Users/mac/Projects/nano-lm/.worktrees/nanolm-lab
for i in $(seq 60); do INFO=$(runpodctl ssh info $POD 2>/dev/null); echo "$INFO" | grep -q '"port"' && break; sleep 10; done
IP=$(echo "$INFO" | python3 -c "import json,sys;print(json.load(sys.stdin)['ip'])"); PORT=$(echo "$INFO" | python3 -c "import json,sys;print(json.load(sys.stdin)['port'])")
KEY=$HOME/.runpod/ssh/runpodctl-ssh-key
SSH=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30 -i $KEY -p $PORT root@$IP)
for i in $(seq 30); do "${SSH[@]}" true 2>/dev/null && break; sleep 10; done
scp -o StrictHostKeyChecking=no -i $KEY -P $PORT $LAB/lab/trackb/eval_pod.sh root@$IP:/workspace/
"${SSH[@]}" "pip install -q --break-system-packages vllm 'lm_eval[ifeval,math]' hf_transfer langdetect immutabledict 2>&1 | tail -3; python -c 'import vllm,lm_eval;print(vllm.__version__, lm_eval.__version__)'" < /dev/null
for item in $LIST; do
  M=${item%%:*}; T=${item##*:}
  echo "=== $(date -u +%FT%TZ) START $M"
  "${SSH[@]}" "bash /workspace/eval_pod.sh $M $T 0" < /dev/null
  mkdir -p $LAB/lab/baselines/results/trackb
  rsync -a -e "ssh -o StrictHostKeyChecking=no -i $KEY -p $PORT" --exclude='*.jsonl' root@$IP:/workspace/results/ $LAB/lab/baselines/results/trackb/ < /dev/null
  rsync -a -e "ssh -o StrictHostKeyChecking=no -i $KEY -p $PORT" root@$IP:/workspace/results/ /Volumes/Express1M2/nanolm-lab/trackb-results/ < /dev/null
  echo "=== $(date -u +%FT%TZ) END $M"
done
echo "EVAL QUEUE COMPLETE"
