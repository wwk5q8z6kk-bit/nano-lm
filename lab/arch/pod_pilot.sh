#!/bin/bash
# usage: pod_pilot.sh <pod_id> <peak_tflops>
POD=$1; PEAK=$2
for i in $(seq 40); do INFO=$(runpodctl ssh info $POD 2>/dev/null); echo "$INFO" | grep -q '"port"' && break; sleep 10; done
IP=$(echo "$INFO" | python3 -c "import json,sys;print(json.load(sys.stdin)['ip'])"); PORT=$(echo "$INFO" | python3 -c "import json,sys;print(json.load(sys.stdin)['port'])")
SSH=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -i $HOME/.runpod/ssh/runpodctl-ssh-key -p $PORT root@$IP)
for i in $(seq 30); do "${SSH[@]}" true 2>/dev/null && break; sleep 10; done
cd "$(dirname $0)/../.."
COPYFILE_DISABLE=1 tar --no-xattrs -czf - lab/__init__.py lab/arch | "${SSH[@]}" "mkdir -p /workspace/nl && tar xzf - -C /workspace/nl"
"${SSH[@]}" "cd /workspace/nl && pip install -q --break-system-packages flash-linear-attention numpy 2>&1 | tail -1; nvidia-smi --query-gpu=name,memory.total --format=csv; MICRO=${MICRO:-32} ARMS=${ARMS:-T0_dense,H2_delta,R1_loop,P2_delta} PEAK_TFLOPS=$PEAK python lab/arch/pilot_p0.py" 2>&1
"${SSH[@]}" "cat /workspace/nl/pilot_p0.json" > lab/reports/pilot_p0_$POD.json
