#!/usr/bin/env bash
# Wave 1 paid pod launcher — native B200, student A100, verifier 4090
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

COMMIT_SHA="${WAVE1_COMMIT_SHA:-3715e5b003df79afdb2891474d94c38613cc5833}"
VOLUME_ID="${RUNPOD_VOLUME_ID:-04himzqxbm}"
IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
LEDGER="${CAMPAIGN_LEDGER:-artifacts/campaign/spend.json}"
TERMINATE_AFTER="${WAVE1_TERMINATE_AFTER:-$(date -u -v+2H '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date -u -d '+2 hours' '+%Y-%m-%dT%H:%M:%SZ')}"

SYNC_CMD="cd /workspace/nano-lm 2>/dev/null || { cd /workspace && git clone https://github.com/wwk5q8z6kk-bit/nano-lm.git && cd nano-lm; }; git fetch origin && git checkout ${COMMIT_SHA}; pip install -q -r requirements.txt; export HF_HOME=/workspace/hf_cache"

launch_pod() {
  local lane="$1" gpu="$2" rate="$3" est_hours="$4" name="$5" remote="$6"
  local est_cost
  est_cost=$(python3 -c "print(round(${rate} * ${est_hours}, 4))")
  echo "==> Lane ${lane}: ${gpu} est \$${est_cost}"
  python3 scripts/campaign_spend.py --ledger "${LEDGER}" gate --amount "${est_cost}" >/dev/null

  local pod_json pod_id
  pod_json=$(runpodctl pod create \
    --name "${name}" \
    --image "${IMAGE}" \
    --gpu-id "${gpu}" \
    --cloud-type SECURE \
    --network-volume-id "${VOLUME_ID}" \
    --container-disk-in-gb 30 \
    --volume-mount-path /workspace \
    --terminate-after "${TERMINATE_AFTER}" \
    --docker-args "bash -lc $(printf %q "${SYNC_CMD}; ${remote}")")

  pod_id=$(echo "${pod_json}" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
  python3 scripts/campaign_spend.py --ledger "${LEDGER}" commit \
    --lane "${lane}" \
    --description "Wave1 ${lane} pod launch" \
    --amount "${est_cost}" \
    --pod-id "${pod_id}" \
    --gpu "${gpu}" \
    --rate-hr "${rate}" >/dev/null
  echo "${lane}=${pod_id}"
}

NATIVE_REMOTE='python3 scripts/train_native_nano.py --run-id A_s0 && python3 scripts/train_native_nano.py --run-id A_s1 && python3 scripts/train_native_nano.py --run-id B_s0 && python3 scripts/train_native_nano.py --run-id B_s1 && python3 scripts/train_native_nano.py --run-id C_s0 && python3 scripts/train_native_nano.py --run-id C_s1 && python3 scripts/train_native_nano.py --run-id D_s0 && python3 scripts/train_native_nano.py --run-id D_s1; echo NATIVE_DONE'

STUDENT_REMOTE='export NANOSCIBE_QWEN_WEIGHTS=Qwen/Qwen2.5-32B-Instruct; python3 scripts/student_structured_eval.py --record-spend; echo STUDENT_DONE'

VERIFIER_REMOTE='python3 scripts/verifier_lane.py --record-spend; echo VERIFIER_DONE'

TS=$(date +%Y%m%d%H%M)
launch_pod "native_b200" "NVIDIA B200" 4.99 1.0 "wave1-native-${TS}" "${NATIVE_REMOTE}" &
launch_pod "student_a" "NVIDIA A100-SXM4-80GB" 1.59 0.75 "wave1-student-${TS}" "${STUDENT_REMOTE}" &
launch_pod "verifier" "NVIDIA GeForce RTX 4090" 0.74 0.25 "wave1-verifier-${TS}" "${VERIFIER_REMOTE}" &
wait
echo "==> All wave1 pods launched"
