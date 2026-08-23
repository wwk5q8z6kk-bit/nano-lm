#!/usr/bin/env bash
# Wave 1 paid pod launcher — native A100 (sm_90-safe), student A100, verifier 4090
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

COMMIT_SHA="${WAVE1_COMMIT_SHA:-a89ecb026d31bb9df3b9122fac6e1dd5bc6c3261}"
VOLUME_ID="${RUNPOD_VOLUME_ID:-04himzqxbm}"
DATA_CENTER_IDS="${RUNPOD_DATA_CENTER_IDS:-EU-RO-1}"
PYTORCH_TEMPLATE_ID="${RUNPOD_PYTORCH_TEMPLATE:-runpod-torch-v240}"
IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
LEDGER="${CAMPAIGN_LEDGER:-artifacts/campaign/spend.json}"

# Native artifact contract (post round1 loss on pod delete):
# - train_native_nano writes artifacts/native_checkpoints/<run_id>/ under the repo.
# - RUNPOD_VOLUME_ID mounts at /workspace; remote jobs MUST archive there before NATIVE_DONE.
# - Operator: before terminating the pod, pull to local:
#     runpodctl ssh <pod_id> -- "tar -C /workspace/nano-lm -czf - artifacts/native_checkpoints" | tar -xzf - -C .
#   or rsync from /workspace/campaign_native_checkpoints on the network volume.
NATIVE_ARCHIVE='mkdir -p /workspace/campaign_native_checkpoints && rsync -a artifacts/native_checkpoints/ /workspace/campaign_native_checkpoints/'
TERMINATE_AFTER="${WAVE1_TERMINATE_AFTER:-$(date -u -v+2H '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date -u -d '+2 hours' '+%Y-%m-%dT%H:%M:%SZ')}"

SYNC_CMD="cd /workspace/nano-lm 2>/dev/null || { cd /workspace && git clone https://github.com/wwk5q8z6kk-bit/nano-lm.git && cd nano-lm; }; git fetch origin && git checkout ${COMMIT_SHA}; pip install -q -r requirements.txt; export HF_HOME=/workspace/hf_cache"
PREFLIGHT_CMD="python3 -m nanoscribe.runpod_gpu_preflight"

launch_pod() {
  local lane="$1" gpu="$2" rate="$3" est_hours="$4" name="$5" remote="$6" use_template="${7:-0}"
  python3 -c "from nanoscribe.runpod_gpu_preflight import block_b200_without_sm100; block_b200_without_sm100('${gpu}', '${PYTORCH_TEMPLATE_ID}')"
  python3 -c "from nanoscribe.runpod_gpu_preflight import block_b200_without_sm100; block_b200_without_sm100('${gpu}', '${IMAGE}')"
  local est_cost
  est_cost=$(python3 -c "print(round(${rate} * ${est_hours}, 4))")
  echo "==> Lane ${lane}: ${gpu} est \$${est_cost}"
  python3 scripts/campaign_spend.py --ledger "${LEDGER}" gate --amount "${est_cost}" >/dev/null

  local pod_json pod_id create_args=()
  if [[ "${use_template}" == "1" ]]; then
    create_args=(--template-id "${PYTORCH_TEMPLATE_ID}")
  else
    create_args=(--image "${IMAGE}")
  fi

  pod_json=$(runpodctl pod create \
    --name "${name}" \
    "${create_args[@]}" \
    --gpu-id "${gpu}" \
    --data-center-ids "${DATA_CENTER_IDS}" \
    --cloud-type SECURE \
    --network-volume-id "${VOLUME_ID}" \
    --container-disk-in-gb 30 \
    --volume-mount-path /workspace \
    --terminate-after "${TERMINATE_AFTER}" \
    --docker-args "bash -lc $(printf %q "${SYNC_CMD}; ${PREFLIGHT_CMD} || exit 1; ${remote}")")

  pod_id=$(echo "${pod_json}" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
  rate=$(echo "${pod_json}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('costPerHr', ${rate}))")
  python3 scripts/campaign_spend.py --ledger "${LEDGER}" commit \
    --lane "${lane}" \
    --description "Wave1 ${lane} pod launch" \
    --amount "${est_cost}" \
    --pod-id "${pod_id}" \
    --gpu "${gpu}" \
    --rate-hr "${rate}" >/dev/null
  echo "${lane}=${pod_id}"
}

NATIVE_REMOTE='python3 scripts/train_native_nano.py --export-train-json; for rid in native30_bottleneck_span_s0 native30_bottleneck_span_s1 native30_bottleneck_struct_s0 native30_bottleneck_struct_s1 native30_decoder_span_s0 native30_decoder_span_s1 native30_decoder_struct_s0 native30_decoder_struct_s1; do python3 scripts/train_native_nano.py --run-id "$rid" || exit 1; done; mkdir -p /workspace/campaign_native_checkpoints && rsync -a artifacts/native_checkpoints/ /workspace/campaign_native_checkpoints/; echo NATIVE_DONE'

STUDENT_REMOTE='export NANOSCIBE_QWEN_WEIGHTS=Qwen/Qwen2.5-32B-Instruct; python3 scripts/student_structured_eval.py --record-spend; echo STUDENT_DONE'

VERIFIER_REMOTE='python3 scripts/verifier_lane.py --record-spend; echo VERIFIER_DONE'

# Native relaunch only (student + verifier already running)
launch_native_split() {
  local half1='python3 scripts/train_native_nano.py --export-train-json; for rid in native30_bottleneck_span_s0 native30_bottleneck_span_s1 native30_bottleneck_struct_s0 native30_bottleneck_struct_s1; do python3 scripts/train_native_nano.py --run-id "$rid" || exit 1; done; mkdir -p /workspace/campaign_native_checkpoints && rsync -a artifacts/native_checkpoints/ /workspace/campaign_native_checkpoints/; echo NATIVE_HALF1_DONE'
  local half2='python3 scripts/train_native_nano.py --export-train-json; for rid in native30_decoder_span_s0 native30_decoder_span_s1 native30_decoder_struct_s0 native30_decoder_struct_s1; do python3 scripts/train_native_nano.py --run-id "$rid" || exit 1; done; mkdir -p /workspace/campaign_native_checkpoints && rsync -a artifacts/native_checkpoints/ /workspace/campaign_native_checkpoints/; echo NATIVE_HALF2_DONE'
  TS=$(date +%Y%m%d%H%M)
  launch_pod "native_a100" "NVIDIA A100 80GB PCIe" 1.39 0.5 "wave1-native-a100-1-${TS}" "${half1}" 1 &
  launch_pod "native_a100" "NVIDIA A100 80GB PCIe" 1.39 0.5 "wave1-native-a100-2-${TS}" "${half2}" 1 &
  wait
}

if [[ "${1:-}" == "native-only" ]]; then
  launch_native_split
  echo "==> Native A100 pods launched"
  exit 0
fi

TS=$(date +%Y%m%d%H%M)
launch_pod "native_a100" "NVIDIA A100 80GB PCIe" 1.39 0.5 "wave1-native-a100-${TS}" "${NATIVE_REMOTE}" 1 &
launch_pod "student_a" "NVIDIA A100-SXM4-80GB" 1.59 0.75 "wave1-student-${TS}" "${STUDENT_REMOTE}" 0 &
launch_pod "verifier" "NVIDIA GeForce RTX 4090" 0.74 0.25 "wave1-verifier-${TS}" "${VERIFIER_REMOTE}" 0 &
wait
echo "==> All wave1 pods launched"
