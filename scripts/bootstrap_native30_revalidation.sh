#!/usr/bin/env bash
# Bootstrap the Wave-1 30M revalidation on a running pod.
#
# Deliberately does NOT install the pinned torch. requirements.txt pins
# torch==2.11.0, whose wheel is built for CUDA 13.0; on a RunPod pytorch template
# with a CUDA 12.4 driver that install silently replaces a WORKING torch with one
# that reports torch.cuda.is_available() == False. The pod then trains on CPU at
# roughly 1/30th speed while still billing for an A100.
#
# The template already ships a torch matched to its driver. Install everything
# else and leave torch alone.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

POD_ID="${1:?pod id required}"
COMMIT_SHA="${2:-$(git rev-parse HEAD)}"
BRANCH="${REVAL_BRANCH:-frontier/accelerated-research-campaign-v2}"

read -r -d '' BOOT <<EOF || true
set -e
cd /workspace/nano-lm 2>/dev/null || { cd /workspace && git clone -q https://github.com/wwk5q8z6kk-bit/nano-lm.git && cd nano-lm; }
git fetch -q origin ${BRANCH}
git checkout -q ${COMMIT_SHA}

echo "TORCH_BEFORE: \$(python3 -c 'import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())')"

# Install dependencies EXCEPT torch — see header.
grep -v -E '^torch==' requirements.txt > /tmp/req_no_torch.txt
pip install -q -r /tmp/req_no_torch.txt 2>&1 | tail -2 || true

echo "TORCH_AFTER: \$(python3 -c 'import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())')"

export HF_HOME=/workspace/hf_cache
nohup bash scripts/native30_revalidation_remote.sh > /workspace/reval30.log 2>&1 &
echo BOOTSTRAP_STARTED
git rev-parse --short HEAD
EOF

echo "==> Bootstrapping 30M revalidation on ${POD_ID} @ ${COMMIT_SHA}"
bash scripts/runpod_pod_ssh.sh "${POD_ID}" "${BOOT}"
