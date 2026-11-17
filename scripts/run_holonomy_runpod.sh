#!/usr/bin/env bash
set -euo pipefail

# RunPod GPU runner for NanoLM HSPA Discovery & Evolution (X1, X2, X3, X4).
# Strictly creates/terminates ONLY MY_POD_ID; never touches peer pods.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${WORKTREE_DIR}/artifacts/nanolm_holonomy_v1"
POD_ID_FILE="/tmp/nanolm_holonomy_9946_pod_id.txt"
SSH_KEY="${HOME}/.runpod/ssh/runpodctl-ssh-key"
IMAGE_NAME="runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B}"

mkdir -p "${OUT_DIR}"

MY_POD_ID=""
WATCHDOG_PID=""
POD_START_EPOCH=""

cleanup() {
  local exit_code=$?
  if [[ -n "${WATCHDOG_PID}" ]]; then
    kill "${WATCHDOG_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${MY_POD_ID}" ]]; then
    echo "[runpod] Terminating ONLY own pod MY_POD_ID=${MY_POD_ID}..."
    runpodctl remove pod "${MY_POD_ID}" >/dev/null 2>&1 || true
    rm -f "${POD_ID_FILE}"
  fi
  exit "${exit_code}"
}
trap cleanup EXIT

BALANCE_BEFORE_JSON="$(runpodctl user)"
BALANCE_BEFORE="$(python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("clientBalance", 0.0))' <<< "${BALANCE_BEFORE_JSON}")"
echo "[runpod] Account balance before launch: \$${BALANCE_BEFORE}"

CAN_LAUNCH="$(python3 -c 'import sys; b=float(sys.argv[1]); print("yes" if b > 28.0 else "no")' "${BALANCE_BEFORE}")"
if [[ "${CAN_LAUNCH}" != "yes" ]]; then
  echo "[runpod] ERROR: Account balance \$${BALANCE_BEFORE} too close to \$20 owner reserve." >&2
  exit 1
fi

# Require >= 48GB VRAM candidates first for comfortable 3.84B (X3) + X4 runs
GPU_TYPES=("NVIDIA A40" "NVIDIA L40S" "NVIDIA RTX A6000")
CREATE_OUT=""
CHOSEN_GPU=""
for GPU in "${GPU_TYPES[@]}"; do
  echo "[runpod] Attempting secureCloud pod creation with GPU=${GPU}..."
  if CREATE_OUT="$(runpodctl create pod \
      --name "nanolm-holonomy-9946" \
      --gpuType "${GPU}" \
      --imageName "${IMAGE_NAME}" \
      --secureCloud \
      --ports "22/tcp" \
      --startSSH \
      --containerDiskSize 45 \
      --volumeSize 45 \
      --volumePath /workspace 2>&1)"; then
    CHOSEN_GPU="${GPU}"
    break
  fi
  echo "[runpod] GPU ${GPU} unavailable: ${CREATE_OUT}"
done

if [[ -z "${CHOSEN_GPU}" ]]; then
  echo "[runpod] ERROR: All candidate 48GB GPU types failed." >&2
  exit 1
fi

echo "[runpod] create pod output: ${CREATE_OUT}"
MY_POD_ID="$(python3 -c '
import re, sys
txt = sys.argv[1]
m = re.search(r"pod\s+\"?([a-z0-9]{10,20})\"?", txt, re.I) or re.search(r"\"([a-z0-9]{10,20})\"", txt)
print(m.group(1) if m else "")
' "${CREATE_OUT}")"

if [[ -z "${MY_POD_ID}" ]]; then
  echo "[runpod] ERROR: Could not parse MY_POD_ID from: ${CREATE_OUT}" >&2
  exit 1
fi

echo "${MY_POD_ID}" > "${POD_ID_FILE}"
POD_START_EPOCH="$(date +%s)"
echo "[runpod] Created MY_POD_ID=${MY_POD_ID} (${CHOSEN_GPU}). Starting 75m watchdog..."

(
  sleep 4500
  runpodctl remove pod "${MY_POD_ID}" >/dev/null 2>&1 || true
) &
WATCHDOG_PID=$!

SSH_HOST=""
SSH_PORT=""
SSH_READY_OK=0
for attempt in $(seq 1 60); do
  INFO_JSON="$(runpodctl ssh info "${MY_POD_ID}" 2>&1 || true)"
  PARSED="$(python3 -c '
import json, sys
try:
    d = json.loads(sys.argv[1])
    ip = d.get("ip") or d.get("publicIp") or ""
    port = d.get("port") or d.get("sshPort") or ""
    if ip and port:
        print(f"{ip} {port}")
except Exception:
    pass
' "${INFO_JSON}")"
  if [[ -n "${PARSED}" ]]; then
    SSH_HOST="$(awk '{print $1}' <<< "${PARSED}")"
    SSH_PORT="$(awk '{print $2}' <<< "${PARSED}")"
    if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=6 \
        -i "${SSH_KEY}" -p "${SSH_PORT}" "root@${SSH_HOST}" "echo SSH_READY" </dev/null 2>/dev/null | grep -q "SSH_READY"; then
      echo "[runpod] SSH ready on ${SSH_HOST}:${SSH_PORT} (attempt ${attempt})"
      SSH_READY_OK=1
      break
    fi
  fi
  if (( attempt % 6 == 0 )); then
    echo "[runpod] waiting for SSH (attempt ${attempt}/60): host=${SSH_HOST:-none} port=${SSH_PORT:-none}"
  fi
  sleep 5
done

if [[ "${SSH_READY_OK}" -ne 1 ]]; then
  echo "[runpod] ERROR: SSH failed to become ready for ${MY_POD_ID}." >&2
  exit 1
fi

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -i "${SSH_KEY}" -p "${SSH_PORT}")

echo "[runpod] Uploading nanolm_holonomy, nanolm_frontier, nanolm_posttrain..."
ssh -n "${SSH_OPTS[@]}" "root@${SSH_HOST}" "mkdir -p /workspace/nanolm_repo"
COPYFILE_DISABLE=1 tar -C "${WORKTREE_DIR}" --no-xattrs --exclude='__pycache__' --exclude='*.pyc' -czf - nanolm_holonomy nanolm_frontier nanolm_posttrain \
  | ssh "${SSH_OPTS[@]}" "root@${SSH_HOST}" "tar --no-same-owner --no-same-permissions -xzf - -C /workspace/nanolm_repo"

echo "[runpod] Running CUDA tests + X1 + X2 + X3 + X4..."
ssh "${SSH_OPTS[@]}" "root@${SSH_HOST}" bash -s <<EOF
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/workspace/hf_cache
mkdir -p /workspace/hf_cache
nvidia-smi
if ! python3 -c "import numpy, transformers, accelerate, pytest; v=tuple(int(x) for x in transformers.__version__.split('.')[:2]); assert v >= (4, 51)" >/dev/null 2>&1; then
  python3 -m pip install --break-system-packages -q --upgrade "transformers>=4.51.0" accelerate sentencepiece protobuf pytest numpy
fi
cd /workspace/nanolm_repo
mkdir -p artifacts/nanolm_holonomy_v1
PYTHONPATH=. python3 -m pytest -q nanolm_holonomy/tests/test_holonomy.py
PYTHONPATH=. python3 -u -m nanolm_holonomy.experiments \
  --mode all \
  --out-dir artifacts/nanolm_holonomy_v1 \
  --model-id "${MODEL_ID}" \
  2>&1 | tee artifacts/nanolm_holonomy_v1/holonomy_run.log
EOF

echo "[runpod] Downloading JSON artifacts & run logs..."
ssh -n "${SSH_OPTS[@]}" "root@${SSH_HOST}" \
  "tar -C /workspace/nanolm_repo/artifacts/nanolm_holonomy_v1 -czf - ." \
  | tar --no-same-owner -xzf - -C "${OUT_DIR}"

POD_END_EPOCH="$(date +%s)"
POD_DETAILS_JSON="$(runpodctl get pod "${MY_POD_ID}" -a || true)"
WALL_SECS=$((POD_END_EPOCH - POD_START_EPOCH))

echo "[runpod] Terminating MY_POD_ID=${MY_POD_ID}..."
runpodctl remove pod "${MY_POD_ID}"
REMOVED_POD_ID="${MY_POD_ID}"
MY_POD_ID=""
rm -f "${POD_ID_FILE}"

ACTIVE_PODS_AFTER="$(runpodctl get pod || true)"
if grep -q "${REMOVED_POD_ID}" <<< "${ACTIVE_PODS_AFTER}"; then
  echo "[runpod] WARNING: retrying removal of ${REMOVED_POD_ID}..." >&2
  runpodctl remove pod "${REMOVED_POD_ID}" || true
fi

BALANCE_AFTER_JSON="$(runpodctl user)"
BALANCE_AFTER="$(python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("clientBalance", 0.0))' <<< "${BALANCE_AFTER_JSON}")"

POD_DETAILS_JSON="${POD_DETAILS_JSON}" \
WALL_SECS="${WALL_SECS}" \
REMOVED_POD_ID="${REMOVED_POD_ID}" \
CHOSEN_GPU="${CHOSEN_GPU}" \
IMAGE_NAME="${IMAGE_NAME}" \
MODEL_ID="${MODEL_ID}" \
BALANCE_BEFORE="${BALANCE_BEFORE}" \
BALANCE_AFTER="${BALANCE_AFTER}" \
OUT_DIR="${OUT_DIR}" \
python3 - <<'PY'
import json
import os
from pathlib import Path

details = os.environ.get("POD_DETAILS_JSON", "")
cost_per_hr = 0.0
lines = [ln for ln in details.splitlines() if "\t" in ln]
if len(lines) >= 2:
    headers = [h.strip() for h in lines[0].split("\t")]
    vals = [v.strip() for v in lines[1].split("\t")]
    if "$/HR" in headers:
        idx = headers.index("$/HR")
        if idx < len(vals):
            try:
                cost_per_hr = float(vals[idx])
            except ValueError:
                pass
wall_secs = int(os.environ["WALL_SECS"])
estimated_pod_usd = round(cost_per_hr * (wall_secs / 3600.0), 4) if cost_per_hr > 0 else None
bal_before = float(os.environ["BALANCE_BEFORE"])
bal_after = float(os.environ["BALANCE_AFTER"])

prov = {
    "pod_id": os.environ["REMOVED_POD_ID"],
    "gpu_type": os.environ["CHOSEN_GPU"],
    "image": os.environ["IMAGE_NAME"],
    "model_id": os.environ["MODEL_ID"],
    "wall_seconds": wall_secs,
    "cost_per_hr_usd": cost_per_hr if cost_per_hr > 0 else None,
    "pod_attributed_spend_usd": estimated_pod_usd,
    "account_balance_before_usd": round(bal_before, 4),
    "account_balance_after_usd": round(bal_after, 4),
    "account_balance_delta_usd": round(bal_before - bal_after, 4),
    "note": "Account delta includes any concurrent peer pods active during window; pod_attributed_spend_usd reflects this pod alone.",
}
Path(os.environ["OUT_DIR"], "SPEND_AND_PROVENANCE.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
print("SPEND_PROVENANCE:", json.dumps(prov))
PY

echo "[runpod] Campaign complete and pod verified terminated."
