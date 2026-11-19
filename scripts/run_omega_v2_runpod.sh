#!/usr/bin/env bash
set -euo pipefail

# Pre-flight-estimated RunPod fallback runner for NANOLM Omega-v2 (EXP-Omega4..Omega7).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${WORKTREE_DIR}/artifacts/nanolm_omega_v2"
POD_ID_FILE="/tmp/nanolm_omega_v2_pod_id.txt"
SSH_KEY="${HOME}/.runpod/ssh/runpodctl-ssh-key"
IMAGE_NAME="runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"

mkdir -p "${OUT_DIR}"

echo "[preflight] Running Roofline Compute, VRAM & Cost estimator before pod launch..."
PYTHONPATH="${WORKTREE_DIR}" python3 -m nanolm_holonomy.compute_planner \
  --preflight \
  --allow-kaggle \
  --calibrate-v2 "${WORKTREE_DIR}/artifacts/nanolm_holonomy_v2" \
  | tee "${OUT_DIR}/preflight_plan.json"

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

GPU_TYPES=("NVIDIA L40S" "NVIDIA A40" "NVIDIA RTX A6000")
CREATE_OUT=""
CHOSEN_GPU=""
for GPU in "${GPU_TYPES[@]}"; do
  echo "[runpod] Attempting secureCloud pod creation with GPU=${GPU}..."
  if CREATE_OUT="$(runpodctl create pod \
      --name "nanolm-omega-v2-9946" \
      --gpuType "${GPU}" \
      --imageName "${IMAGE_NAME}" \
      --secureCloud \
      --ports "22/tcp" \
      --startSSH \
      --containerDiskSize 50 \
      --volumeSize 50 \
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

MY_POD_ID="$(python3 -c '
import re, sys
txt = sys.argv[1]
m = re.search(r"pod\s+\"?([a-z0-9]{10,20})\"?", txt, re.I) or re.search(r"\"([a-z0-9]{10,20})\"", txt)
print(m.group(1) if m else "")
' "${CREATE_OUT}")"

if [[ -z "${MY_POD_ID}" ]]; then
  echo "[runpod] ERROR: Could not parse MY_POD_ID from ${CREATE_OUT}" >&2
  exit 1
fi

echo "${MY_POD_ID}" > "${POD_ID_FILE}"
POD_START_EPOCH="$(date +%s)"
echo "[runpod] Created MY_POD_ID=${MY_POD_ID} (${CHOSEN_GPU}). Starting 60m watchdog..."

(
  sleep 3600
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
  sleep 5
done

if [[ "${SSH_READY_OK}" -ne 1 ]]; then
  echo "[runpod] ERROR: SSH failed for ${MY_POD_ID}." >&2
  exit 1
fi

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -i "${SSH_KEY}" -p "${SSH_PORT}")

echo "[runpod] Uploading codebase..."
ssh -n "${SSH_OPTS[@]}" "root@${SSH_HOST}" "mkdir -p /workspace/nanolm_repo"
COPYFILE_DISABLE=1 tar -C "${WORKTREE_DIR}" --no-xattrs --exclude='__pycache__' --exclude='*.pyc' -czf - nanolm_holonomy nanolm_frontier nanolm_posttrain \
  | ssh "${SSH_OPTS[@]}" "root@${SSH_HOST}" "tar --no-same-owner --no-same-permissions -xzf - -C /workspace/nanolm_repo"

echo "[runpod] Running pytest + EXP-Omega4..Omega7..."
ssh "${SSH_OPTS[@]}" "root@${SSH_HOST}" bash -s <<'EOF'
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/workspace/hf_cache
mkdir -p /workspace/hf_cache
nvidia-smi
if ! python3 -c "import numpy, transformers, accelerate, pytest, triton; v=tuple(int(x) for x in transformers.__version__.split('.')[:2]); assert v >= (4, 51)" >/dev/null 2>&1; then
  python3 -m pip install --break-system-packages -q --upgrade "transformers>=4.51.0" accelerate sentencepiece protobuf pytest numpy
fi
cd /workspace/nanolm_repo
mkdir -p artifacts/nanolm_omega_v2
PYTHONPATH=. python3 -m pytest -q nanolm_holonomy/tests/
PYTHONPATH=. python3 -u -m nanolm_holonomy.cycle3_omega_v2 \
  --mode all \
  --out-dir artifacts/nanolm_omega_v2
EOF

echo "[runpod] Downloading JSON artifacts..."
ssh -n "${SSH_OPTS[@]}" "root@${SSH_HOST}" \
  "tar -C /workspace/nanolm_repo/artifacts/nanolm_omega_v2 -czf - ." \
  | tar --no-same-owner -xzf - -C "${OUT_DIR}"

POD_END_EPOCH="$(date +%s)"
POD_DETAILS_JSON="$(runpodctl get pod "${MY_POD_ID}" -a || true)"
WALL_SECS=$((POD_END_EPOCH - POD_START_EPOCH))

echo "[runpod] Terminating MY_POD_ID=${MY_POD_ID}..."
runpodctl remove pod "${MY_POD_ID}"
REMOVED_POD_ID="${MY_POD_ID}"
MY_POD_ID=""
rm -f "${POD_ID_FILE}"

BALANCE_AFTER_JSON="$(runpodctl user)"
BALANCE_AFTER="$(python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("clientBalance", 0.0))' <<< "${BALANCE_AFTER_JSON}")"

POD_DETAILS_JSON="${POD_DETAILS_JSON}" \
WALL_SECS="${WALL_SECS}" \
REMOVED_POD_ID="${REMOVED_POD_ID}" \
CHOSEN_GPU="${CHOSEN_GPU}" \
IMAGE_NAME="${IMAGE_NAME}" \
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
    "wall_seconds": wall_secs,
    "wall_minutes": round(wall_secs / 60.0, 2),
    "cost_per_hr_usd": cost_per_hr if cost_per_hr > 0 else None,
    "pod_attributed_spend_usd": estimated_pod_usd,
    "account_balance_before_usd": round(bal_before, 4),
    "account_balance_after_usd": round(bal_after, 4),
    "account_balance_delta_usd": round(bal_before - bal_after, 4),
}
Path(os.environ["OUT_DIR"], "SPEND_AND_PROVENANCE.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
print("SPEND_PROVENANCE:", json.dumps(prov))
PY

echo "[runpod] Omega-v2 campaign complete and pod verified terminated."
