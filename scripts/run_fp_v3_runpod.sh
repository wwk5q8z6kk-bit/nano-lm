#!/usr/bin/env bash
set -euo pipefail

# RunPod A40 (48GB native BF16) Stage-B runner for NANOLM FP-v3 EXP-FP4 (1.747B on 86 tasks).
# Auto-terminates ONLY its own pod on exit and preserves Stage-A Kaggle JSONs in artifacts/nanolm_fp_v3/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${WORKTREE_DIR}/artifacts/nanolm_fp_v3"
POD_ID_FILE="/tmp/nanolm_fp_v3_pod_id.txt"
SSH_KEY="${HOME}/.runpod/ssh/runpodctl-ssh-key"
IMAGE_NAME="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

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
    echo "[runpod-fp4] Terminating ONLY own pod MY_POD_ID=${MY_POD_ID}..."
    runpodctl remove pod "${MY_POD_ID}" >/dev/null 2>&1 || true
    rm -f "${POD_ID_FILE}"
  fi
  exit "${exit_code}"
}
trap cleanup EXIT

BALANCE_BEFORE_JSON="$(runpodctl user)"
BALANCE_BEFORE="$(python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("clientBalance", 0.0))' <<< "${BALANCE_BEFORE_JSON}")"
echo "[runpod-fp4] Account balance before launch: \$${BALANCE_BEFORE}"

CAN_LAUNCH="$(python3 -c 'import sys; b=float(sys.argv[1]); print("yes" if b > 28.0 else "no")' "${BALANCE_BEFORE}")"
if [[ "${CAN_LAUNCH}" != "yes" ]]; then
  echo "[runpod-fp4] ERROR: Account balance \$${BALANCE_BEFORE} too close to \$20 owner reserve." >&2
  exit 1
fi

GPU_TYPES=("NVIDIA A40" "NVIDIA L40S" "NVIDIA RTX A6000")
CREATE_OUT=""
CHOSEN_GPU=""
for GPU in "${GPU_TYPES[@]}"; do
  echo "[runpod-fp4] Attempting secureCloud pod creation with GPU=${GPU}..."
  if CREATE_OUT="$(runpodctl create pod \
      --name "nanolm-fp4-bf16-9946" \
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
  echo "[runpod-fp4] GPU ${GPU} unavailable: ${CREATE_OUT}"
done

if [[ -z "${CHOSEN_GPU}" ]]; then
  echo "[runpod-fp4] ERROR: All candidate 48GB GPU types failed." >&2
  exit 1
fi

MY_POD_ID="$(python3 -c '
import re, sys
txt = sys.argv[1]
m = re.search(r"pod\s+\"?([a-z0-9]{10,20})\"?", txt, re.I) or re.search(r"\"([a-z0-9]{10,20})\"", txt)
print(m.group(1) if m else "")
' "${CREATE_OUT}")"

if [[ -z "${MY_POD_ID}" ]]; then
  echo "[runpod-fp4] ERROR: Could not parse MY_POD_ID from ${CREATE_OUT}" >&2
  exit 1
fi

echo "${MY_POD_ID}" > "${POD_ID_FILE}"
POD_START_EPOCH="$(date +%s)"
echo "[runpod-fp4] Created MY_POD_ID=${MY_POD_ID} (${CHOSEN_GPU}). Starting 60m watchdog..."

(
  sleep 3600
  runpodctl remove pod "${MY_POD_ID}" >/dev/null 2>&1 || true
) &
WATCHDOG_PID=$!

SSH_HOST=""
SSH_PORT=""
SSH_READY_OK=0
for attempt in $(seq 1 100); do
  INFO_JSON="$(runpodctl ssh info "${MY_POD_ID}" 2>&1 || true)"
  PARSED="$(python3 -c '
import json, re, sys
txt = sys.argv[1]
ip = port = ""
m_json = re.search(r"\{.*\}", txt, flags=re.DOTALL)
if m_json:
    try:
        d = json.loads(m_json.group(0))
        ip = str(d.get("ip") or d.get("publicIp") or d.get("host") or "")
        port = str(d.get("port") or d.get("sshPort") or "")
        cmd = str(d.get("command") or "")
        if (not ip or not port) and cmd:
            mc = re.search(r"@([0-9.]+)\s+-p\s+(\d+)", cmd) or re.search(r"-p\s+(\d+)\s+\S+@([0-9.]+)", cmd)
            if mc:
                if "@" in mc.group(0).split()[0]:
                    ip, port = mc.group(1), mc.group(2)
                else:
                    port, ip = mc.group(1), mc.group(2)
    except Exception:
        pass
if not ip or not port:
    mc = re.search(r"@([0-9.]+)\s+-p\s+(\d+)", txt) or re.search(r"([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+):(\d+)", txt)
    if mc:
        ip, port = mc.group(1), mc.group(2)
if ip and port:
    print(f"{ip} {port}")
' "${INFO_JSON}")"
  if [[ -n "${PARSED}" ]]; then
    SSH_HOST="$(awk '{print $1}' <<< "${PARSED}")"
    SSH_PORT="$(awk '{print $2}' <<< "${PARSED}")"
    if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=6 \
        -i "${SSH_KEY}" -p "${SSH_PORT}" "root@${SSH_HOST}" "echo SSH_READY" </dev/null 2>/dev/null | grep -q "SSH_READY"; then
      echo "[runpod-fp4] SSH ready on ${SSH_HOST}:${SSH_PORT} (attempt ${attempt})"
      SSH_READY_OK=1
      break
    fi
  elif [[ "${attempt}" -eq 12 || "${attempt}" -eq 35 ]]; then
    echo "[runpod-fp4] Waiting for SSH (attempt ${attempt} raw info): ${INFO_JSON}"
  fi
  sleep 6
done

if [[ "${SSH_READY_OK}" -ne 1 ]]; then
  echo "[runpod-fp4] ERROR: SSH failed for ${MY_POD_ID}." >&2
  exit 1
fi

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -i "${SSH_KEY}" -p "${SSH_PORT}")

echo "[runpod-fp4] Uploading codebase (including read-only nano & fabric)..."
ssh -n "${SSH_OPTS[@]}" "root@${SSH_HOST}" "mkdir -p /workspace/nanolm_repo"
COPYFILE_DISABLE=1 tar -C "${WORKTREE_DIR}" --no-xattrs --exclude='__pycache__' --exclude='*.pyc' -czf - nanolm_holonomy nanolm_frontier nanolm_posttrain nano fabric \
  | ssh "${SSH_OPTS[@]}" "root@${SSH_HOST}" "tar --no-same-owner --no-same-permissions -xzf - -C /workspace/nanolm_repo"

echo "[runpod-fp4] Running EXP-FP4 (1.747B BF16 on FP-5Axis-30 + benchmark_15d)..."
ssh "${SSH_OPTS[@]}" "root@${SSH_HOST}" bash -s <<'EOF'
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/workspace/hf_cache
mkdir -p /workspace/hf_cache
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
if ! python3 -c "import numpy, transformers, accelerate, pytest; v=tuple(int(x) for x in transformers.__version__.split('.')[:2]); assert v >= (4, 51)" >/dev/null 2>&1; then
  python3 -m pip install --break-system-packages -q --upgrade "transformers>=4.51.0" accelerate sentencepiece protobuf pytest numpy
fi
cd /workspace/nanolm_repo
mkdir -p artifacts/nanolm_fp_v3_stage_b
PYTHONPATH=. python3 -u -m nanolm_holonomy.cycle4_first_principles \
  --mode fp4 \
  --out-dir artifacts/nanolm_fp_v3_stage_b
EOF

echo "[runpod-fp4] Downloading Stage-B JSON artifacts..."
mkdir -p /tmp/fp4_stage_b_dl
ssh -n "${SSH_OPTS[@]}" "root@${SSH_HOST}" \
  "tar -C /workspace/nanolm_repo/artifacts/nanolm_fp_v3_stage_b -czf - ." \
  | tar --no-same-owner -xzf - -C /tmp/fp4_stage_b_dl
cp -f /tmp/fp4_stage_b_dl/exp_fp4_1p7b_integration.json "${OUT_DIR}/"
cp -f /tmp/fp4_stage_b_dl/nanolm_fp_v3_1.7b_manifest.json "${OUT_DIR}/"

POD_END_EPOCH="$(date +%s)"
POD_DETAILS_JSON="$(runpodctl get pod "${MY_POD_ID}" -a || true)"
WALL_SECS=$((POD_END_EPOCH - POD_START_EPOCH))

echo "[runpod-fp4] Terminating MY_POD_ID=${MY_POD_ID}..."
runpodctl remove pod "${MY_POD_ID}"
REMOVED_POD_ID="${MY_POD_ID}"
MY_POD_ID=""
rm -f "${POD_ID_FILE}"

BALANCE_AFTER_JSON="$(runpodctl user)"
BALANCE_AFTER="$(python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("clientBalance", 0.0))' <<< "${BALANCE_AFTER_JSON}")"

OUT_DIR="${OUT_DIR}" \
WALL_SECS="${WALL_SECS}" \
REMOVED_POD_ID="${REMOVED_POD_ID}" \
CHOSEN_GPU="${CHOSEN_GPU}" \
IMAGE_NAME="${IMAGE_NAME}" \
BALANCE_BEFORE="${BALANCE_BEFORE}" \
BALANCE_AFTER="${BALANCE_AFTER}" \
python3 - <<'PYEOF'
import json, os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
gpu = os.environ["CHOSEN_GPU"]
rate = 0.40 if "A40" in gpu else (0.86 if "L40S" in gpu else 0.79)
wall_s = int(os.environ["WALL_SECS"])
est_usd = round((wall_s / 3600.0) * rate, 4)
bal_before = float(os.environ["BALANCE_BEFORE"])
bal_after = float(os.environ["BALANCE_AFTER"])
delta_usd = round(max(0.0, bal_before - bal_after), 4)

stage_b = {
    "stage": "stage_b_exp_fp4",
    "backend": "runpod_secure_cloud",
    "gpu": gpu,
    "pod_id": os.environ["REMOVED_POD_ID"],
    "image": os.environ["IMAGE_NAME"],
    "wall_seconds": wall_s,
    "rate_usd_per_hr": rate,
    "estimated_usd": est_usd,
    "account_balance_before_usd": round(bal_before, 4),
    "account_balance_after_usd": round(bal_after, 4),
    "account_balance_delta_usd": delta_usd,
    "pod_terminated": True,
}
(out_dir / "SPEND_STAGE_B_RUNPOD.json").write_text(json.dumps(stage_b, indent=2) + "\n", encoding="utf-8")
print("[runpod-fp4] Stage B provenance written:", json.dumps(stage_b, indent=2))
PYEOF
echo "[runpod-fp4] COMPLETE!"
