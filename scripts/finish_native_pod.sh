#!/usr/bin/env bash
# Wait for native training, pull weights (SCP-first when direct TCP exposed), verify, terminate.
# Weight pull order: direct TCP scp via runpodctl port mapping, then ssh base64 chunks.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

POD_ID="${1:?pod id required}"
RUN_ID="${2:-native100_evidence_bottleneck_s1}"
MARKER="${NATIVE_DONE_MARKER:-NATIVE100_EXTENDED_DONE}"
LOG_PATH="${NATIVE_LOG_PATH:-/workspace/native100_extended.log}"
MIN_BYTES="${NATIVE_MIN_WEIGHT_BYTES:-1000000}"
MAX_POLLS="${NATIVE_MAX_POLLS:-90}"
POLL_SEC="${NATIVE_POLL_SEC:-20}"

echo "==> Monitoring pod ${POD_ID} for ${MARKER} (run=${RUN_ID})"

for i in $(seq 1 "${MAX_POLLS}"); do
  OUT=$(bash scripts/runpod_pod_ssh.sh "${POD_ID}" \
    "grep -q '${MARKER}' ${LOG_PATH} 2>/dev/null && echo DONE || echo RUNNING; ls -la /workspace/nano-lm/artifacts/native_checkpoints/${RUN_ID}/latest.pt 2>/dev/null || echo NO_WEIGHTS" 2>&1 || true)
  echo "poll ${i}: ${OUT}"
  if echo "${OUT}" | grep -q DONE; then
    break
  fi
  if [[ "${i}" -eq "${MAX_POLLS}" ]]; then
    echo "ABORT_TERMINATE: training marker ${MARKER} not seen" >&2
    exit 1
  fi
  sleep "${POLL_SEC}"
done

echo "==> Pulling weights for ${RUN_ID}"
MAX_PULL_ATTEMPTS="${NATIVE_MAX_PULL_ATTEMPTS:-5}"
PULL_OK=0
for attempt in $(seq 1 "${MAX_PULL_ATTEMPTS}"); do
  echo "pull attempt ${attempt}/${MAX_PULL_ATTEMPTS}"
  if python3 scripts/pull_native_weights.py "${POD_ID}" "${RUN_ID}"; then
    PULL_OK=1
    break
  fi
  echo "pull attempt ${attempt} failed; retrying in 10s" >&2
  sleep 10
done
if [[ "${PULL_OK}" -ne 1 ]]; then
  echo "ABORT_TERMINATE: weight pull failed after ${MAX_PULL_ATTEMPTS} attempts — pod ${POD_ID} left running for retry" >&2
  exit 1
fi

LOCAL_PT="artifacts/native_checkpoints/${RUN_ID}/latest.pt"
VERIFY_JSON=$(python3 scripts/pull_native_weights.py "${POD_ID}" "${RUN_ID}" --verify-only)
echo "${VERIFY_JSON}"
if ! echo "${VERIFY_JSON}" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('ok') else 1)"; then
  echo "ABORT_TERMINATE: torch.load verify failed — pod ${POD_ID} left running" >&2
  exit 1
fi
SIZE=$(stat -f%z "${LOCAL_PT}" 2>/dev/null || stat -c%s "${LOCAL_PT}")
echo "WEIGHT_PULL_OK path=${LOCAL_PT} bytes=${SIZE} torch_load=ok"

echo "==> Pulling step JSON metrics"
bash scripts/runpod_pod_ssh.sh "${POD_ID}" \
  "for f in /workspace/nano-lm/artifacts/native_checkpoints/${RUN_ID}/step_*.json; do echo ===FILE=== \$f; cat \$f; done" \
  | python3 - <<'PY'
import json, sys
from pathlib import Path
buf = sys.stdin.read()
for part in buf.split("===FILE=== ")[1:]:
    lines = part.strip().split("\n", 1)
    if len(lines) < 2:
        continue
    remote = lines[0].strip()
    name = Path(remote).name
    data = json.loads(lines[1])
    run_id = data["config"]["run_id"]
    out = Path("artifacts/native_checkpoints") / run_id / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n")
    print("wrote", out)
PY

if [[ "${SKIP_TERMINATE:-0}" == "1" ]]; then
  echo "SKIP_TERMINATE=1 — pod ${POD_ID} left running for post-pull eval"
  echo "finish_ok run_id=${RUN_ID} weights=${LOCAL_PT} bytes=${SIZE}"
  exit 0
fi

echo "==> Terminating pod ${POD_ID}"
runpodctl pod delete "${POD_ID}"
echo "POD_TERMINATED ${POD_ID}"
runpodctl pod list
echo "finish_ok run_id=${RUN_ID} weights=${LOCAL_PT} bytes=${SIZE}"
