#!/usr/bin/env bash
# Quick CUDA availability check on a running pod — delete if torch.cuda unavailable.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
POD_ID="${1:?pod id required}"
OUT=$(bash scripts/runpod_pod_ssh.sh "${POD_ID}" \
  "python3 -c \"
import json, torch
ok = torch.cuda.is_available()
mem = 0
name = None
err = None
if ok:
    try:
        t = torch.zeros(256, 256, device='cuda')
        mem = int(torch.cuda.memory_allocated())
        name = torch.cuda.get_device_name(0)
        del t
        torch.cuda.empty_cache()
    except Exception as e:
        ok = False
        err = str(e)
print(json.dumps({'cuda': ok and mem > 0, 'mem': mem, 'name': name, 'err': err}))
\"" 2>&1)
echo "${OUT}"
if echo "${OUT}" | grep -q '"cuda": true'; then
  echo "CUDA_OK"
  exit 0
fi
echo "CUDA_FAIL: terminating pod ${POD_ID}" >&2
runpodctl pod delete "${POD_ID}" >/dev/null
exit 1
