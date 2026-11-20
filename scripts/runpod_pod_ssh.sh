#!/usr/bin/env bash
# Resolve RunPod pod SSH (direct IP or ssh.runpod.io proxy) and run a remote command.
set -euo pipefail

POD_ID="${1:?pod id required}"
shift
REMOTE_CMD="${*:-echo ok}"

SSH_KEY="${RUNPOD_SSH_KEY:-${HOME}/.runpod/ssh/runpodctl-ssh-key}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if ! [[ "$POD_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$ ]]; then
  echo "invalid pod ID" >&2
  exit 1
fi

# Provision the endpoint's public key through a trusted channel in ~/.ssh/known_hosts.
SSH_OPTIONS=(-F /dev/null -o StrictHostKeyChecking=yes -o VerifyHostKeyDNS=no -o BatchMode=yes)

direct_endpoint() {
  python3 - "$SCRIPT_DIR" "$POD_ID" <<'PYHOST'
import sys
sys.path.insert(0, sys.argv[1])
from pull_native_weights import resolve_direct_scp
endpoint = resolve_direct_scp(sys.argv[2])
if endpoint:
    print(endpoint[0], endpoint[1])
PYHOST
}

pod_host_id() {
  local pod_id="$1"
  python3 - "$pod_id" <<'PY'
import json, os, re, subprocess, sys

pod_id = sys.argv[1]
cfg_path = os.path.expanduser("~/.runpod/config.toml")
if not os.path.isfile(cfg_path):
    sys.exit(0)
cfg = open(cfg_path).read()
m = re.search(r"apikey\s*=\s*'([^']+)'", cfg)
if not m:
    sys.exit(0)
api = m.group(1)
query = json.dumps(
    {"query": f'query {{ pod(input: {{podId: "{pod_id}"}}) {{ machine {{ podHostId }} }} }}'}
)
out = subprocess.check_output(
    [
        "curl", "-s", "-X", "POST", "https://api.runpod.io/graphql",
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: {api}",
        "-d", query,
    ],
    text=True,
)
pod = (json.loads(out).get("data") or {}).get("pod") or {}
print(pod.get("machine", {}).get("podHostId", "") or "")
PY
}

run_proxy() {
  local host_id="$1"
  local cmd="$2"
  ssh -tt "${SSH_OPTIONS[@]}" -o ConnectTimeout=45 \
    -i "$SSH_KEY" "${host_id}@ssh.runpod.io" \
    "bash -lc $(printf %q "$cmd"); exit" 2>&1 \
    | grep -v -F 'WARNING: connection is not using' \
    | grep -v -F 'WARNING: Remote port forwarding' \
    | grep -v -F 'Error: Your SSH client' \
    | grep -v -F -- '-- RUNPOD' \
    | grep -v -F 'Enjoy your Pod' \
    | grep -v -F 'docs.runpod.io' \
    | grep -v -F 'blog.runpod.io' \
    | grep -v -F 'For detailed documentation' \
    | grep -v -E '^[[:space:]]*$' \
    | grep -v -E '^root@.*:.*#$'
}

if read -r host port < <(direct_endpoint) && [[ -n "$host" && -n "$port" ]]; then
  if out="$(ssh "${SSH_OPTIONS[@]}" -o ConnectTimeout=20 -i "$SSH_KEY" -p "$port" \
    "root@$host" "$REMOTE_CMD")"; then
    printf '%s\n' "$out"
    exit 0
  fi
fi

HOST_ID="$(pod_host_id "$POD_ID")"
if ! [[ "$HOST_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "ssh unavailable for pod $POD_ID (no ssh_command or podHostId)" >&2
  exit 1
fi

if ! [[ -f "$SSH_KEY" ]]; then
  echo "missing RunPod SSH key at $SSH_KEY (run runpodctl doctor)" >&2
  exit 1
fi

run_proxy "$HOST_ID" "$REMOTE_CMD"
