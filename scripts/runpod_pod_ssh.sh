#!/usr/bin/env bash
# Resolve RunPod pod SSH (direct IP or ssh.runpod.io proxy) and run a remote command.
set -euo pipefail

POD_ID="${1:?pod id required}"
shift
REMOTE_CMD="${*:-echo ok}"

SSH_KEY="${RUNPOD_SSH_KEY:-${HOME}/.runpod/ssh/runpodctl-ssh-key}"

ssh_command_from_api() {
  local pod_id="$1"
  local cmd=""
  cmd="$(runpodctl pod get "$pod_id" -o json 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d.get('ssh', {}).get('ssh_command') or d.get('sshCmd', '') or '')
" 2>/dev/null || true)"
  if [[ -z "$cmd" ]]; then
    cmd="$(runpodctl ssh info "$pod_id" -o json 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d.get('ssh_command', '') or d.get('command', '') or '')
" 2>/dev/null || true)"
  fi
  printf '%s' "$cmd"
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

run_direct() {
  local ssh_cmd="$1"
  local cmd="$2"
  # shellcheck disable=SC2086
  bash -c "${ssh_cmd} -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new $(printf %q "$cmd")" 2>/dev/null
}

run_proxy() {
  local host_id="$1"
  local cmd="$2"
  ssh -T -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new \
    -i "$SSH_KEY" "${host_id}@ssh.runpod.io" \
    "bash -lc $(printf %q "$cmd")" 2>&1 \
    | grep -v -F 'WARNING: Remote port forwarding' \
    | grep -v -F 'Error: Your SSH client' \
    | grep -v -F -- '-- RUNPOD' \
    | grep -v -F 'Enjoy your Pod' \
    | grep -v -F 'docs.runpod.io' \
    | grep -v -F 'For detailed documentation' \
    | grep -v -E '^[[:space:]]*$' \
    | grep -v -E '^root@.*:.*#$'
}

SSH_CMD="$(ssh_command_from_api "$POD_ID")"
if [[ -n "$SSH_CMD" ]]; then
  if out="$(run_direct "$SSH_CMD" "$REMOTE_CMD")" && [[ -n "$out" ]]; then
    printf '%s\n' "$out"
    exit 0
  fi
fi

HOST_ID="$(pod_host_id "$POD_ID")"
if [[ -z "$HOST_ID" ]]; then
  echo "ssh unavailable for pod $POD_ID (no ssh_command or podHostId)" >&2
  exit 1
fi

if ! [[ -f "$SSH_KEY" ]]; then
  echo "missing RunPod SSH key at $SSH_KEY (run runpodctl doctor)" >&2
  exit 1
fi

run_proxy "$HOST_ID" "$REMOTE_CMD"
