#!/bin/bash
# Deletes this TPU node. Installed by vm_deadline.sh; run by a systemd timer at the node's deadline.
# usage: tpu_deadline_delete.sh <project> <zone> <node>
# Sends DELETE first (time-limited), then reads the node's state, and repeats every 30 s until the TPU API
# reports DELETING or the node is gone. Nothing (no upload, no log push) runs before the request.
P=$1; Z=$2; N=$3
URL="https://tpu.googleapis.com/v2/projects/$P/locations/$Z/nodes/$N"; LOG=/var/log/tpu-deadline.log
tok(){ curl -s --max-time 10 -H Metadata-Flavor:Google "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])' 2>/dev/null; }
for i in $(seq 1 240); do
  T=$(tok)
  D=$(curl -s --max-time 30 -o /dev/null -w '%{http_code}' -X DELETE -H "Authorization: Bearer $T" "$URL")
  S=$(curl -s --max-time 30 -H "Authorization: Bearer $T" "$URL" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("state") or d.get("error",{}).get("code"))' 2>/dev/null)
  echo "$(date -u +%FT%TZ) attempt $i: DELETE HTTP $D, node state $S" >> $LOG
  case "$S" in DELETING|404) exit 0;; esac
  sleep 30
done
exit 1
