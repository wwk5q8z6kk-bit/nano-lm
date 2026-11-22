#!/bin/bash
# Gives this TPU VM a deadline that no controller process, ssh session, laptop or reboot can remove.
# usage: sudo bash vm_deadline.sh <project> <zone> <node> "<YYYY-MM-DD HH:MM:SS>"     (time in UTC, must be in the future)
# Installs /usr/local/bin/tpu-deadline-delete and a persistent systemd timer (it also fires at boot if the
# deadline passed while the machine was down), then checks without deleting anything that this machine's
# identity can read its node record and holds the permission to delete it.
set -eu
P=$1; Z=$2; N=$3; WHEN=$4
DIR=$(cd "$(dirname "$0")" && pwd)
NOW=$(date -u +%s); AT=$(date -u -d "$WHEN" +%s)
[ "$AT" -gt "$((NOW + 600))" ] || { echo "refusing: deadline $WHEN UTC is less than 10 minutes away"; exit 1; }
systemd-analyze calendar "$WHEN UTC" | grep -E "Normalized|Next elapse"
install -m 755 "$DIR/tpu_deadline_delete.sh" /usr/local/bin/tpu-deadline-delete
printf '[Unit]\nDescription=Delete TPU node %s at its deadline\n[Service]\nType=oneshot\nExecStart=/usr/local/bin/tpu-deadline-delete %s %s %s\n' "$N" "$P" "$Z" "$N" > /etc/systemd/system/tpu-deadline.service
printf '[Unit]\nDescription=Deadline for TPU node %s\n[Timer]\nOnCalendar=%s UTC\nPersistent=true\nAccuracySec=1s\n[Install]\nWantedBy=timers.target\n' "$N" "$WHEN" > /etc/systemd/system/tpu-deadline.timer
systemctl daemon-reload
systemctl enable --now tpu-deadline.timer >/dev/null 2>&1
echo "timer: $(systemctl is-enabled tpu-deadline.timer), $(systemctl is-active tpu-deadline.timer); next: $(systemctl show tpu-deadline.timer -p NextElapseUSecRealtime --value)"
T=$(curl -s --max-time 10 -H Metadata-Flavor:Google "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')
echo "reads own record: HTTP $(curl -s --max-time 30 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $T" "https://tpu.googleapis.com/v2/projects/$P/locations/$Z/nodes/$N")"
echo "permissions held: $(curl -s --max-time 30 -X POST -H "Authorization: Bearer $T" -H 'Content-Type: application/json' -d '{"permissions":["tpu.nodes.delete","tpu.nodes.get"]}' "https://cloudresourcemanager.googleapis.com/v1/projects/$P:testIamPermissions" | tr -d '\n ')"
