#!/bin/bash
# Runs on the TPU VM while vm_all.sh is already running there.
# usage: vm_more.sh <gs://bucket> <first-run-name> <second-run-name> "<multi spec>" <project> <zone> <node> [nested-run-name]
# Waits for the first batch to finish, takes over the final duties of vm_all.sh (so the machine is not
# deleted while the second batch runs), trains the second batch, waits for the CPU runs, then deletes the machine.
# The hard time limit started by vm_all.sh stays in force throughout.
set -u
B=$1; RUN1=$2; RUN=$3; SPEC=$4; P=$5; Z=$6; NODE=$7; NESTED=${8:-}
cd ~; LOG=~/$RUN.log
tok(){ curl -s -H Metadata-Flavor:Google "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])'; }
pushlog(){ gcloud storage cp $LOG $B/runs/$RUN/log.txt >/dev/null 2>&1; }
log(){ echo "$(date -u +%H:%M:%S) $*" >> $LOG; pushlog; }
selfdelete(){ log "deleting this machine: $1"; curl -s -X DELETE -H "Authorization: Bearer $(tok)" "https://tpu.googleapis.com/v2/projects/$P/locations/$Z/nodes/$NODE" >> $LOG 2>&1; pushlog; }
log "waiting for the first batch"
until [ -f ~/DONE-$RUN1 ]; do sleep 20; done
MAIN=$(pgrep -o -f "vm_all.sh gs://")
if [ -n "$MAIN" ]; then kill $MAIN; log "took over from the first controller (pid $MAIN)"; fi
( while true; do sleep 180; pushlog; done ) &
python3 ~/onto/tpu/train_jax.py --data ~/data --out ~/out --gcs $B/runs/$RUN --save --resume --multi "$SPEC" >> $LOG 2>&1; rc=$?
log "second batch finished with exit code $rc"
echo "rc $rc" > ~/DONE-$RUN; gcloud storage cp ~/DONE-$RUN $B/runs/$RUN/DONE >/dev/null 2>&1
if [ -n "$NESTED" ]; then
  log "waiting for the CPU runs"
  until [ -f ~/DONE-$NESTED ]; do sleep 60; done
  log "CPU runs finished"
fi
selfdelete "all work finished"
