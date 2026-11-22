#!/bin/bash
# Runs on a TPU VM that already has a controller (vm_all.sh, vm_more.sh or an earlier vm_chain.sh) in charge.
# usage: vm_chain.sh <gs://bucket> <previous-run> <this-run> <project> <zone> <node> <nested-run or "-"> <job script>
# Waits for the previous batch to finish, takes over the previous controller's final duty (deleting the
# machine), runs the job script, waits for the CPU runs if a nested run is named, then deletes the machine.
# The hard time limit started by the first controller is never touched and stays in force.
# The job script is run as:  B=<bucket> RUN=<this-run> LOG=<log file> bash <job script>
set -u
B=$1; PREV=$2; RUN=$3; P=$4; Z=$5; NODE=$6; NESTED=$7; JOB=$8
cd ~; LOG=~/$RUN.log; echo $$ > ~/controller-$RUN.pid
tok(){ curl -s -H Metadata-Flavor:Google "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])'; }
pushlog(){ gcloud storage cp $LOG $B/runs/$RUN/log.txt >/dev/null 2>&1; }
log(){ echo "$(date -u +%H:%M:%S) $*" >> $LOG; pushlog; }
selfdelete(){ log "deleting this machine: $1"; curl -s -X DELETE -H "Authorization: Bearer $(tok)" "https://tpu.googleapis.com/v2/projects/$P/locations/$Z/nodes/$NODE" >> $LOG 2>&1; pushlog; }
log "waiting for $PREV"
until [ -f ~/DONE-$PREV ]; do sleep 20; done
if [ -f ~/controller-$PREV.pid ]; then OLD=$(cat ~/controller-$PREV.pid); else OLD=$(pgrep -o -f "vm_more.sh gs://"); fi
if [ -n "$OLD" ] && [ "$OLD" != "$$" ]; then kill $OLD 2>/dev/null; log "took over from the previous controller (pid $OLD)"; fi
( while true; do sleep 180; pushlog; done ) &
B=$B RUN=$RUN LOG=$LOG bash $JOB >> $LOG 2>&1; rc=$?
log "job finished with exit code $rc"
echo "rc $rc" > ~/DONE-$RUN; gcloud storage cp ~/DONE-$RUN $B/runs/$RUN/DONE >/dev/null 2>&1
if [ "$NESTED" != "-" ]; then
  log "waiting for the CPU runs"
  until [ -f ~/DONE-$NESTED ]; do sleep 60; done
  log "CPU runs finished"
fi
selfdelete "all work finished"
