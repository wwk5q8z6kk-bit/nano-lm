#!/bin/bash
# Laptop-side supervisor for a spot TPU running a queue job through vm_run.sh.
# Every 2 minutes: past the deadline -> delete the node and stop. Controller log says "queue finished" -> wait until the
# node is gone, then stop. Node preempted, stopped or gone without that line -> delete it, create a new spot node in the
# first zone with capacity, point the Cloud Scheduler delete at it, and start the same controller and queue entry.
# Training resumes from the bucket checkpoint (train_jax.py --resume). At most MAXR recreations.
# usage: spot_supervisor.sh <project> <node> <run> <queue entry name> <job script> "<deadline UTC YYYY-MM-DD HH:MM:SS>" <zone file>
set -u
P=$1; N=$2; RUN=$3; QN=$4; JOB=$5; DL=$6; ZF=$7; B=gs://nanolm-lab-eu-0478806098; MAXR=6; nr=0
ZONES="us-east1-c us-west1-c us-west4-a us-central1-a us-east5-a us-east5-b us-east5-c us-south1-a europe-west4-b europe-west4-a asia-southeast1-b"
DLS=$(date -j -u -f "%Y-%m-%d %H:%M:%S" "$DL" +%s)
say(){ echo "$(date -u +%H:%M:%S) $*"; }
state(){ gcloud compute tpus tpu-vm describe $N --zone $(cat $ZF) --project $P --format="value(state)" 2>/dev/null || echo GONE; }
delete(){ for i in 1 2 3 4 5; do gcloud compute tpus tpu-vm delete $N --zone $(cat $ZF) --project $P --quiet >/dev/null 2>&1; [ "$(state)" = GONE ] && return 0; sleep 60; done; return 1; }
launch(){
  for Z in $ZONES; do
    gcloud compute tpus tpu-vm create $N --zone $Z --project $P --accelerator-type v5litepod-4 --version v5e-ubuntu-2204 --spot >/dev/null 2>&1 || continue
    echo $Z > $ZF; say "created spot node in $Z"
    gcloud scheduler jobs update http tpu-deadline-$N --location us-west1 --project $P --uri "https://tpu.googleapis.com/v2/projects/$P/locations/$Z/nodes/$N" >/dev/null 2>&1 || say "WARNING: scheduler job not updated"
    for i in 1 2 3 4 5; do
      gcloud compute tpus tpu-vm ssh $N --zone $Z --project $P --quiet --ssh-flag="-o ConnectTimeout=30" --command "mkdir -p ~/onto ~/queue && gcloud storage cp $B/code/onto.tgz ~/onto.tgz >/dev/null 2>&1 && tar xzf ~/onto.tgz -C ~/onto && echo 'bash ~/onto/tpu/$JOB' > ~/queue/$QN && touch ~/controller.started && ( setsid nohup bash ~/onto/tpu/vm_run.sh $B $RUN $P $Z $N '$DL' > ~/vm_run.out 2>&1 < /dev/null & )" >/dev/null 2>&1 && { say "controller started in $Z"; return 0; }
      sleep 30
    done
    say "could not start the controller in $Z; deleting"; delete
  done
  return 1
}
while true; do
  now=$(date -u +%s)
  if [ $now -ge $DLS ]; then say "deadline reached"; delete && say "node deleted" || say "DELETE FAILED"; exit 0; fi
  st=$(state); lg=$(gcloud storage cat $B/runs/$RUN/log.txt 2>/dev/null | tail -3)
  if echo "$lg" | grep -q "queue finished"; then
    [ "$st" = GONE ] && { say "queue finished and node deleted"; exit 0; }
  elif [ "$st" = PREEMPTED ] || [ "$st" = TERMINATED ] || [ "$st" = STOPPED ] || [ "$st" = GONE ]; then
    say "node state $st without a finished queue ($nr recreations so far, at most $MAXR)"
    [ $nr -ge $MAXR ] && { say "too many recreations; stopping"; [ "$st" = GONE ] || delete; exit 1; }
    [ "$st" = GONE ] || delete || { say "DELETE FAILED"; }
    gcloud storage mv $B/runs/$RUN/log.txt $B/runs/$RUN/log-before-recreation-$((nr + 1)).txt >/dev/null 2>&1   # keep the old controller log
    if launch; then nr=$((nr + 1)); else say "no spot capacity in any zone; will retry"; fi
  fi
  sleep 120
done
