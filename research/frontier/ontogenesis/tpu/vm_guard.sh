#!/bin/bash
# Runs on a TPU VM.  usage: vm_guard.sh <gs://bucket> <run> <controller script name>
# Closes a race at the end of a batch. The controller of <run> deletes the machine as soon as its training
# ends if the CPU runs are already finished, and the next controller only checks every 20 seconds, so a
# queued batch could be lost. This guard waits until that controller has started its training process, ends
# the controller (the training process is not touched and keeps running), and writes the batch's DONE
# marker itself when training exits. The hard time limit is not touched.
set -u
B=$1; RUN=$2; CTRL=$3
cd ~; LOG=~/guard-$RUN.log
say(){ echo "$(date -u +%H:%M:%S) $*" >> $LOG; gcloud storage cp $LOG $B/runs/$RUN/guard-log.txt >/dev/null 2>&1; }
PAT="train_jax.py .*runs/$RUN "
say "waiting for the training process of $RUN"
until pgrep -f "$PAT" >/dev/null; do
  if [ -f ~/DONE-$RUN ]; then say "batch already finished; nothing to do"; exit 0; fi
  sleep 2
done
TP=$(pgrep -o -f "$PAT"); CP=$(ps -o ppid= -p $TP | tr -d ' ')
if ps -o args= -p $CP | grep -q "$CTRL"; then
  kill $CP; say "ended controller $CP; training process $TP continues"
else
  say "parent of $TP (pid $CP) is not $CTRL; left alone"; exit 1
fi
while kill -0 $TP 2>/dev/null; do sleep 5; done
say "training process $TP has exited"
gcloud storage cp ~/$RUN.log $B/runs/$RUN/log.txt >/dev/null 2>&1
echo "rc unknown (controller replaced by vm_guard.sh; see the result files)" > ~/DONE-$RUN
gcloud storage cp ~/DONE-$RUN $B/runs/$RUN/DONE >/dev/null 2>&1
say "wrote DONE-$RUN"
