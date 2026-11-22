#!/bin/bash
# Runs on a Cloud TPU VM.
# usage: vm_all.sh <gs://bucket> <tpu-run-name> "<multi spec>" <project> <zone> <node> <max_hours> [nested-run-name]
# Trains the listed models (one per chip at a time), optionally runs the nested protocol on the host CPUs,
# pushes results and the log to the bucket, then deletes this machine. A hard time limit also deletes it,
# whatever else happens. Nothing here depends on any other computer staying on.
set -u
B=$1; RUN=$2; SPEC=$3; P=$4; Z=$5; NODE=$6; MAXH=$7; NESTED=${8:-}
export HOME=${HOME:-/root}; cd ~; mkdir -p data out; LOG=~/$RUN.log
tok(){ curl -s -H Metadata-Flavor:Google "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])'; }
pushlog(){ gcloud storage cp $LOG $B/runs/$RUN/log.txt >/dev/null 2>&1; }
log(){ echo "$(date -u +%H:%M:%S) $*" >> $LOG; pushlog; }
selfdelete(){ log "deleting this machine: $1"; curl -s -X DELETE -H "Authorization: Bearer $(tok)" "https://tpu.googleapis.com/v2/projects/$P/locations/$Z/nodes/$NODE" >> $LOG 2>&1; pushlog; }
( sleep $(python3 -c "print(int(float('$MAXH')*3600))"); selfdelete "hard limit of $MAXH hours reached" ) &
( while true; do sleep 180; pushlog; done ) &
log "start on $(hostname): $(nproc) cpus; this machine can see its own record: HTTP $(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $(tok)" "https://tpu.googleapis.com/v2/projects/$P/locations/$Z/nodes/$NODE")"
NPID=""
if [ -n "$NESTED" ]; then ( bash ~/onto/tpu/vm_nested.sh $B $NESTED ) & NPID=$!; fi
pip install -q -U "jax[tpu]==0.6.2" "optax==0.2.9" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html >> $LOG 2>&1 \
  || pip install -q -U "jax[tpu]" optax -f https://storage.googleapis.com/jax-releases/libtpu_releases.html >> $LOG 2>&1
[ -s data/train.bin ] || gcloud storage cp "$B/corpus/v1/*" data/ >> $LOG 2>&1
ls -la data >> $LOG 2>&1
python3 -c "import jax, optax; print('jax', jax.__version__, 'optax', optax.__version__, jax.devices())" >> $LOG 2>&1
log "setup done"
python3 ~/onto/tpu/train_jax.py --data ~/data --out ~/out --gcs $B/runs/$RUN --save --resume --multi "$SPEC" >> $LOG 2>&1; rc=$?
log "training finished with exit code $rc"
echo "rc $rc" > ~/DONE-$RUN; gcloud storage cp ~/DONE-$RUN $B/runs/$RUN/DONE >/dev/null 2>&1
if [ -n "$NPID" ]; then log "waiting for the CPU runs"; wait $NPID; log "CPU runs finished"; fi
selfdelete "all work finished"
