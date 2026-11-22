#!/bin/bash
# Controller for a fresh TPU VM. usage: vm_run.sh <gs://bucket> <run> <project> <zone> <node> "<deadline UTC YYYY-MM-DD HH:MM:SS>"
# Installs the persistent deadline first, then JAX, then runs ~/queue/ (job_queue.sh), then deletes the machine
# with the same retrying, DELETE-first script the deadline uses. The log goes to the bucket every 3 minutes.
set -u
B=$1; RUN=$2; P=$3; Z=$4; NODE=$5; DL=$6
D=$(cd "$(dirname "$0")" && pwd); LOG=~/$RUN.log; cd ~
pushlog(){ timeout 60 gcloud storage cp $LOG $B/runs/$RUN/log.txt >/dev/null 2>&1; }
log(){ echo "$(date -u +%H:%M:%S) $*" >> $LOG; pushlog; }
sudo bash $D/vm_deadline.sh $P $Z $NODE "$DL" >> $LOG 2>&1 || { log "deadline could not be installed; deleting now"; sudo bash $D/tpu_deadline_delete.sh $P $Z $NODE; exit 1; }
( while true; do sleep 180; pushlog; done ) &
log "start on $(hostname), $(nproc) cpus; deadline $DL UTC"
pip install -q -U "jax[tpu]==0.6.2" "optax==0.2.9" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html >> $LOG 2>&1
python3 -c "import jax, optax; print('jax', jax.__version__, 'optax', optax.__version__, jax.devices())" >> $LOG 2>&1
B=$B bash $D/job_queue.sh >> $LOG 2>&1
log "queue finished; deleting this machine"
sudo bash $D/tpu_deadline_delete.sh $P $Z $NODE
