#!/bin/bash
# Runs on a Cloud TPU VM.  usage: vm_ladder.sh <gs://bucket> <run-name> "<size> <lr> <micro>;<size> <lr> <micro>;..."
# Trains each listed model for one pass, pushes each result and the log to the bucket, then writes a DONE marker.
set -u
B=$1; RUN=$2; SPECS=$3
cd ~; mkdir -p data out
LOG=~/$RUN.log
log(){ echo "$(date -u +%H:%M:%S) $*" >> $LOG; gcloud storage cp $LOG $B/runs/$RUN/log.txt >/dev/null 2>&1; }
log "setup start on $(hostname)"
pip install -q -U "jax[tpu]==0.6.2" "optax==0.2.9" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html >> $LOG 2>&1 \
  || pip install -q -U "jax[tpu]" optax -f https://storage.googleapis.com/jax-releases/libtpu_releases.html >> $LOG 2>&1
[ -s data/train.bin ] || gcloud storage cp "$B/corpus/v1/*" data/ >> $LOG 2>&1
ls -la data >> $LOG 2>&1
python3 -c "import jax, optax; print('jax', jax.__version__, 'optax', optax.__version__, jax.devices())" >> $LOG 2>&1
log "setup done"
IFS=';' read -ra LIST <<< "$SPECS"
for spec in "${LIST[@]}"; do
  set -- $spec
  log "start size $1 lr $2 micro $3"
  python3 train_jax.py --size $1 --lr $2 --micro $3 --data ~/data --out ~/out --gcs $B/runs/$RUN --save >> $LOG 2>&1; rc=$?
  if [ $rc -ne 0 ]; then
    log "failed rc $rc; one retry with micro $(( $3 * 2 ))"
    python3 train_jax.py --size $1 --lr $2 --micro $(( $3 * 2 )) --data ~/data --out ~/out --gcs $B/runs/$RUN --save >> $LOG 2>&1; rc=$?
  fi
  log "end size $1 lr $2 rc $rc"
done
echo done > ~/DONE-$RUN; gcloud storage cp ~/DONE-$RUN $B/runs/$RUN/DONE; log "all done"
