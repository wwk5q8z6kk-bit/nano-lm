#!/bin/bash
# Training-speed benchmark on one TPU chip at the 386M model's per-chip load (16 sequences of 512 per step).
# Arms: current setup (batch in 2 passes), --remat (1 pass), --attn splash (1 pass, then 2 if it does not fit), both.
# Uses the first 100M tokens of corpus v2 only; speed, not quality, is measured (loss is printed as a sanity check).
# usage: bench_speed.sh <gs://bucket>      results: <bucket>/runs/bench-speed/
set -u
B=$1; D=$(cd "$(dirname "$0")" && pwd); O=~/bench; mkdir -p ~/d $O; LOG=~/bench.log
log(){ echo "$(date -u +%H:%M:%S) $*" | tee -a $LOG; gcloud storage cp $LOG $B/runs/bench-speed/log.txt >/dev/null 2>&1; }
pip install -q -U "jax[tpu]==0.6.2" "optax==0.2.9" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html >> $LOG 2>&1
for f in manifest.json tokenizer_32k.json val_web.bin val_textbook.bin val_math.bin val_code.bin; do gcloud storage cp $B/corpus/v2/$f ~/d/ >/dev/null 2>&1; done
gcloud storage cat -r 0-199999999 $B/corpus/v2/train.bin > ~/d/train.bin
log "data ready: $(stat -c %s ~/d/train.bin) bytes of train.bin"
C="--size XXL --lr 2e-3 --bs 16 --data $HOME/d --max-train-tokens 100000000 --train-tokens 8.192e6 --eval-every 500 --out $O --opt muon --wd 0.1 --lr-floor 0.01 --tag bench"
arm(){ name=$1; shift; python3 $D/train_jax.py $C "$@" > $O/$name.out 2>&1; rc=$?
  r=$(grep -h '"tokens_per_second"' $O/$name.out | tail -1 | python3 -c "import json,sys; l=sys.stdin.read().strip(); d=json.loads(l) if l else {}; print(d.get('tokens_per_second'), d.get('train_loss'), d.get('device_gb_in_use'))")
  log "$name exit $rc: tokens/s, loss, GB = $r"; grep -m1 -E "RESOURCE_EXHAUSTED|Error|error" $O/$name.out | cut -c1-300 >> $LOG; return $rc; }
arm base-micro2 --micro 2
arm remat-micro1 --micro 1 --remat
arm splash-micro1 --micro 1 --attn splash || arm splash-micro2 --micro 2 --attn splash
arm remat-splash-micro1 --micro 1 --remat --attn splash
gcloud storage cp $O/*.out $B/runs/bench-speed/ >/dev/null 2>&1
log "bench finished"
