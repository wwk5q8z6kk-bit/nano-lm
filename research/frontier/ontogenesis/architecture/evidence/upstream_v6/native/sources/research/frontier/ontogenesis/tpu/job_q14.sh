#!/bin/bash
# Question 14 (INVESTIGATION.md): head size 128 with fused attention against head size 64, 101M, 0.8B tokens, v5e-4.
set -u
RUN=q14-v5e; D=$(cd "$(dirname "$0")" && pwd); T="$D/train_jax.py"; OUT=$HOME/out14; LOGF=$HOME/queue/done/30-q14.log
log(){ echo "$(date -u +%H:%M:%S) $*"; }
( while true; do sleep 180; timeout 60 gcloud storage cp $LOGF $B/runs/$RUN/job-log.txt >/dev/null 2>&1; done ) & PUSHER=$!
mkdir -p ~/data2 $OUT
[ -s ~/data2/train.bin ] || gcloud storage cp "$B/corpus/v2/*" ~/data2/
[ "$(stat -c %s ~/data2/train.bin)" = 20000000000 ] || { log "corpus v2 incomplete"; exit 1; }
C="--data $HOME/data2 --out $OUT --gcs $B/runs/$RUN --save --resume --dp 4 --opt muon --wd 0.1 --lr-floor 0.01 --lr 2e-3 --train-tokens 0.8e9 --tag v2"
python3 $T --size L128 --seed 1000 --attn splash $C; log "c L128 splash s1000 exit $?"
python3 $T --size L --seed 1001 $C; log "b L s1001 exit $?"
python3 $T --size L128 --seed 1001 --attn splash $C; log "d L128 splash s1001 exit $?"
python3 $T --size L128 --seed 1000 $C; log "e L128 standard s1000 exit $?"
kill $PUSHER; gcloud storage cp $LOGF $B/runs/$RUN/job-log.txt >/dev/null 2>&1
