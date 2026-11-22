#!/bin/bash
# Question 13, second attempt at step 3 (INVESTIGATION.md, operational note 17:10 UTC 2026-11-22).
# The first attempt (job_scale2.sh) ran the three 101M rate checks; the 386M run then failed about a minute after
# starting, before its first checkpoint, and that script treated the failure as success. Its log was lost with the machine.
# Same design: rate 2e-3 (the registered rule's choice, written to runs/scale2-v5e/chosen-lr.txt), 7.7B tokens, batch 64 x 512.
# Changes, all operational: the batch may be split into 1, 2, 4 or 8 passes with gradients averaged (the same update);
# any failure is a failure; this job's own log goes to the bucket every 3 minutes.
set -u
RUN=scale2-v5e; D=$(cd "$(dirname "$0")" && pwd); T="$D/train_jax.py"; OUT=$HOME/out7; LOGF=$HOME/queue/done/20-scale2b.log
log(){ echo "$(date -u +%H:%M:%S) $*"; }
( while true; do sleep 180; timeout 60 gcloud storage cp $LOGF $B/runs/$RUN/job-scale2b-log.txt >/dev/null 2>&1; done ) &
PUSHER=$!
mkdir -p ~/data2 $OUT
[ -s ~/data2/train.bin ] || gcloud storage cp "$B/corpus/v2/*" ~/data2/
[ "$(stat -c %s ~/data2/train.bin)" = 20000000000 ] || { log "corpus v2 incomplete"; exit 1; }
C="--data $HOME/data2 --out $OUT --gcs $B/runs/$RUN --save --resume --dp 4 --opt muon --wd 0.1 --lr-floor 0.01 --tag v2"
LR=$(gcloud storage cat $B/runs/$RUN/chosen-lr.txt 2>/dev/null || echo 0.002); log "rate for 386M: $LR"
X="--size XXL --lr $LR --train-tokens 7.7e9 --ckpt-every 10000"
rc=1
for M in 1 2 4 8; do
  python3 $T $X --micro $M $C; rc=$?; log "XXL micro $M exit $rc"
  [ $rc = 0 ] || [ $rc = 2 ] && break
  ls $OUT/dense-XXL-*.state.npz >/dev/null 2>&1 && { log "a checkpoint exists, so this was not a start-up failure; retrying once with resume"; python3 $T $X --micro $M $C; rc=$?; log "XXL micro $M resume exit $rc"; break; }
done
if [ $rc = 2 ]; then LR2=$(python3 -c "print($LR/2)"); python3 $T --size XXL --lr $LR2 --train-tokens 7.7e9 --ckpt-every 10000 --micro $M $C --tag v2-half; rc=$?; log "XXL half-rate exit $rc"; fi
gcloud storage cp $B/eval/tasks-v1.json ~/tasks-v1.json >/dev/null 2>&1
for f in $OUT/dense-XXL-*.npz; do
  case $f in *.state.npz) continue;; esac
  python3 $D/eval_score.py --size XXL --ckpt $f --tasks ~/tasks-v1.json --batch 16 --out ${f%.npz}.bench.json > /dev/null 2>&1; log "benchmarks exit $? for $(basename $f)"
  python3 $D/eval_score.py --size XXL --ckpt $f --tasks ~/tasks-v1.json --batch 16 --control nocontext --out ${f%.npz}.bench-nocontext.json > /dev/null 2>&1; log "no-context control exit $?"
done
gcloud storage cp $OUT/*.bench*.json $B/runs/$RUN/ >/dev/null 2>&1
kill $PUSHER; gcloud storage cp $LOGF $B/runs/$RUN/job-scale2b-log.txt >/dev/null 2>&1
exit $rc
