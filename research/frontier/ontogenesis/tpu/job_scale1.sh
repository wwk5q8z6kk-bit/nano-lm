#!/bin/bash
# Question 12 (INVESTIGATION.md): dense models on corpus v2, one run across the four chips of the v6e machine.
# Placed in the machine's queue (job_queue.sh). B (bucket) comes from the queue.
set -u
RUN=scale1-v6e; D=$(cd "$(dirname "$0")" && pwd); T="$D/train_jax.py"
log(){ echo "$(date -u +%H:%M:%S) $*"; }
mkdir -p ~/data2 ~/tp ~/out6
[ -s ~/data2/train.bin ] || gcloud storage cp "$B/corpus/v2/*" ~/data2/
[ "$(stat -c %s ~/data2/train.bin)" = 20000000000 ] || { log "corpus v2 incomplete; stopping"; exit 1; }
# 1. Throughput of one 211M run on 1, 2 and 4 chips (16 sequences per chip per micro-batch in every case), 300 steps each.
for cfg in "1 64 4" "2 64 2" "4 64 1" "4 256 4"; do set -- $cfg
  python3 $T --size XL --data ~/data2 --out ~/tp --dp $1 --bs $2 --micro $3 --train-tokens $((300*$2*512)) --eval-every 100000 --tag tp > ~/tp/xl-dp$1-bs$2.log 2>&1
  log "XL dp $1 bs $2 micro $3: $(grep -o '"tokens_per_second": [0-9]*' ~/tp/xl-dp$1-bs$2.log | tail -1) $(grep -o '"device_gb_in_use": [0-9.]*' ~/tp/xl-dp$1-bs$2.log | tail -1)"
done
gcloud storage cp ~/tp/*.json ~/tp/*.log $B/runs/$RUN/throughput/ >/dev/null 2>&1
# 2. The runs, most valuable first. Default recipe (AdamW 1e-3, decay 0.1 on all, floor 10%), batch 64 x 512, seed 1000.
C="--data $HOME/data2 --out $HOME/out6 --gcs $B/runs/$RUN --save --resume --dp 4 --tag v2"
python3 $T --size XL --train-tokens 4.2e9 --ckpt-every 20000 $C; log "XL 4.2B exit $?"
python3 $T --size L --train-tokens 2.0e9 --ckpt-every 20000 $C; log "L 2.0B exit $?"
python3 $T --size L --train-tokens 0.8e9 $C; log "L 0.8B exit $?"
# 3. Benchmarks on each finished model, with the no-context control.
gcloud storage cp $B/eval/tasks-v1.json ~/tasks-v1.json >/dev/null 2>&1
for f in ~/out6/dense-*.npz; do
  S=$(basename $f | cut -d- -f2)
  python3 $D/eval_score.py --size $S --ckpt $f --tasks ~/tasks-v1.json --batch 128 --out ${f%.npz}.bench.json > /dev/null 2>&1
  python3 $D/eval_score.py --size $S --ckpt $f --tasks ~/tasks-v1.json --batch 128 --control nocontext --out ${f%.npz}.bench-nocontext.json > /dev/null 2>&1
  log "benchmarks done for $(basename $f)"
done
gcloud storage cp ~/out6/*.bench*.json $B/runs/$RUN/ >/dev/null 2>&1
