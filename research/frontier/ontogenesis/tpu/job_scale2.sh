#!/bin/bash
# Question 13 (INVESTIGATION.md): the next-scale dense model on corpus v2, new recipe, four v5e chips.
set -u
RUN=scale2-v5e; D=$(cd "$(dirname "$0")" && pwd); T="$D/train_jax.py"; OUT=$HOME/out7
log(){ echo "$(date -u +%H:%M:%S) $*"; }
mkdir -p ~/data2 $OUT
[ -s ~/data2/train.bin ] || gcloud storage cp "$B/corpus/v2/*" ~/data2/
[ "$(stat -c %s ~/data2/train.bin)" = 20000000000 ] || { log "corpus v2 incomplete"; exit 1; }
C="--data $HOME/data2 --out $OUT --gcs $B/runs/$RUN --save --resume --dp 4 --opt muon --wd 0.1 --lr-floor 0.01 --tag v2"
# 1. Learning-rate check at 101M, 0.8B tokens.
for LR in 2e-3 4e-3 8e-3; do python3 $T --size L --lr $LR --train-tokens 0.8e9 $C; log "L lr $LR exit $?"; done
# 2. Mechanical choice: best of the three by final mix (diverged runs excluded), capped at 4e-3.
LR=$(python3 - <<PY
import json, glob
best = None
for f in glob.glob("$OUT/dense-L-*-muon-fl0.01-dp4-hs-v2.json"):
    d = json.load(open(f))
    if d.get("finished") and d.get("curve"):
        m = d["curve"][-1]["mix"]
        if best is None or m < best[0]: best = (m, d["lr"])
print(min(best[1], 4e-3) if best else 2e-3)
PY
); log "chosen rate for 386M: $LR"; echo "$LR" > $OUT/chosen-lr.txt; gcloud storage cp $OUT/chosen-lr.txt $B/runs/$RUN/ >/dev/null 2>&1
# 3. The 386M model on 7.7B tokens (about 20 per parameter); half the rate once if it diverges.
python3 $T --size XXL --lr $LR --train-tokens 7.7e9 --ckpt-every 10000 $C; rc=$?; log "XXL exit $rc"
if [ $rc = 2 ]; then LR2=$(python3 -c "print($LR/2)"); python3 $T --size XXL --lr $LR2 --train-tokens 7.7e9 --ckpt-every 10000 $C --tag v2-half; log "XXL half-rate exit $?"; fi
# 4. Benchmarks with the no-context control.
gcloud storage cp $B/eval/tasks-v1.json ~/tasks-v1.json >/dev/null 2>&1
for f in $OUT/dense-XXL-*.npz; do
  python3 $D/eval_score.py --size XXL --ckpt $f --tasks ~/tasks-v1.json --batch 64 --out ${f%.npz}.bench.json > /dev/null 2>&1
  python3 $D/eval_score.py --size XXL --ckpt $f --tasks ~/tasks-v1.json --batch 64 --control nocontext --out ${f%.npz}.bench-nocontext.json > /dev/null 2>&1
  log "benchmarks done for $(basename $f)"
done
gcloud storage cp $OUT/*.bench*.json $B/runs/$RUN/ >/dev/null 2>&1
