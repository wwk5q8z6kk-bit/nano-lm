#!/bin/bash
# PROTOCOL-NESTED-LR.md on a TPU host's CPU. usage: vm_nested_lr.sh <gs://bucket>
# 40 runs: arms looped, dense, tf-4x2-d64, tf-8x1-d64 at lr 3e-4 and 3e-3, seeds 21-25, one CPU thread each, all in parallel.
# Skips any run whose result is already in the bucket (so it can be restarted on a replacement node).
set -u
B=$1; RUN=nested-lr; O=~/nested-lr-out; mkdir -p $O; LOG=~/$RUN.log
log(){ echo "$(date -u +%H:%M:%S) $*" >> $LOG; gcloud storage cp $LOG $B/runs/$RUN/log.txt >/dev/null 2>&1; }
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh) >> $LOG 2>&1
[ -x ~/venv-torch/bin/python ] || { uv venv ~/venv-torch --python 3.12 >> $LOG 2>&1; uv pip install --python ~/venv-torch/bin/python numpy==2.5.1 >> $LOG 2>&1; uv pip install --python ~/venv-torch/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu >> $LOG 2>&1; }
~/venv-torch/bin/python -c "import torch, numpy, sys; print('torch', torch.__version__, 'numpy', numpy.__version__, sys.version.split()[0])" >> $LOG 2>&1
done_list=$(gcloud storage ls $B/runs/$RUN/ 2>/dev/null)
cd "$(dirname "$0")/.."; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
C="--steps 26000 --eval-every 1000 --warmup 300 --churn 2 --hops 1,2 --symbols 512 --live 48 --device cpu"
n=0
for lr in 3e-4 3e-3; do for arm in looped dense tf-4x2-d64 tf-8x1-d64; do for s in 21 22 23 24 25; do
  f=$arm-lr$lr-s$s.json
  echo "$done_list" | grep -q "/$f$" && continue
  n=$((n + 1))
  ( nice -n 10 ~/venv-torch/bin/python -B train.py --arm $arm --seed $s --lr $lr $C --out $O/$f > /dev/null 2> $O/${f%.json}.err; gcloud storage cp $O/$f $B/runs/$RUN/ >/dev/null 2>&1 ) &
done; done; done
log "started $n runs"
wait
log "runs finished: $(ls $O/*.json 2>/dev/null | wc -l) local result files"
