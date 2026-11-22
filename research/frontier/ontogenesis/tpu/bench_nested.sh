#!/bin/bash
# Times the nested-protocol script on this machine under its current load, to forecast when the 35 CPU runs end.
# Per-step time = (time for 300 steps - time for 100 steps) / 200. Evaluation cost is measured separately.
# Uses seed 99 and writes no result file, so it cannot be confused with a registered run.
cd ~/onto; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
C="--lr 1e-3 --warmup 300 --churn 2 --hops 1,2 --symbols 512 --live 48 --device cpu"
b(){ s=$(date +%s.%N); ~/venv-torch/bin/python -B train.py --arm $1 --seed 99 --steps $2 --eval-every $3 $C > /dev/null 2>&1; e=$(date +%s.%N); echo "$1 $2 $3 $(python3 -c "print(round($e-$s,1))")"; }
for arm in tf-8x1-d64 tf-4x1-d40 looped dense; do b $arm 100 1000 & b $arm 300 1000 & done
b tf-8x1-d64 300 100 &
wait
