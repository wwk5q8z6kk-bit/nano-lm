#!/bin/bash
# Runs on a cloud VM (CPU only).  usage: vm_nested.sh <gs://bucket> <run-name>
# Executes PROTOCOL-NESTED.md (five new arms) plus dense and looped re-runs, seeds 21-25, one CPU thread per run.
set -u
B=$1; RUN=$2
cd ~; mkdir -p nested-out; LOG=~/$RUN.log
log(){ echo "$(date -u +%H:%M:%S) $*" >> $LOG; gcloud storage cp $LOG $B/runs/$RUN/log.txt >/dev/null 2>&1; }
log "setup start; cpus $(nproc)"
pip install -q uv >> $LOG 2>&1; export PATH=$HOME/.local/bin:$PATH
uv venv ~/venv-torch --python 3.12 >> $LOG 2>&1
uv pip install --python ~/venv-torch/bin/python numpy==2.5.1 >> $LOG 2>&1
uv pip install --python ~/venv-torch/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu >> $LOG 2>&1
~/venv-torch/bin/python -c "import torch, numpy, sys; print('torch', torch.__version__, 'numpy', numpy.__version__, sys.version.split()[0])" >> $LOG 2>&1
log "setup done"
cd ~/onto; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
C="--steps 26000 --eval-every 1000 --lr 1e-3 --warmup 300 --churn 2 --hops 1,2 --symbols 512 --live 48 --device cpu"
for arm in tf-2x2-d64 tf-4x2-d64 tf-8x1-d64 tf-1x8-d64 tf-4x1-d40 dense looped; do
  for s in 21 22 23 24 25; do
    ( ~/venv-torch/bin/python -B train.py --arm $arm --seed $s $C --out ~/nested-out/$arm-s$s.json > /dev/null 2> ~/nested-out/$arm-s$s.err; gcloud storage cp ~/nested-out/$arm-s$s.json $B/runs/$RUN/ >/dev/null 2>&1 ) &
  done
done
wait
log "runs finished: $(ls ~/nested-out/*.json 2>/dev/null | wc -l) of 35"
echo done > ~/DONE-$RUN; gcloud storage cp ~/DONE-$RUN $B/runs/$RUN/DONE; log "all done"
