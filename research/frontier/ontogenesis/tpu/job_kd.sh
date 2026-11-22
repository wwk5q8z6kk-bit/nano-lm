#!/bin/bash
# Job for vm_chain.sh on the first TPU machine (4 chips). Registered in INVESTIGATION.md, Question 10,
# with the pre-run amendment of 2026-11-22: every arm uses 4 micro-batches; three 14M baseline seeds.
# Second version of the training script (uniform host-side sampling) on the 800M-token corpus.
# Teacher: the finished 101M model at learning rate 1e-3, seed 1000 (chosen in advance, not by its score).
# Longest jobs first, so the four chips finish close together.
python3 ~/onto3/tpu/train_jax.py --data ~/data --out ~/out3 --gcs $B/runs/$RUN --save --resume \
  --teacher ~/out/dense-L-lr0.001-s1000.npz --teacher-size L --kd-alpha 0.5 \
  --multi "M:1e-3:4:::1000:0.5;S:1e-3:4:::1000:0.5;S:1e-3:4:::1000:0.0;M:1e-3:4:::1000:none;S:1e-3:4:::1000:none;S:1e-3:4:::1001:none;S:1e-3:4:::1002:none"
rc=$?
# Stay alive while another task on this machine (for example a corpus build) holds this marker.
while [ -e ~/KEEPALIVE ]; do sleep 30; done
exit $rc
