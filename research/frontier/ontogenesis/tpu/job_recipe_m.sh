#!/bin/bash
# Job for vm_chain.sh on the second TPU machine (v6e). Registered in INVESTIGATION.md, Question 11, 40M extension.
set -u
D=$(cd "$(dirname "$0")" && pwd)
python3 $D/train_jax.py --data $HOME/data --out $HOME/out5 --gcs $B/runs/$RUN --save --resume --multi "$(python3 $D/recipe_select.py gridm)"
