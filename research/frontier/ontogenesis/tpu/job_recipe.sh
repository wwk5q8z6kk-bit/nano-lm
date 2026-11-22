#!/bin/bash
# Job for vm_chain.sh on the first TPU machine, after the distillation batch. Registered in INVESTIGATION.md, Question 11.
# Phase A: the 14M recipe grid. Phase B: chosen mechanically from phase A by recipe_select.py (no human in the loop).
set -u
D=$(cd "$(dirname "$0")" && pwd); T="$D/train_jax.py"
C="--data $HOME/data --out $HOME/out4 --gcs $B/runs/$RUN --save --resume"
python3 $T $C --multi "$(python3 $D/recipe_select.py grid)"
python3 $D/recipe_select.py pick $HOME/out4 $HOME/out3 > ~/recipe-pick.json
gcloud storage cp ~/recipe-pick.json $B/runs/$RUN/recipe-pick.json >/dev/null 2>&1
B2="$(python3 $D/recipe_select.py phaseb ~/recipe-pick.json)"; echo "phase B: $B2"
[ -n "$B2" ] && python3 $T $C --multi "$B2"
rc=$?
while [ -e ~/KEEPALIVE ]; do sleep 30; done
exit $rc
