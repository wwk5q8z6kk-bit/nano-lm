#!/bin/bash
# Builds corpus v2 on a cloud machine's spare CPU cores and uploads it to the bucket.
# usage: job_corpus_v2.sh <gs://bucket> [total tokens] [workers]
# Holds ~/KEEPALIVE while it runs, so a controller that honours that marker does not delete the machine early.
B=$1; TOTAL=${2:-10e9}; W=${3:-48}; DIR=$(cd "$(dirname "$0")" && pwd)
touch ~/KEEPALIVE; LOG=~/corpus-v2.log
push(){ gcloud storage cp $LOG $B/runs/corpus-v2/log.txt >/dev/null 2>&1; }
( while [ -e ~/KEEPALIVE ]; do sleep 120; push; done ) &
echo "$(date -u +%H:%M:%S) start" >> $LOG
pip install -q pyarrow tokenizers huggingface_hub >> $LOG 2>&1
python3 $DIR/build_corpus_v2.py --v1 ~/data --out ~/corpus-v2 --cache ~/hf-cache --total $TOTAL --workers $W --gcs $B/corpus/v2 >> $LOG 2>&1
echo "$(date -u +%H:%M:%S) builder exit code $?" >> $LOG
df -h ~ | tail -1 >> $LOG; push
rm -f ~/KEEPALIVE
