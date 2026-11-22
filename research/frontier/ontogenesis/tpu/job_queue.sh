#!/bin/bash
# Job for vm_chain.sh: run every script in ~/queue/ (name order), oldest name first, until none is left.
# Scripts may be added while it runs; each finished one is moved to ~/queue/done/ with its log.
# Each queued script sets its own RUN (bucket folder); B (bucket) is passed through.
set -u
mkdir -p ~/queue/done
while true; do
  N=$(ls ~/queue/*.sh 2>/dev/null | sort | head -1)
  [ -z "$N" ] && break
  echo "$(date -u +%H:%M:%S) queue: starting $(basename $N)"
  B=$B bash "$N" > ~/queue/done/$(basename $N .sh).log 2>&1
  echo "$(date -u +%H:%M:%S) queue: $(basename $N) exited with $?"
  mv "$N" ~/queue/done/
done
echo "$(date -u +%H:%M:%S) queue empty"
