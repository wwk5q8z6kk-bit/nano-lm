#!/bin/bash
# Cost watchdog. Every 2 min:
#  - remove any pod whose driver log says its queue is complete
#  - if estimated campaign spend (from pod list costPerHr x uptime since START) exceeds CAP, remove ALL nanolm pods
# usage: watchdog.sh <cap_usd>
CAP=${1:-45}
LOGS=/Volumes/Express1M2/nanolm-lab/logs
LEDGER=$LOGS/watchdog.log
SPENT_BASE=1.5   # pre-watchdog spend: H100 pilot (~$0.70) + 4090/A40 time before 05:08Z (~$0.8)
START=$(date +%s)
while true; do
  now=$(date +%s)
  for f in $LOGS/grid_*.log $LOGS/trackb_eval_*.log; do
    [ -f "$f" ] || continue
    if grep -q "QUEUE COMPLETE" "$f"; then
      pod=$(grep -o "QUEUE COMPLETE [a-z0-9]*" "$f" | awk '{print $3}')
      [ -z "$pod" ] && pod=$(basename $f .log | sed 's/grid_//')
      if [ -n "$pod" ] && runpodctl get pod 2>/dev/null | grep -q "^$pod"; then
        runpodctl remove pod $pod >> $LEDGER 2>&1 && echo "$(date -u +%FT%TZ) removed $pod (queue complete)" >> $LEDGER
      fi
    fi
  done
  # spend estimate: running nanolm pods x rate x elapsed since watchdog start (conservative upper bound)
  n=$(runpodctl get pod 2>/dev/null | grep -c nanolm)
  rate=$(python3 -c "print(round($n*0.89,2))")   # upper bound: every pod billed at the 4090 rate
  [ -z "$rate" ] && rate=5.04
  ACC=$(python3 -c "import os;p='$LOGS/.wd_acc';a=float(open(p).read()) if os.path.exists(p) else $SPENT_BASE;a+=$rate*120/3600;open(p,'w').write(str(a));print(round(a,2))")
  echo "$(date -u +%FT%TZ) rate=\$$rate/h est_spent=\$$ACC cap=\$$CAP" >> $LEDGER
  if python3 -c "import sys;sys.exit(0 if $ACC>$CAP else 1)"; then
    echo "$(date -u +%FT%TZ) CAP EXCEEDED - removing all nanolm pods" >> $LEDGER
    for p in $(runpodctl get pod 2>/dev/null | awk '/nanolm/{print $1}'); do runpodctl remove pod $p >> $LEDGER 2>&1; done
    exit 0
  fi
  sleep 120
done
