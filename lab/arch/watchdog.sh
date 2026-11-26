#!/bin/bash
# Cost & lifecycle watchdog (Phase 2: supports NextGen continuation pods & Track B B1/B2 pod).
CAP=${1:-45}
LOGS="${NANOLM_LOGS_DIR:-artifacts/nanolm_lab/logs}"
LEDGER=$LOGS/watchdog.log

while true; do
  # 1. Remove e71ekqgldujy96 and pahl3151c37dlu as soon as their Wave-1 grid finishes
  for f in $LOGS/none_wave1.log; do
    [ -f "$f" ] || continue
    if grep -q "QUEUE COMPLETE" "$f"; then
      pod=$(grep -o "QUEUE COMPLETE [a-z0-9]*" "$f" | awk '{print $3}')
      if [ -n "$pod" ] && runpodctl get pod 2>/dev/null | grep -q "^$pod"; then
        runpodctl remove pod $pod >> $LEDGER 2>&1 && echo "$(date -u +%FT%TZ) removed $pod (wave1 complete)" >> $LEDGER
      fi
    fi
  done

  # 2. Remove NextGen pods when NEXTGEN_COMPLETE
  for f in $LOGS/nextgen_*.log; do
    [ -f "$f" ] || continue
    if grep -q "NEXTGEN_COMPLETE" "$f"; then
      pod=$(grep -o "NEXTGEN_COMPLETE [a-z0-9]*" "$f" | awk '{print $2}')
      if [ -n "$pod" ] && runpodctl get pod 2>/dev/null | grep -q "^$pod"; then
        runpodctl remove pod $pod >> $LEDGER 2>&1 && echo "$(date -u +%FT%TZ) removed $pod (nextgen complete)" >> $LEDGER
      fi
    fi
  done

  # 3. Remove Track B A40 pod once local sync confirms B1_B2_SYNCED
  if [ -f "$LOGS/trackb_b1_b2_synced.flag" ]; then
    if runpodctl get pod 2>/dev/null | grep -q "^2wjeaobs3gnszi"; then
      runpodctl remove pod 2wjeaobs3gnszi >> $LEDGER 2>&1 && echo "$(date -u +%FT%TZ) removed 2wjeaobs3gnszi (trackb complete)" >> $LEDGER
    fi
  fi

  n=$(runpodctl get pod 2>/dev/null | grep -c nanolm)
  rate=$(python3 -c "print(round($n*0.89,2))")
  ACC=$(python3 -c "import os;p='$LOGS/.wd_acc';a=float(open(p).read()) if os.path.exists(p) else 19.2;a+=$rate*120/3600;open(p,'w').write(str(a));print(round(a,2))")
  echo "$(date -u +%FT%TZ) pods=$n rate=\$$rate/h est_spent=\$$ACC cap=\$$CAP" >> $LEDGER
  if python3 -c "import sys;sys.exit(0 if $ACC>$CAP else 1)"; then
    echo "$(date -u +%FT%TZ) CAP EXCEEDED - removing all nanolm pods" >> $LEDGER
    for p in $(runpodctl get pod 2>/dev/null | awk '/nanolm/{print $1}'); do runpodctl remove pod $p >> $LEDGER 2>&1; done
    exit 0
  fi
  sleep 120
done
