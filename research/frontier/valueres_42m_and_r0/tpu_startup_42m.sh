#!/bin/bash
# Self-bootstrapping, self-deleting Cloud TPU v5litepod-4 startup script for the pre-registered
# 40.38M (`M`-scale, 800M tokens x 4 arms x 3 seeds = 9.6B tokens) ValueRes Confirmation Gate.
#
# Safety layers:
#   1. Immediately installs a persistent systemd timer (`tpu-deadline.timer`) + `/usr/local/bin/tpu-deadline-delete`
#      BEFORE installing packages or downloading data, and verifies `tpu.nodes.delete` IAM permission; if installation
#      or IAM check fails, immediately deletes the TPU node via the TPU REST API.
#   2. Streams `startup.log` to GCS every 120s and uploads each completed `run_*.json` receipt as soon as it finishes.
#   3. Immediately calls `/usr/local/bin/tpu-deadline-delete` on normal or error exit (`trap cleanup EXIT`).
set -u

meta() {
  curl -s -f --max-time 10 -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

PROJECT_ID=$(meta PROJECT_ID)
ZONE_NAME=$(meta ZONE_NAME)
NODE_NAME=$(meta NODE_NAME)
DEADLINE_UTC=$(meta DEADLINE_UTC)
GCS_BUCKET=$(meta GCS_BUCKET)
RUN_ID=$(meta RUN_ID)
SEEDS=$(meta SEEDS || echo "1000,1001,1002")

WORK=/opt/nanolm_42m
mkdir -p "$WORK/data" "$WORK/out"
LOG="$WORK/startup.log"
GCS_RUN="$GCS_BUCKET/runs/$RUN_ID"

push_log() {
  timeout 45 gcloud storage cp "$LOG" "$GCS_RUN/startup.log" --quiet >/dev/null 2>&1 || true
}
log() {
  echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"
  push_log
}

# 1. Install self-deletion script first
cat << 'DEL_EOF' > /usr/local/bin/tpu-deadline-delete
#!/bin/bash
P=$1; Z=$2; N=$3
URL="https://tpu.googleapis.com/v2/projects/$P/locations/$Z/nodes/$N"
LOGF=/var/log/tpu-deadline.log
tok(){ curl -s --max-time 10 -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])' 2>/dev/null; }
for i in $(seq 1 240); do
  T=$(tok)
  D=$(curl -s --max-time 30 -o /dev/null -w '%{http_code}' -X DELETE -H "Authorization: Bearer $T" "$URL")
  S=$(curl -s --max-time 30 -H "Authorization: Bearer $T" "$URL" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("state") or d.get("error",{}).get("code"))' 2>/dev/null)
  echo "$(date -u +%FT%TZ) attempt $i: DELETE HTTP $D, state $S" >> "$LOGF"
  case "$S" in DELETING|404) exit 0;; esac
  sleep 20
done
exit 1
DEL_EOF
chmod 755 /usr/local/bin/tpu-deadline-delete

cleanup() {
  rc=$?
  log "EXIT trap fired (rc=$rc); uploading final logs and self-deleting TPU node $NODE_NAME in $ZONE_NAME"
  timeout 60 gcloud storage cp "$WORK/out/"*.json "$GCS_RUN/" --quiet >/dev/null 2>&1 || true
  push_log
  /usr/local/bin/tpu-deadline-delete "$PROJECT_ID" "$ZONE_NAME" "$NODE_NAME"
}
trap cleanup EXIT

NOW_S=$(date -u +%s)
AT_S=$(date -u -d "$DEADLINE_UTC" +%s)
if [ "$AT_S" -le "$((NOW_S + 600))" ]; then
  log "Refusing startup: deadline $DEADLINE_UTC UTC is <10 min away"
  exit 1
fi

cat << EOF > /etc/systemd/system/tpu-deadline.service
[Unit]
Description=Delete TPU node $NODE_NAME at deadline
[Service]
Type=oneshot
ExecStart=/usr/local/bin/tpu-deadline-delete $PROJECT_ID $ZONE_NAME $NODE_NAME
EOF

cat << EOF > /etc/systemd/system/tpu-deadline.timer
[Unit]
Description=Hard deadline for TPU node $NODE_NAME
[Timer]
OnCalendar=$DEADLINE_UTC UTC
Persistent=true
AccuracySec=1s
[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now tpu-deadline.timer >> "$LOG" 2>&1

TOK=$(curl -s --max-time 10 -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')
PERMS=$(curl -s --max-time 30 -X POST -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" -d '{"permissions":["tpu.nodes.delete","tpu.nodes.get"]}' "https://cloudresourcemanager.googleapis.com/v1/projects/$PROJECT_ID:testIamPermissions" | tr -d '\n ')
log "Booted $NODE_NAME ($ZONE_NAME); deadline=$DEADLINE_UTC UTC; IAM=$PERMS"
echo "$PERMS" | grep -q "tpu.nodes.delete" || { log "Missing tpu.nodes.delete permission; aborting"; exit 1; }

( while true; do sleep 120; push_log; done ) &

# 2. Install JAX + Optax for TPU v5e
log "Installing jax[tpu]==0.6.2 and optax==0.2.9"
pip install -q -U "jax[tpu]==0.6.2" "optax==0.2.9" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html >> "$LOG" 2>&1
python3 -c "import jax, optax; print('DEVICES:', jax.devices(), 'JAX:', jax.__version__, 'OPTAX:', optax.__version__)" >> "$LOG" 2>&1

# 3. Stage trainer script, Corpus v2 shards, and 7-task eval file from GCS
log "Downloading trainer code, Corpus v2, and tasks-v1.json from $GCS_BUCKET"
gcloud storage cp "$GCS_RUN/code/valueres_42m_jax.py" "$WORK/valueres_42m_jax.py" --quiet >> "$LOG" 2>&1
gcloud storage cp "$GCS_BUCKET/eval/tasks-v1.json" "$WORK/tasks-v1.json" --quiet >> "$LOG" 2>&1
gcloud storage cp "$GCS_BUCKET/corpus/v2/*" "$WORK/data/" --quiet >> "$LOG" 2>&1
ls -lh "$WORK/data/" >> "$LOG" 2>&1
# Download any previously completed run_*.json receipts in case of spot preemption resume
gcloud storage cp "$GCS_RUN/run_*.json" "$WORK/out/" --quiet >> "$LOG" 2>&1 || true

# 4. Run 4-arm x 3-seed 40.38M tournament across all 4 TPU v5e chips
log "Starting valueres_42m_jax.py for seeds=$SEEDS"
python3 -u "$WORK/valueres_42m_jax.py" \
  --data-dir "$WORK/data" \
  --val-dir "$WORK/data" \
  --tasks "$WORK/tasks-v1.json" \
  --out-dir "$WORK/out" \
  --gcs "$GCS_RUN" \
  --size M \
  --seeds "$SEEDS" \
  --train-tokens 800000000 \
  --batch-size 64 \
  --seq 512 \
  --lr 0.004 \
  --wd 0.1 \
  --lr-floor 0.01 \
  --warmup 500 \
  --eval-every 4000 >> "$LOG" 2>&1
RC=$?
log "valueres_42m_jax.py finished with rc=$RC"
exit $RC
