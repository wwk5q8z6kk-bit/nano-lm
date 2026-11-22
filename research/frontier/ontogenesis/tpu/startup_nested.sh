#!/bin/bash
# Startup script for a plain CPU VM: fetch the code, run the nested protocol, push results, delete this VM.
export HOME=/root; cd /root
B=gs://nanolm-lab-eu-0478806098; RUN=nested-cpu-1
for i in 1 2 3 4 5; do gcloud storage cp $B/code/onto.tgz . && break; sleep 10; done
mkdir -p onto && tar xzf onto.tgz -C onto
bash onto/tpu/vm_nested.sh $B $RUN
ZONE=$(curl -s -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
gcloud compute instances delete "$(hostname)" --zone "$ZONE" --quiet
