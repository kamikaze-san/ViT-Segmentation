#!/bin/bash
# Arm D (LocateAnything-3B): wait for the qwen3.5 run to release the GPU and for the weight download to finish,
# then run in the isolated venv; restart on exit 3 (CUDA OOM poisons the context).
cd "$(dirname "$0")"
L=../outputs/logs/vlm_armA.log
WAIT_PID=${1:?usage: vlm_locany.sh <q35 windows pid>}
alive () { powershell -NoProfile -Command "if (Get-Process -Id $WAIT_PID -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; }
while alive; do sleep 30; done
until grep -q "^path=" ../outputs/logs/locany_download.log; do sleep 30; done
echo "=== armD start $(date '+%m-%d %H:%M:%S') ===" >> "$L"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
for i in $(seq 1 20); do
  ../external/locany_env/Scripts/python.exe -u vlm_locany.py >> "$L" 2>&1
  rc=$?; [ $rc -eq 3 ] || break; sleep 5
done
echo "=== armD end rc $rc after $i runs $(date '+%m-%d %H:%M:%S') ===" >> "$L"
