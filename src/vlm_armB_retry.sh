#!/bin/bash
# Re-run arm B (WeVisOne) after the VLM queue finishes: restart the process whenever it exits 3 on a CUDA OOM
# (an OOM poisons the CUDA context), resuming from the strips already scored. At most 40 restarts.
cd "$(dirname "$0")"
L=../outputs/logs/vlm_armA.log
CHAIN_PID=${1:?usage: vlm_armB_retry.sh <vlm_chain windows pid>}
alive () { powershell -NoProfile -Command "if (Get-Process -Id $CHAIN_PID -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; }
while alive; do sleep 30; done
ollama stop qwen3-vl:4b-instruct >> "$L" 2>&1
echo "=== armB retry start $(date '+%m-%d %H:%M:%S') ===" >> "$L"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
for i in $(seq 1 40); do
  ../external/wevisone_env/Scripts/python.exe -u vlm_wevisone.py >> "$L" 2>&1
  rc=$?
  [ $rc -eq 3 ] || break
  sleep 5
done
echo "=== armB retry end rc $rc after $i runs $(date '+%m-%d %H:%M:%S') ===" >> "$L"
