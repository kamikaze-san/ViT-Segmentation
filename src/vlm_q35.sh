#!/bin/bash
# qwen3.5:4b zero-shot on the first 60 holdout strips (paired with arm A), thinking off, greedy;
# waits for the arm-B retry loop to release the GPU.
cd "$(dirname "$0")"
L=../outputs/logs/vlm_armA.log
WAIT_PID=${1:?usage: vlm_q35.sh <armB-retry windows pid>}
alive () { powershell -NoProfile -Command "if (Get-Process -Id $WAIT_PID -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; }
while alive; do sleep 30; done
echo "=== q35 start $(date '+%m-%d %H:%M:%S') ===" >> "$L"
python -u vlm_zeroshot.py --model qwen3.5:4b --tag q35 --think off --level 0 --n 60 >> "$L" 2>&1
python -u vlm_zeroshot.py --model qwen3.5:4b --tag q35 --think off --level 1 --n 60 >> "$L" 2>&1
ollama stop qwen3.5:4b >> "$L" 2>&1
echo "=== q35 end $(date '+%m-%d %H:%M:%S') ===" >> "$L"
