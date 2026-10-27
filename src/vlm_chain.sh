#!/bin/bash
# Zero-shot VLM queue: wait for the running arm-A level-0 process, then level 1 -> arm B (WeVisOne, isolated
# venv; Ollama model unloaded first to free VRAM) -> arm C (LLMSeg-style loop). Each stage runs even if the
# previous one failed, so one broken arm does not block the others.
cd "$(dirname "$0")"
L=../outputs/logs/vlm_armA.log
L0_PID=${1:?usage: vlm_chain.sh <level-0 windows pid>}
alive () { powershell -NoProfile -Command "if (Get-Process -Id $L0_PID -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; }
while alive; do sleep 20; done
echo "=== chain: level 0 process $L0_PID ended $(date '+%m-%d %H:%M:%S') ===" >> "$L"
python -u vlm_zeroshot.py --level 0 --n 200 >> "$L" 2>&1      # resumes; no-op if level 0 already finished
python -u vlm_zeroshot.py --level 1 --n 200 >> "$L" 2>&1
ollama stop qwen3-vl:4b-instruct >> "$L" 2>&1
echo "=== chain: arm B start $(date '+%m-%d %H:%M:%S') ===" >> "$L"
HF_HUB_OFFLINE=1 ../external/wevisone_env/Scripts/python.exe -u vlm_wevisone.py >> "$L" 2>&1
echo "=== chain: arm B exit $? $(date '+%m-%d %H:%M:%S') ===" >> "$L"
python -u vlm_llmseg_lite.py >> "$L" 2>&1
echo "=== chain: ALL DONE $(date '+%m-%d %H:%M:%S') ===" >> "$L"
