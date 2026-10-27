#!/bin/bash
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
if [ -n "$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -i python)" ]; then echo "REFUSING TO RUN: python process on GPU"; exit 3; fi
python -u slide_assets.py > ../outputs/logs/slide_assets.log 2>&1
echo "exit $? $(date '+%H:%M:%S')" >> ../outputs/logs/slide_assets.log
