#!/bin/bash
# Slide overlays (soft ViT-B on PCA-figure strips + holdout) then the H-B attention diagnostic (E27).
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
LD=../outputs/logs
if [ -n "$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -i python)" ]; then
  echo "REFUSING TO RUN: a python process is using the GPU"; exit 3
fi
python -u slide_overlays.py > $LD/slide_overlays.log 2>&1
echo "overlays exit $? $(date '+%H:%M:%S')" >> $LD/slide_overlays.log
python -u cpt_attn_diag.py > $LD/cpt_attn_diag.log 2>&1
echo "attn exit $? $(date '+%H:%M:%S')" >> $LD/cpt_attn_diag.log
