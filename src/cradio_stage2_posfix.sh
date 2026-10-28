#!/bin/bash
# E31 stage 2, rerun with the C-RADIO position fix: identical to cradio_stage2.sh except that train_full_ft.py now
# keeps C-RADIO's CPE patch generator in eval mode while the rest trains (backbones.RadioBackbone.train_mode_fix).
# The first run (cradio_stage2.sh, tag _cradio_so_lora_warm_soft_pseudo01) trained LoRA on randomly cropped/stretched
# position embeddings that eval never uses and fell from 0.6906 (epoch 0) to 0.5954 / 0.6357 / 0.6126 at epochs 1-3,
# while soft ViT-B's stage 2 fell only 0.009 at epoch 1. If the fix is the cause, epoch 1 here should stay near 0.69.
# Usage: cradio_stage2_posfix.sh [--grad-checkpoint]
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=../outputs/logs/cradio_stage2_posfix.log
TAG=_cradio_so_lora_warm_soft_posfix_pseudo01
HEAD=fullft_lr0.0001_g0.85_h1024_cradio_so_fh10_ep10.pt
[ -f "../outputs/checkpoints/$HEAD" ] || { echo "REFUSING TO RUN: $HEAD missing"; exit 1; }
grep -q "ADAPTER.train_mode_fix()" train_full_ft.py || { echo "REFUSING TO RUN: train_full_ft.py lacks the fix"; exit 1; }
if ls ../outputs/checkpoints/fullft_lr0.0001_g0.85_h1024${TAG}*.pt >/dev/null 2>&1; then
  echo "REFUSING TO RUN: a checkpoint tagged $TAG already exists"; exit 1
fi
if [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 1500 ]; then
  echo "REFUSING TO RUN: the GPU is busy"; exit 3
fi
: > "$L"
echo "=== cradio stage 2 posfix START $(date '+%m-%d %H:%M:%S') | extra flags: $* ===" >> "$L"
python -u train_full_ft.py --backbone cradio \
  --backbone-path "C:/Users/NewGr/Downloads/ViT Project severstal/severstal/external/C-RADIOv4-SO400M" \
  --lora-r 16 --lora-alpha 32 "$@" --head-arch conv --hidden 1024 --init-head $HEAD \
  --bb-lr 1e-4 --gamma 0.85 --bs 4 --seed 0 --epochs 10 --save-epochs 10 \
  --augment --aug-strength blackout --train-split train12k --val-split holdout \
  --pseudo-ids ../data/processed/pseudo_ids.csv --weight-power 0 --target soft --tag $TAG >> "$L" 2>&1
echo "=== exit $? $(date '+%m-%d %H:%M:%S') ===" >> "$L"
