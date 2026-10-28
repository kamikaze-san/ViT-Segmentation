#!/bin/bash
# E31 stage 2: C-RADIOv4-SO400M, passed the stage-1 gate at +0.038 (holdout 0.6907 vs soft ViT-B 0.6529).
# Exactly stage 2 of vitb_soft_pseudo.sh with only the backbone swapped: joint LoRA r16 a32 on every block's
# qkv/proj/fc1/fc2, from the stage-1 epoch-10 head (trained on the Mac, reproduced 0.6906 here), train12k + round-1
# pseudo labels, soft targets, no class weights, blackout, seed 0, 10 epochs. Then a rate-matched solo Kaggle
# submission against the soft ViT-B's 0.90695. A seed-1 repeat of the gate runs on the Mac in parallel; if it falls
# back near 0.65 the gain was seed luck and this run is stopped.
# Usage: cradio_stage2.sh [--grad-checkpoint]
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=../outputs/logs/cradio_stage2.log
TAG=_cradio_so_lora_warm_soft_pseudo01
HEAD=fullft_lr0.0001_g0.85_h1024_cradio_so_fh10_ep10.pt
[ -f "../outputs/checkpoints/$HEAD" ] || { echo "REFUSING TO RUN: $HEAD missing"; exit 1; }
if ls ../outputs/checkpoints/fullft_lr0.0001_g0.85_h1024${TAG}*.pt >/dev/null 2>&1; then
  echo "REFUSING TO RUN: a checkpoint tagged $TAG already exists"; exit 1
fi
if [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 1500 ]; then
  echo "REFUSING TO RUN: the GPU is busy"; exit 3
fi
: > "$L"
echo "=== cradio stage 2 START $(date '+%m-%d %H:%M:%S') | extra flags: $* ===" >> "$L"
python -u train_full_ft.py --backbone cradio \
  --backbone-path "C:/Users/NewGr/Downloads/ViT Project severstal/severstal/external/C-RADIOv4-SO400M" \
  --lora-r 16 --lora-alpha 32 "$@" --head-arch conv --hidden 1024 --init-head $HEAD \
  --bb-lr 1e-4 --gamma 0.85 --bs 4 --seed 0 --epochs 10 --save-epochs 10 \
  --augment --aug-strength blackout --train-split train12k --val-split holdout \
  --pseudo-ids ../data/processed/pseudo_ids.csv --weight-power 0 --target soft --tag $TAG >> "$L" 2>&1
echo "=== exit $? $(date '+%m-%d %H:%M:%S') ===" >> "$L"
