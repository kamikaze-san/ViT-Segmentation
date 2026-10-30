#!/bin/bash
# E31 stage 2 for the Qwen3-VL ViT 300M (from WeVisOne-VL-2B): passed its stage-1 gate at +0.011 (0.6640 vs soft ViT-B's
# 0.6529). Exactly C-RADIO run 2's recipe (cradio_stage2_posfix.sh) with only the backbone swapped: joint LoRA r16 a32
# on qkv/proj/linear_fc1/linear_fc2 of all 24 blocks, from the epoch-10 stage-1 head, train12k + round-1 pseudo labels,
# soft targets, no class weights, blackout, seed 0, 10 epochs. The Qwen ViT has no train-only behaviour (attention
# dropout hard-coded 0; train_mode_fix is a no-op), so no position fix is needed. Judge at epoch 10 and on Kaggle,
# not mid-run (C-RADIO dipped 0.69 -> 0.56 at epoch 1 and recovered to 0.717). Role: an ensemble candidate for
# diversity (E29: slightly more independent of DINOv3 on c3 than C-RADIO).
# Usage: qwen_stage2.sh [--grad-checkpoint]
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=../outputs/logs/qwen_stage2.log
TAG=_qwenvit_lora_warm_soft_pseudo01
HEAD=fullft_lr0.0001_g0.85_h1024_qwenvit_soft_fh10_ep10.pt
[ -f "../outputs/checkpoints/$HEAD" ] || { echo "REFUSING TO RUN: $HEAD missing"; exit 1; }
if ls ../outputs/checkpoints/fullft_lr0.0001_g0.85_h1024${TAG}*.pt >/dev/null 2>&1; then
  echo "REFUSING TO RUN: a checkpoint tagged $TAG already exists"; exit 1
fi
if [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 1500 ]; then
  echo "REFUSING TO RUN: the GPU is busy"; exit 3
fi
: > "$L"
echo "=== qwen stage 2 START $(date '+%m-%d %H:%M:%S') | extra flags: $* ===" >> "$L"
python -u train_full_ft.py --backbone qwenvit \
  --backbone-path "C:/Users/NewGr/Downloads/ViT Project severstal/severstal/external/WeVisOne-VL-2B" \
  --lora-r 16 --lora-alpha 32 "$@" --head-arch conv --hidden 1024 --init-head $HEAD \
  --bb-lr 1e-4 --gamma 0.85 --bs 4 --seed 0 --epochs 10 --save-epochs 10 \
  --augment --aug-strength blackout --train-split train12k --val-split holdout \
  --pseudo-ids ../data/processed/pseudo_ids.csv --weight-power 0 --target soft --tag $TAG >> "$L" 2>&1
echo "=== exit $? $(date '+%m-%d %H:%M:%S') ===" >> "$L"
