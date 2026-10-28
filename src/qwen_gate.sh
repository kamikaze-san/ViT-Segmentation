#!/bin/bash
# Backbone gate, Qwen3-VL ViT 300M (vision tower of WeVisOne-VL-2B): stage 1 of the soft ViT-B recipe
# (vitb_soft_pseudo.sh) with only the backbone swapped. Reference: soft ViT-B stage 1, holdout at epoch 10 = 0.6529
# (c1 0.560, c2 0.525, c3 0.757, c4 0.770). Rule, set before the run: compare epoch 10; within ~0.009 (seed sigma)
# is a tie and still proceeds to stage 2; clearly below stops this backbone. Runs in parallel with the C-RADIOv4-SO400M
# gate on the Mac.
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=../outputs/logs/qwen_gate.log
TAG=_qwenvit_soft_fh10
if ls ../outputs/checkpoints/fullft_lr0.0001_g0.85_h1024${TAG}*.pt >/dev/null 2>&1; then
  echo "REFUSING TO RUN: a checkpoint tagged $TAG already exists"; exit 1
fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -v '\[N/A\]' | tr -d ' ')" ] && \
   [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 1500 ]; then
  echo "REFUSING TO RUN: the GPU is busy"; exit 3
fi
: > "$L"
echo "=== qwen gate START $(date '+%m-%d %H:%M:%S') ===" >> "$L"
python -u train_full_ft.py --backbone qwenvit \
  --backbone-path "C:/Users/NewGr/Downloads/ViT Project severstal/severstal/external/WeVisOne-VL-2B" \
  --freeze-backbone --init-head "" --head-arch conv --hidden 1024 --bb-lr 1e-4 --bs 4 --seed 0 --epochs 10 \
  --save-epochs 10 --augment --aug-strength blackout --train-split train12k --val-split holdout \
  --weight-power 0 --target soft --tag $TAG >> "$L" 2>&1
echo "=== exit $? $(date '+%m-%d %H:%M:%S') ===" >> "$L"
