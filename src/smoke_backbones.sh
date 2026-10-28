#!/bin/bash
# Smoke tests for the backbone adapters (no long runs). Each step is its own process.
#  1 DINOv3 regression: the refactored script must reproduce the soft ViT-B stage-1 epoch-0 holdout score (0.0160)
#  2 Qwen patchify == Qwen's own processor
#  3-6 frozen-head and LoRA steps for C-RADIOv4-SO400M and the Qwen ViT: loss, speed, peak memory, checkpoint save
cd "$(dirname "$0")"
L=../outputs/logs/smoke_backbones.log
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EXT="C:/Users/NewGr/Downloads/ViT Project severstal/severstal/external"
COMMON="--head-arch conv --hidden 1024 --bb-lr 1e-4 --bs 4 --seed 0 --augment --aug-strength blackout
        --train-split train12k --val-split holdout --weight-power 0 --target soft"
: > "$L"
step () { echo "=== $1 START $(date '+%H:%M:%S') ===" >> "$L"; shift; "$@" >> "$L" 2>&1; echo "=== exit $? $(date '+%H:%M:%S') ===" >> "$L"; }

step dino_epoch0 python -u train_full_ft.py --model-id facebook/dinov3-vitb16-pretrain-lvd1689m --freeze-backbone \
  --head-arch conv --hidden 1024 --bb-lr 1e-4 --bs 4 --seed 0 --epochs 0 --augment --aug-strength blackout \
  --train-split train12k --val-split holdout --weight-power 0 --target soft --init-head "" --tag _SMOKE_dino_e0
step qwen_patchify python -u qwen_patchify_check.py
step cradio_frozen python -u train_full_ft.py --backbone cradio --backbone-path "$EXT/C-RADIOv4-SO400M" \
  --freeze-backbone $COMMON --init-head "" --epochs 1 --max-train 200 --val-limit 100 --tag _SMOKE_cradio_fh
step cradio_lora python -u train_full_ft.py --backbone cradio --backbone-path "$EXT/C-RADIOv4-SO400M" \
  --lora-r 16 --lora-alpha 32 --grad-checkpoint --gamma 0.85 $COMMON --init-head "" --epochs 1 --max-train 40 --val-limit 20 --tag _SMOKE_cradio_jt
step qwen_frozen python -u train_full_ft.py --backbone qwenvit --backbone-path "$EXT/WeVisOne-VL-2B" \
  --freeze-backbone $COMMON --init-head "" --epochs 1 --max-train 200 --val-limit 100 --tag _SMOKE_qwen_fh
step qwen_lora python -u train_full_ft.py --backbone qwenvit --backbone-path "$EXT/WeVisOne-VL-2B" \
  --lora-r 16 --lora-alpha 32 --grad-checkpoint --gamma 0.85 $COMMON --init-head "" --epochs 1 --max-train 40 --val-limit 20 --tag _SMOKE_qwen_jt
echo "=== SMOKE DONE $(date '+%H:%M:%S') ===" >> "$L"
