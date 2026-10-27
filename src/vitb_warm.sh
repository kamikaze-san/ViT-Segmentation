#!/bin/bash
# Give ViT-B the warm head it has never had.
#
# The warm head (spatial_conv_up4_h1024_wp0.3_dice1.0_ep30.pt) is ViT-S shaped -
# first conv (1024,1536,1,1) for 4x384. ViT-B needs 3072 and ViT-L 4096, so BOTH
# always trained COLD while every ViT-S model trained WARM. Measured cost of a cold
# head: -0.00627 on target (vits_lora_r16 0.90063 vs jnt_conv 0.89436), the only
# local finding that transferred - rank, taps, decoder and copy-paste were all
# noise or inverted.
#
# ViT-B scored 0.90006 COLD against warm ViT-S at 0.90082/0.90063, i.e. it is
# already competitive while carrying that handicap.
#
# Stage 1 builds a ViT-B-shaped head on FROZEN ViT-B features. Stage 2 uses it as
# --init-head for a joint LoRA run, matching the vits_lora_r16 recipe otherwise.
#
# Stage 1 runs 10 epochs, not the original head's 30. That is deliberate: if 10 is
# enough, the benefit is INITIALISATION; if it under-recovers, the benefit was
# partly just more total head training. Those are confounded in the ViT-S warm head
# and this separates them.
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
L=../outputs/logs/vitb_warm.log
: > "$L"
MID=facebook/dinov3-vitb16-pretrain-lvd1689m
ALL6="q_proj k_proj v_proj o_proj up_proj down_proj"

echo "=== stage1 frozen-head START $(date '+%m-%d %H:%M:%S') ===" >> "$L"
python -u train_full_ft.py --model-id $MID --freeze-backbone --init-head "" \
  --head-arch conv --hidden 1024 --bb-lr 1e-4 --bs 4 --seed 0 --epochs 10 \
  --save-epochs 10 --augment --aug-strength blackout \
  --train-split train12k --val-split holdout --tag _vitb_frozenhead >> "$L" 2>&1
echo "=== exit $? $(date '+%m-%d %H:%M:%S') ===" >> "$L"

echo "=== stage2 joint-warm START $(date '+%m-%d %H:%M:%S') ===" >> "$L"
python -u train_full_ft.py --model-id $MID --lora-r 16 --lora-alpha 32 --lora-targets $ALL6 \
  --grad-checkpoint --head-arch conv --hidden 1024 \
  --init-head fullft_lr0.0001_g0.85_h1024_vitb_frozenhead_ep10.pt \
  --bb-lr 1e-4 --gamma 0.85 --bs 4 --seed 0 --epochs 10 --save-epochs 10 \
  --augment --aug-strength blackout --train-split train12k --val-split holdout \
  --tag _vitb_lora_warm >> "$L" 2>&1
echo "=== exit $? $(date '+%m-%d %H:%M:%S') ===" >> "$L"
echo "=== VITB WARM DONE $(date) ===" >> "$L"
