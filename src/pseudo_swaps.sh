#!/bin/bash
# Pseudo-label versions of two more 0.90874-ensemble members, run back to back.
# Each command is the member's ORIGINAL recipe with exactly one addition:
# --pseudo-ids (round-1 labels from the independent 0.90756 teacher, 3,402 images).
# Class weights and BCE pos_weight are still computed from train12k alone.
#
#   stage 1  vits_lora_r16 : ablation_queue.sh  `run vits_lora_r16` (no grad checkpoint,
#                            default ViT-S warm head)          Kaggle solo 0.90063
#   stage 2  bce16         : overnight_2026_08_31.sh bce_dice  Kaggle solo 0.90082
#
# Launch-time checks against the originals' logs:
#   class weights  [0.17, 1.367, 2.176, 0.494, 0.792]
#   head initialised from spatial_conv_up4_h1024_wp0.3_dice1.0_ep30.pt (val 0.7135...)
#   stage 1: LoRA 1,327,104 trainable params | stage 2: bce pos_weight [2.85, 3.595, 1.707, 2.168]
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
L=../outputs/logs/pseudo_swaps.log
PS=../data/processed/pseudo_ids.csv
CK=../outputs/checkpoints
ALL6="q_proj k_proj v_proj o_proj up_proj down_proj"

if [ ! -f "$PS" ]; then echo "REFUSING TO RUN: $PS missing"; exit 1; fi
for t in _vits_lora_r16_pseudo01 _bce_blackout_pseudo01; do
  if ls "$CK"/fullft_lr0.0001_g0.85_h1024${t}*.pt >/dev/null 2>&1; then
    echo "REFUSING TO RUN: a checkpoint tagged $t already exists"; exit 1
  fi
done
: > "$L"

echo "=== stage1 vits_lora_r16_pseudo START $(date '+%m-%d %H:%M:%S') ===" >> "$L"
python -u train_full_ft.py --bb-lr 1e-4 --gamma 0.85 --hidden 1024 --bs 4 --seed 0 --epochs 10 \
  --save-epochs 10 --augment --aug-strength blackout \
  --train-split train12k --val-split holdout \
  --lora-r 16 --lora-alpha 32 --lora-targets $ALL6 \
  --pseudo-ids "$PS" --tag _vits_lora_r16_pseudo01 >> "$L" 2>&1
echo "=== exit $? stage1 $(date '+%m-%d %H:%M:%S') ===" >> "$L"

echo "=== stage2 bce16_pseudo START $(date '+%m-%d %H:%M:%S') ===" >> "$L"
python -u train_full_ft.py --loss bce_dice --pos-weight-power 0.15 \
  --bb-lr 1e-4 --gamma 0.85 --hidden 1024 --epochs 16 --seed 0 \
  --augment --aug-strength blackout \
  --train-split train12k --val-split holdout --save-epochs 16 \
  --pseudo-ids "$PS" --tag _bce_blackout_pseudo01 >> "$L" 2>&1
echo "=== exit $? stage2 $(date '+%m-%d %H:%M:%S') ===" >> "$L"
echo "=== PSEUDO SWAPS DONE $(date '+%m-%d %H:%M:%S') ===" >> "$L"
