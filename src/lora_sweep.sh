#!/bin/bash
# Which LoRA configuration to spend the 14-hour ViT-L run on.
#
# Rank was measured FLAT (r4 0.7016, r8 0.7013, r16 0.7030) on ViT-S - but only
# ever with q/k/v adapters, 3 of 6 Linears per block, leaving the MLP (two thirds
# of block parameters) untouched. Full FT beat that LoRA by 0.022 at 11 sigma, so
# capacity was binding; rank simply could not reach it. This tests coverage and
# rank together, because "rank is flat" may itself be an artifact of narrow
# coverage.
#
# Matched: ViT-L, blackout aug, 1500 train images, 3 epochs, holdout eval, seed 0.
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
L=../outputs/logs/lora_sweep.log
: > "$L"
run () {
  echo "=== $1  $(date +%H:%M:%S) ===" >> "$L"
  shift
  python -u train_full_ft.py --model-id facebook/dinov3-vitl16-pretrain-lvd1689m \
    --init-head "" --grad-checkpoint --bs 4 --bb-lr 1e-4 --hidden 1024 --seed 0 \
    --augment --aug-strength blackout --max-train 1500 --epochs 3 \
    --train-split train12k --val-split holdout "$@" >> "$L" 2>&1
  echo "=== exit $? $(date +%H:%M:%S) ===" >> "$L"
}
run qkv_r16      --lora-r 16 --lora-targets q_proj k_proj v_proj              --tag _sw_qkv16
run all6_r16     --lora-r 16 --lora-targets q_proj k_proj v_proj o_proj up_proj down_proj --tag _sw_all16
run all6_r8      --lora-r 8  --lora-targets q_proj k_proj v_proj o_proj up_proj down_proj --tag _sw_all8
echo "=== SWEEP DONE $(date) ===" >> "$L"
