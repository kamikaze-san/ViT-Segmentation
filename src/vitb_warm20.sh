#!/bin/bash
# ViT-B warm head, done cleanly with a longer head phase.
#
# The first attempt used a 10-epoch frozen head -> 0.90424 on target, the best
# single model. Two reasons to think more head training helps: ViT-S's warm head
# had 30 epochs and gave +0.00627 on target vs ViT-B's 10-epoch head at +0.00471,
# and the ViT-B frozen head was still improving at epoch 10 (0.6607 -> 0.6658).
#
# Stage 1 is 20 epochs FROM SCRATCH with a single cosine - not a resume. The
# earlier 10+10 attempt restarted the LR at 1e-4 with AdamW moments from zero and
# immediately fell from 0.6658 to 0.6431; that is the same warm-restart failure
# this project already documented twice (C1 32ep, Lovasz 8->13, both worse on
# target). A clean single schedule avoids it.
#
# More JOINT epochs are deliberately NOT tried: 10 vs 16 measured as a tie on
# target (vits_ft10 0.89960 vs blackout 0.89933).
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
L=../outputs/logs/vitb_warm20.log
: > "$L"
MID=facebook/dinov3-vitb16-pretrain-lvd1689m
ALL6="q_proj k_proj v_proj o_proj up_proj down_proj"

echo "=== stage1 frozen-head 20ep clean START $(date '+%m-%d %H:%M:%S') ===" >> "$L"
python -u train_full_ft.py --model-id $MID --freeze-backbone --init-head "" \
  --head-arch conv --hidden 1024 --bb-lr 1e-4 --bs 4 --seed 0 --epochs 20 \
  --save-epochs 20 --augment --aug-strength blackout \
  --train-split train12k --val-split holdout --tag _vitb_fh20 >> "$L" 2>&1
echo "=== exit $? $(date '+%m-%d %H:%M:%S') ===" >> "$L"

echo "=== stage2 joint-warm20 START $(date '+%m-%d %H:%M:%S') ===" >> "$L"
python -u train_full_ft.py --model-id $MID --lora-r 16 --lora-alpha 32 --lora-targets $ALL6 \
  --grad-checkpoint --head-arch conv --hidden 1024 \
  --init-head fullft_lr0.0001_g0.85_h1024_vitb_fh20_ep20.pt \
  --bb-lr 1e-4 --gamma 0.85 --bs 4 --seed 0 --epochs 10 --save-epochs 10 \
  --augment --aug-strength blackout --train-split train12k --val-split holdout \
  --tag _vitb_lora_warm20 >> "$L" 2>&1
echo "=== exit $? $(date '+%m-%d %H:%M:%S') ===" >> "$L"
echo "=== VITB WARM20 DONE $(date) ===" >> "$L"
