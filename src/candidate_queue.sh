#!/bin/bash
# Candidate ensemble members, built while the Kaggle GPU quota is out.
#
# Guiding principle from the results so far: single-model quality barely moves the
# target score (every single lands 0.899-0.901), while each NEW DIVERSITY AXIS has
# paid - lora_r8 +0.00194, bce16 +0.00138. So these are built to be DIFFERENT from
# the existing four, not better than them.
#
# Existing members cover: ViT-B/fullFT/blackout/12k, ViT-S/fullFT/strong/12k,
# ViT-S/LoRA-qkv/unaugmented/8.8k, ViT-S/sigmoid/blackout/12k, plus ViT-L/LoRA.
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
L=../outputs/logs/candidates.log
: > "$L"
ALL6="q_proj k_proj v_proj o_proj up_proj down_proj"
run () {
  echo "=== $1 START $(date '+%m-%d %H:%M:%S') ===" >> "$L"
  shift
  python -u train_full_ft.py "$@" >> "$L" 2>&1
  echo "=== exit $? $(date '+%m-%d %H:%M:%S') ===" >> "$L"
}

# 1. ViT-L LoRA + sigmoid/BCE (~14h). Combines the two axes that each paid on their
#    own - the new backbone scale and the per-class decision structure - which have
#    never been combined. pos-weight-power 0.15 as validated: inheriting 0.30 gives
#    c2 a pos_weight of 73 and pays the model heavily to fire, against the one thing
#    we know (we over-predict on target).
run vitl_lora_bce \
  --model-id facebook/dinov3-vitl16-pretrain-lvd1689m --init-head "" \
  --lora-r 16 --lora-alpha 16 --lora-targets $ALL6 --grad-checkpoint \
  --loss bce_dice --pos-weight-power 0.15 \
  --bb-lr 1e-4 --hidden 1024 --bs 4 --seed 0 --epochs 10 --save-epochs 10 \
  --augment --aug-strength blackout --train-split train12k --val-split holdout \
  --tag _vitl_lora_bce

# 2. ViT-B LoRA all-6 (~7h). LoRA currently exists only on ViT-S, and only over
#    q/k/v - the narrow coverage the sweep showed was the binding constraint.
run vitb_lora \
  --model-id facebook/dinov3-vitb16-pretrain-lvd1689m --init-head "" \
  --lora-r 16 --lora-alpha 16 --lora-targets $ALL6 --grad-checkpoint \
  --bb-lr 1e-4 --hidden 1024 --bs 4 --seed 0 --epochs 10 --save-epochs 10 \
  --augment --aug-strength blackout --train-split train12k --val-split holdout \
  --tag _vitb_lora

# 3. ViT-S full FT with DIFFERENT TAPS (~3.5h). Every model in the project reads
#    blocks (2,5,8,11); (0,3,7,11) feeds the head a different mix of early and late
#    features. Cold head, since the warm-start head was trained on (2,5,8,11)
#    features and would be a mismatch.
run vits_taps \
  --init-head "" --taps 0 3 7 11 \
  --bb-lr 1e-4 --gamma 0.85 --hidden 1024 --bs 4 --seed 0 --epochs 10 --save-epochs 10 \
  --augment --aug-strength blackout --train-split train12k --val-split holdout \
  --tag _vits_taps0371

echo "=== QUEUE DONE $(date) ===" >> "$L"
