#!/bin/bash
# Calibration A/B (E25): does training on UNBIASED targets fix the over-extension?
# E24 found the model over-extends even on its own training images because it was
# taught to: stride-4 targets mark a cell defect if ANY of its 16 pixels is (labels
# inflated 6-23%), and weighted CE makes a missed defect pixel 3-13x costlier than a
# false one. Both arms below drop both biases; B also adds a Kaggle-shaped loss.
#
#   A  soft (per-cell fraction) targets + no class weights, CE + batch soft Dice
#   B  A + per-image Dice + per-pair top-k presence term (--loss metric)
#
# Control = E23 control arm (any-pixel targets, class weights, CE + batch Dice),
# identical otherwise: ViT-S, 10-epoch frozen head -> 10-epoch joint LoRA r16 all-6,
# train12k, blackout, seed 0. Kaggle solo 0.89912. Compare at matched test-set rates.
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
L=../outputs/logs/calib_ab.log
CK=../outputs/checkpoints
ALL6="q_proj k_proj v_proj o_proj up_proj down_proj"
COMMON="--hidden 1024 --bb-lr 1e-4 --gamma 0.85 --bs 4 --seed 0 --epochs 10 --save-epochs 10 \
  --augment --aug-strength blackout --train-split train12k --val-split holdout \
  --head-arch conv --weight-power 0 --target soft"
LOSS_A="--loss ce_dice"
LOSS_B="--loss metric --presence-w 1.0"

for t in _vits_soft_fh10 _vits_metric_fh10 _vits_lora_r16_soft10 _vits_lora_r16_metric10; do
  if ls "$CK"/fullft_lr0.0001_g0.85_h1024${t}*.pt >/dev/null 2>&1; then
    echo "REFUSING TO RUN: a checkpoint tagged $t already exists"; exit 1
  fi
done
: > "$L"

stage () { echo "=== $1 START $(date '+%m-%d %H:%M:%S') ===" >> "$L"; shift
           python -u train_full_ft.py "$@" >> "$L" 2>&1
           rc=$?; echo "=== exit $rc $(date '+%m-%d %H:%M:%S') ===" >> "$L"; return $rc; }

# refuse a joint stage whose warm head is not this arm's (shape, target and loss)
check_head () {
  python - "$CK/$1" "$2" >> "$L" 2>&1 <<'EOF'
import sys, torch
from spatial_head import build
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
h = build("conv", dim=384 * 4, upscale=4, hidden=1024).state_dict()
ok = (set(ck["head"]) == set(h) and all(ck["head"][k].shape == h[k].shape for k in h)
      and ck.get("target") == "soft" and ck.get("loss") == sys.argv[2])
print(f"warm-head check {sys.argv[1]}: {'OK' if ok else 'MISMATCH'} "
      f"(arch {ck.get('head_arch')}, target {ck.get('target')}, loss {ck.get('loss')}, want {sys.argv[2]})")
sys.exit(0 if ok else 1)
EOF
}

stage A_frozenhead --freeze-backbone --init-head "" $LOSS_A $COMMON --tag _vits_soft_fh10   || exit 1
stage B_frozenhead --freeze-backbone --init-head "" $LOSS_B $COMMON --tag _vits_metric_fh10 || exit 1

check_head fullft_lr0.0001_g0.85_h1024_vits_soft_fh10_ep10.pt ce_dice || { echo "=== ABORT A head ===" >> "$L"; exit 1; }
stage A_joint --lora-r 16 --lora-alpha 32 --lora-targets $ALL6 \
  --init-head fullft_lr0.0001_g0.85_h1024_vits_soft_fh10_ep10.pt $LOSS_A $COMMON --tag _vits_lora_r16_soft10

check_head fullft_lr0.0001_g0.85_h1024_vits_metric_fh10_ep10.pt metric || { echo "=== ABORT B head ===" >> "$L"; exit 1; }
stage B_joint --lora-r 16 --lora-alpha 32 --lora-targets $ALL6 \
  --init-head fullft_lr0.0001_g0.85_h1024_vits_metric_fh10_ep10.pt $LOSS_B $COMMON --tag _vits_lora_r16_metric10

echo "=== CALIB AB DONE $(date '+%m-%d %H:%M:%S') ===" >> "$L"
