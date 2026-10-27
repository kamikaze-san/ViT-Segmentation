#!/bin/bash
# Pseudo-label student: vitb_lora_warm EXACTLY (stage 2 of vitb_warm.sh - same warm
# head, LoRA r16 all-6, blackout, 10 epochs, seed 0, ep10 snapshot) with one change:
# the pseudo-labelled competition test images from build_pseudo_labels.py are
# appended to train12k. Class weights are still computed from train12k alone.
#
# Baseline to beat, single model, same thresholds: 0.90424 private / 0.91595 public.
#
# Teacher = the 0.90756 5-member submission (pseudo_teacher.py, verified against
# E17 decisions; reproduced the submission defect rate 10.95% vs 10.97%).
# Labels = all 5 members agree AND (defect: ensemble fires | clean: max-prob < 0.1);
# an image is kept only if all 4 classes get a label.
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
L=../outputs/logs/vitb_pseudo.log
PS=../data/processed/pseudo_ids.csv
MID=facebook/dinov3-vitb16-pretrain-lvd1689m
ALL6="q_proj k_proj v_proj o_proj up_proj down_proj"

if [ ! -f "$PS" ]; then
  echo "REFUSING TO RUN: $PS missing - run build_pseudo_labels.py --write 0.1 first"
  exit 1
fi
: > "$L"
echo "=== pseudo student START $(date '+%m-%d %H:%M:%S') | $(($(wc -l < "$PS") - 1)) pseudo images ===" >> "$L"
python -u train_full_ft.py --model-id $MID --lora-r 16 --lora-alpha 32 --lora-targets $ALL6 \
  --grad-checkpoint --head-arch conv --hidden 1024 \
  --init-head fullft_lr0.0001_g0.85_h1024_vitb_frozenhead_ep10.pt \
  --bb-lr 1e-4 --gamma 0.85 --bs 4 --seed 0 --epochs 10 --save-epochs 10 \
  --augment --aug-strength blackout --train-split train12k --val-split holdout \
  --pseudo-ids "$PS" --tag _vitb_lora_warm_pseudo01 >> "$L" 2>&1
echo "=== exit $? $(date '+%m-%d %H:%M:%S') ===" >> "$L"
