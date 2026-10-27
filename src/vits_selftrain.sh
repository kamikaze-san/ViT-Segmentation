#!/bin/bash
# Per-member self-training (E22): vits_lora_r16's exact recipe (ablation_queue.sh,
# `run vits_lora_r16`) with ONE change - pseudo-labels made by vits_lora_r16 ITSELF
# (self_teacher.py: its own 3 TTA views as voters, single-model thresholds), instead
# of the shared 0.90756 ensemble teacher behind vits_lora_r16_pseudo01.
#
# Compare solo: original 0.90063 | shared-teacher pseudo 0.90380 (V84, solo config).
# Before any submission, measure its test-set disagreement with the V79 members:
# the point is to stay DIFFERENT (E21: shared-teacher students converged).
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
L=../outputs/logs/vits_selftrain.log
PS=../data/processed/pseudo_self_vits_ids.csv
MD=pseudo_self_vits_masks
TAG=_vits_lora_r16_self01
ALL6="q_proj k_proj v_proj o_proj up_proj down_proj"

if [ ! -f "$PS" ]; then echo "REFUSING TO RUN: $PS missing - run self_teacher.py + build_pseudo_labels.py first"; exit 1; fi
if ls ../outputs/checkpoints/fullft_lr0.0001_g0.85_h1024${TAG}*.pt >/dev/null 2>&1; then
  echo "REFUSING TO RUN: a checkpoint tagged $TAG already exists"; exit 1
fi
: > "$L"
echo "=== self-train vits START $(date '+%m-%d %H:%M:%S') | $(($(wc -l < "$PS") - 1)) self-labelled images ===" >> "$L"
python -u train_full_ft.py --bb-lr 1e-4 --gamma 0.85 --hidden 1024 --bs 4 --seed 0 --epochs 10 \
  --save-epochs 10 --augment --aug-strength blackout \
  --train-split train12k --val-split holdout \
  --lora-r 16 --lora-alpha 32 --lora-targets $ALL6 \
  --pseudo-ids "$PS" --pseudo-mask-dir "$MD" --tag "$TAG" >> "$L" 2>&1
echo "=== exit $? $(date '+%m-%d %H:%M:%S') ===" >> "$L"
