#!/bin/bash
# E31 stage 1 (the gate; seed repeat via the seed argument): C-RADIOv4-SO400M, frozen backbone, soft-target conv head. Same recipe as
# vitb_soft_pseudo.sh stage 1; only the backbone changes. Rule (pre-registered in docs/experiments.md):
# epoch-10 holdout vs 0.6529; within 0.009 = tie (proceeds), clearly below = stop. For seed 1: stop PC stage 2 if it falls back near 0.65.
# Mac launcher for the E31 stage-1 gate (PC adapter API). Frozen adapter runs save the head only.
# Usage: cradio_so_stage1.sh smoke|full [seed]   (seed defaults to 0; seed N>0 gets tag suffix _sN, log cradio_so_stage1_sN.log)
set -u
SEED=${2:-0}
SFX=$([ "$SEED" = 0 ] && echo '' || echo "_s$SEED")
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
CRADIO=${CRADIO_PATH:-../external/C-RADIOv4-SO400M}
CK=../outputs/checkpoints
mkdir -p "$CK" ../outputs/logs
COMMON="--backbone cradio --backbone-path $CRADIO --freeze-backbone --init-head '' --head-arch conv --hidden 1024
  --bb-lr 1e-4 --bs 4 --seed $SEED --augment --aug-strength blackout --train-split train12k --val-split holdout
  --weight-power 0 --target soft"
case "${1:-}" in
  smoke) L=../outputs/logs/cradio_so_smoke.log; : > "$L"
         eval ../.venv/bin/python -u train_full_ft.py $COMMON --max-train 200 --val-limit 40 --epochs 1 \
           --tag _cradio_so_smoke >> "$L" 2>&1 ;;
  full)  T=_cradio_so_fh10$SFX; L=../outputs/logs/cradio_so_stage1$SFX.log
         ls "$CK"/fullft_lr0.0001_g0.85_h1024${T}*.pt >/dev/null 2>&1 && { echo "REFUSING TO RUN: checkpoint tagged $T exists"; exit 1; }
         echo "=== E31 stage1 START $(date '+%m-%d %H:%M:%S') ===" > "$L"
         eval caffeinate -i ../.venv/bin/python -u train_full_ft.py $COMMON --epochs 10 --save-epochs 10 \
           --tag $T >> "$L" 2>&1
         echo "=== exit $? $(date '+%m-%d %H:%M:%S') ===" >> "$L" ;;
  *) echo "usage: $0 smoke|full"; exit 2 ;;
esac
