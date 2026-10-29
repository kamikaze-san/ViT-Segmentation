#!/bin/bash
# Mac chain, pre-registered in docs/experiments.md (E31 follow-ups):
#  (a) full-epoch head-only control: the PC's run 2 epoch 1 recipe with --freeze-backbone instead of LoRA
#  (b) seed-1 stage-1 gate: cradio_so_stage1.sh full 1
# Usage: nohup caffeinate -i bash cradio_mac_chain.sh > ../outputs/logs/cradio_mac_chain.log 2>&1 &
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1
CK=../outputs/checkpoints
HEAD=fullft_lr0.0001_g0.85_h1024_cradio_so_fh10_ep10.pt
TAG=_cradio_headonly_fullep1
L=../outputs/logs/cradio_headonly_fullep1.log
stamp () { date '+%m-%d %H:%M:%S'; }
[ -f "$CK/$HEAD" ] || { echo "REFUSING TO RUN: $HEAD missing"; exit 1; }
if ls "$CK"/fullft_lr0.0001_g0.85_h1024${TAG}*.pt >/dev/null 2>&1; then echo "REFUSING TO RUN (a): checkpoint tagged $TAG exists"; exit 1; fi
echo "=== (a) head-only full epoch START $(stamp) ===" | tee "$L"
../.venv/bin/python -u train_full_ft.py --backbone cradio --backbone-path ../external/C-RADIOv4-SO400M \
  --freeze-backbone --head-arch conv --hidden 1024 --init-head $HEAD --bb-lr 1e-4 --gamma 0.85 --bs 4 --seed 0 \
  --epochs 1 --augment --aug-strength blackout --train-split train12k --val-split holdout \
  --pseudo-ids ../data/processed/pseudo_ids.csv --weight-power 0 --target soft --tag $TAG >> "$L" 2>&1
echo "=== (a) exit $? $(stamp) ===" | tee -a "$L"
echo "=== (b) seed-1 gate START $(stamp) ==="
bash ./cradio_so_stage1.sh full 1
echo "=== (b) exit $? $(stamp) ==="
