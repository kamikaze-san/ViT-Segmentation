#!/bin/bash
# E26: the strongest member rebuilt with E25's recipe (soft per-cell targets, no
# class weights - +0.00107 private on ViT-S). ViT-B LoRA warm + round-1 pseudo labels,
# matched against vitb_lora_warm_pseudo01 (Kaggle 0.90606): same backbone, 10-epoch
# frozen head, LoRA r16 all-6, data (train12k + 3,402 round-1 pseudo labels),
# blackout, seed 0 - only the training targets and class weights differ, in both
# stages. If it wins solo, swap it into V79 (0.90874) in place of that member.
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
L=../outputs/logs/vitb_soft_pseudo.log
CK=../outputs/checkpoints
PS=../data/processed/pseudo_ids.csv
MID=facebook/dinov3-vitb16-pretrain-lvd1689m
ALL6="q_proj k_proj v_proj o_proj up_proj down_proj"
CAL="--weight-power 0 --target soft"
HEAD=fullft_lr0.0001_g0.85_h1024_vitb_soft_fh10_ep10.pt

[ -f "$PS" ] || { echo "REFUSING TO RUN: $PS missing"; exit 1; }
for t in _vitb_soft_fh10 _vitb_lora_warm_soft_pseudo01; do
  if ls "$CK"/fullft_lr0.0001_g0.85_h1024${t}*.pt >/dev/null 2>&1; then
    echo "REFUSING TO RUN: a checkpoint tagged $t already exists"; exit 1
  fi
done
: > "$L"

stage () { echo "=== $1 START $(date '+%m-%d %H:%M:%S') ===" >> "$L"; shift
           python -u train_full_ft.py "$@" >> "$L" 2>&1
           rc=$?; echo "=== exit $rc $(date '+%m-%d %H:%M:%S') ===" >> "$L"; return $rc; }

# stage 1: ViT-B head on the FROZEN backbone, train12k only (as vitb_warm.sh stage 1)
stage frozenhead --model-id $MID --freeze-backbone --init-head "" --head-arch conv --hidden 1024 \
  --bb-lr 1e-4 --bs 4 --seed 0 --epochs 10 --save-epochs 10 --augment --aug-strength blackout \
  --train-split train12k --val-split holdout $CAL --tag _vitb_soft_fh10 || exit 1

# refuse the joint stage if the warm head is not a soft-target ViT-B conv head
python - "$CK/$HEAD" >> "$L" 2>&1 <<'EOF'
import sys, torch
from spatial_head import build
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
h = build("conv", dim=768 * 4, upscale=4, hidden=1024).state_dict()
ok = (set(ck["head"]) == set(h) and all(ck["head"][k].shape == h[k].shape for k in h)
      and ck.get("target") == "soft" and ck.get("loss") == "ce_dice")
print(f"warm-head check {sys.argv[1]}: {'OK' if ok else 'MISMATCH'} (target {ck.get('target')}, loss {ck.get('loss')})")
sys.exit(0 if ok else 1)
EOF
[ $? -eq 0 ] || { echo "=== ABORT head ===" >> "$L"; exit 1; }

# stage 2: joint LoRA + round-1 pseudo labels (as vitb_pseudo.sh)
stage joint --model-id $MID --lora-r 16 --lora-alpha 32 --lora-targets $ALL6 --grad-checkpoint \
  --head-arch conv --hidden 1024 --init-head $HEAD \
  --bb-lr 1e-4 --gamma 0.85 --bs 4 --seed 0 --epochs 10 --save-epochs 10 \
  --augment --aug-strength blackout --train-split train12k --val-split holdout \
  --pseudo-ids "$PS" $CAL --tag _vitb_lora_warm_soft_pseudo01

echo "=== VITB SOFT PSEUDO DONE $(date '+%m-%d %H:%M:%S') ===" >> "$L"
