#!/bin/bash
# Resume the E26 chain after its launcher (vitb_soft_pseudo.sh) was killed by a tool
# time limit while stage 1 kept running as an orphaned process. Does NOT re-run
# stage 1: waits for that process to exit, verifies stage 1 completed, then runs the
# same warm-head check and the same stage-2 command as vitb_soft_pseudo.sh, appending
# to the same log.
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
STAGE1_PID=${1:?usage: vitb_soft_pseudo_resume.sh <stage-1 windows pid>}

alive () { powershell -NoProfile -Command "if (Get-Process -Id $STAGE1_PID -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; }
while alive; do sleep 60; done
echo "=== exit (stage-1 process $STAGE1_PID ended; launcher had been killed) $(date '+%m-%d %H:%M:%S') ===" >> "$L"

if ! grep -qa "BEST val mean Dice" "$L" || [ ! -f "$CK/$HEAD" ]; then
  echo "=== ABORT: stage 1 did not complete (no BEST line or no $HEAD) ===" >> "$L"; exit 1
fi
if ls "$CK"/fullft_lr0.0001_g0.85_h1024_vitb_lora_warm_soft_pseudo01*.pt >/dev/null 2>&1; then
  echo "=== ABORT: stage-2 checkpoint already exists ===" >> "$L"; exit 1
fi

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

echo "=== joint START $(date '+%m-%d %H:%M:%S') ===" >> "$L"
python -u train_full_ft.py --model-id $MID --lora-r 16 --lora-alpha 32 --lora-targets $ALL6 --grad-checkpoint \
  --head-arch conv --hidden 1024 --init-head $HEAD \
  --bb-lr 1e-4 --gamma 0.85 --bs 4 --seed 0 --epochs 10 --save-epochs 10 \
  --augment --aug-strength blackout --train-split train12k --val-split holdout \
  --pseudo-ids "$PS" $CAL --tag _vitb_lora_warm_soft_pseudo01 >> "$L" 2>&1
echo "=== exit $? $(date '+%m-%d %H:%M:%S') ===" >> "$L"
echo "=== VITB SOFT PSEUDO DONE $(date '+%m-%d %H:%M:%S') ===" >> "$L"
