#!/bin/bash
# A/B: learned classifier (HeadNet conv) vs FIXED simplex-ETF classifier (SimplexHead),
# everything else identical. ViT-S, the vits_lora_r16 recipe, two stages per arm:
#   stage 1  10-epoch head on a FROZEN backbone (LP stage, cold start)
#   stage 2  10-epoch joint LoRA r16 all-6 from that head (FT stage)
# The control arm doubles as the matched fine-tune control for the continued-
# pretraining experiment (same 10-epoch head recipe for both).
#
# Compare on Kaggle solo at MATCHED positive rates: a cosine head changes the
# probability scale, so the tuned PIX/MINA do not transfer between arms.
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
L=../outputs/logs/simplex_ab.log
CK=../outputs/checkpoints
ALL6="q_proj k_proj v_proj o_proj up_proj down_proj"
COMMON="--hidden 1024 --bb-lr 1e-4 --gamma 0.85 --bs 4 --seed 0 --epochs 10 --save-epochs 10 \
  --augment --aug-strength blackout --train-split train12k --val-split holdout"

for t in _vits_ctl_frozenhead10 _vits_splx_frozenhead10 _vits_lora_r16_ctl10 _vits_lora_r16_splx10; do
  if ls "$CK"/fullft_lr0.0001_g0.85_h1024${t}*.pt >/dev/null 2>&1; then
    echo "REFUSING TO RUN: a checkpoint tagged $t already exists"; exit 1
  fi
done
: > "$L"

stage () { echo "=== $1 START $(date '+%m-%d %H:%M:%S') ===" >> "$L"; shift
           python -u train_full_ft.py "$@" >> "$L" 2>&1
           rc=$?; echo "=== exit $rc $(date '+%m-%d %H:%M:%S') ===" >> "$L"; return $rc; }

# refuse a joint stage whose warm head would silently fall back to random init
check_head () {
  python - "$CK/$1" "$2" >> "$L" 2>&1 <<'EOF'
import sys, torch
from spatial_head import build
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
h = build(sys.argv[2], dim=384 * 4, upscale=4, hidden=1024).state_dict()
ok = set(ck["head"]) == set(h) and all(ck["head"][k].shape == h[k].shape for k in h)
print(f"warm-head check {sys.argv[1]} as {sys.argv[2]}: {'OK' if ok else 'INCOMPATIBLE'} (saved arch {ck.get('head_arch')})")
sys.exit(0 if ok and ck.get("head_arch") == sys.argv[2] else 1)
EOF
}

stage ctl_frozenhead  --freeze-backbone --init-head "" --head-arch conv    $COMMON --tag _vits_ctl_frozenhead10  || exit 1
stage splx_frozenhead --freeze-backbone --init-head "" --head-arch simplex $COMMON --tag _vits_splx_frozenhead10 || exit 1

check_head fullft_lr0.0001_g0.85_h1024_vits_ctl_frozenhead10_ep10.pt conv || { echo "=== ABORT ctl head ===" >> "$L"; exit 1; }
stage ctl_joint --lora-r 16 --lora-alpha 32 --lora-targets $ALL6 --head-arch conv \
  --init-head fullft_lr0.0001_g0.85_h1024_vits_ctl_frozenhead10_ep10.pt $COMMON --tag _vits_lora_r16_ctl10

check_head fullft_lr0.0001_g0.85_h1024_vits_splx_frozenhead10_ep10.pt simplex || { echo "=== ABORT splx head ===" >> "$L"; exit 1; }
stage splx_joint --lora-r 16 --lora-alpha 32 --lora-targets $ALL6 --head-arch simplex \
  --init-head fullft_lr0.0001_g0.85_h1024_vits_splx_frozenhead10_ep10.pt $COMMON --tag _vits_lora_r16_splx10

echo "=== SIMPLEX AB DONE $(date '+%m-%d %H:%M:%S') ===" >> "$L"
