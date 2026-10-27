#!/bin/bash
# E27 step 3, the gate: does the continued-pretrained backbone (cpt_vitb_ep5.pt) give
# better frozen features for segmentation than stock DINOv3 ViT-B?
# Exactly stage 1 of vitb_soft_pseudo.sh (stock: holdout best 0.6567, ep10 0.6529),
# with only --backbone-ckpt added. Pre-registered rule: compare ep10 holdout mean;
# within ~0.009 (seed sigma) is a tie, clearly below stock means stop E27.
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
L=../outputs/logs/cpt_gate.log
CK=../outputs/checkpoints
MID=facebook/dinov3-vitb16-pretrain-lvd1689m
CAL="--weight-power 0 --target soft"
BB=cpt_vitb_ep5.pt
TAG=_vitb_cpt_soft_fh10
OUT=fullft_lr0.0001_g0.85_h1024${TAG}_ep10.pt

[ -f "$CK/$BB" ] || { echo "REFUSING TO RUN: $CK/$BB missing"; exit 1; }
if ls "$CK"/fullft_lr0.0001_g0.85_h1024${TAG}*.pt >/dev/null 2>&1; then
  echo "REFUSING TO RUN: a checkpoint tagged $TAG already exists"; exit 1
fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "REFUSING TO RUN: another process is using the GPU"; exit 3
fi
: > "$L"

echo "=== frozenhead (CPT backbone) START $(date '+%m-%d %H:%M:%S') ===" >> "$L"
python -u train_full_ft.py --model-id $MID --backbone-ckpt $BB --freeze-backbone --init-head "" \
  --head-arch conv --hidden 1024 --bb-lr 1e-4 --bs 4 --seed 0 --epochs 10 --save-epochs 10 \
  --augment --aug-strength blackout --train-split train12k --val-split holdout $CAL \
  --tag $TAG >> "$L" 2>&1
rc=$?; echo "=== exit $rc $(date '+%m-%d %H:%M:%S') ===" >> "$L"
[ $rc -eq 0 ] || exit $rc

# provenance: the frozen backbone saved with the head must be the CPT backbone, not stock
python - "$CK/$BB" "$CK/$OUT" >> "$L" 2>&1 <<'EOF'
import sys, torch
a = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["backbone"]
b = torch.load(sys.argv[2], map_location="cpu", weights_only=False)["backbone"]
d = max(float((a[k].float() - b[k].float()).abs().max()) for k in a)
print(f"provenance: max |cpt - saved backbone| = {d:.3g} -> {'OK' if d == 0 and set(a) == set(b) else 'MISMATCH'}")
EOF
echo "=== CPT GATE DONE $(date '+%m-%d %H:%M:%S') ===" >> "$L"
