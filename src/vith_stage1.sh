#!/bin/bash
# DINOv3 ViT-H+/16 stage-1 gate (840M, 1280 x 32 blocks, Meta's official distill of the 7B, like our S/B/L), soft
# recipe: exactly stage 1 of vitb_soft_pseudo.sh with only the model swapped. Frozen backbone, so no LoRA or
# train-mode questions; train_full_ft.py rescales the taps to (7, 15, 23, 31) and the depth map to 32 blocks itself.
# Pre-registered rule, epoch-10 holdout mean against C-RADIO's two-seed gate mean 0.688 (seed sigma ~0.009):
#   >= 0.697 clearly better -> stage 2 worth considering; 0.679-0.697 tie with C-RADIO -> not worth a 40-50 h stage 2;
#   < 0.679 -> stop. Reference: soft ViT-B 0.6529.
# Usage: vith_stage1.sh [seed]     env: VITH_PATH (default ../external/dinov3-vith16plus), PYTHON (default python)
# Frozen DINOv3 checkpoints carry the stock backbone: ~3.4 GB each, two of them (best + epoch-10 snapshot).
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
SEED=${1:-0}
SFX=$([ "$SEED" = 0 ] && echo '' || echo "_s$SEED")
VITH=${VITH_PATH:-../external/dinov3-vith16plus}
PY=${PYTHON:-python}
CK=$($PY -c "from train_lora import CKPT_DIR; print(CKPT_DIR)")
TAG=_vith_soft_fh10$SFX
L=../outputs/logs/vith_stage1$SFX.log
[ -f "$VITH/model.safetensors" ] || { echo "REFUSING TO RUN: $VITH/model.safetensors missing (copy the folder from the PC, or hf download facebook/dinov3-vith16plus-pretrain-lvd1689m --local-dir $VITH)"; exit 1; }
ls "$CK"/fullft_lr0.0001_g0.85_h1024${TAG}*.pt >/dev/null 2>&1 && { echo "REFUSING TO RUN: checkpoint tagged $TAG exists"; exit 1; }
KEEP=$(command -v caffeinate >/dev/null && echo "caffeinate -i" || echo "")
mkdir -p ../outputs/logs
echo "=== ViT-H+ stage 1 seed $SEED START $(date '+%m-%d %H:%M:%S') ===" > "$L"
$KEEP $PY -u train_full_ft.py --model-id "$VITH" --freeze-backbone --init-head "" --head-arch conv --hidden 1024 \
  --bb-lr 1e-4 --bs 4 --seed "$SEED" --epochs 10 --save-epochs 10 --augment --aug-strength blackout \
  --train-split train12k --val-split holdout --weight-power 0 --target soft --tag "$TAG" >> "$L" 2>&1
echo "=== exit $? $(date '+%m-%d %H:%M:%S') ===" >> "$L"
grep -E "taps rescaled|depth map|^ep 10 " "$L"
