#!/bin/bash
# Sequential, not parallel: one 8 GB GPU. Each run must finish before the next.
set -u
cd "$(dirname "$0")"
LOG=../outputs/logs
STAMP=$(date +%Y%m%d_%H%M)

# ---- 1+2. Frozen-backbone probe: does ViT-B give BETTER PRETRAINED FEATURES? --
# Gate for a ~24 hr full ViT-B fine-tune. Both probes start from a COLD head
# (--init-head "") because the default warm-start head was trained on ViT-S
# features; letting ViT-S keep it and forcing ViT-B to start cold would compare
# the head's history, not the backbones.
for MID in facebook/dinov3-vits16-pretrain-lvd1689m facebook/dinov3-vitb16-pretrain-lvd1689m; do
  SHORT=$(echo "$MID" | sed 's/.*dinov3-//; s/-pretrain.*//')
  echo "=== probe $SHORT  $(date) ==="
  python train_full_ft.py \
    --model-id "$MID" --freeze-backbone --init-head "" \
    --epochs 10 --bb-lr 1e-4 --gamma 0.85 --hidden 1024 --seed 0 \
    --train-split train --val-split val \
    --tag "_probe_${SHORT}" > "$LOG/probe_${SHORT}_${STAMP}.log" 2>&1
  echo "    exit $?  $(date)"
done

# ---- 3. sigmoid + BCE, everything else held at the blackout recipe ----------
# Only the loss changes vs fullft_lr0.0001_g0.85_h1024_blackout_ep16.pt, so a
# difference is attributable to the decision structure and nothing else.
# Validated on holdout, which is clean for train12k models.
echo "=== bce_dice  $(date) ==="
python train_full_ft.py \
  --loss bce_dice --pos-weight-power 0.15 \
  --bb-lr 1e-4 --gamma 0.85 --hidden 1024 --epochs 16 --seed 0 \
  --augment --aug-strength blackout \
  --train-split train12k --val-split holdout --save-epochs 16 \
  --tag _bce_blackout > "$LOG/bce_blackout_${STAMP}.log" 2>&1
echo "    exit $?  $(date)"
echo "=== ALL DONE $(date) ==="
