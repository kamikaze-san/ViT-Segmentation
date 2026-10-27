#!/bin/bash
# Sequential on one 8 GB GPU. Ordered by VALUE, not by cost: the first epoch
# measured 413s (~7 min), so the original probes-first order would have pushed
# the BCE run to ~noon. BCE is the only one of the three that yields a
# submittable model, so it goes first and the probes are bonus.
#
# Everything appends to ONE log so a single tail -f follows the whole chain
# (globbing separate per-run logs fails for files that do not exist yet).
set -u
export HF_HUB_OFFLINE=1   # AutoImageProcessor makes a network HEAD request even when
                          # weights are cached; an SSL failure killed both probes at 05:19
cd "$(dirname "$0")"
L=../outputs/logs/overnight_b.log

run () {   # run <label> <args...>
  echo "=== $1 START $(date +%H:%M:%S) ===" >> "$L"
  shift
  python -u train_full_ft.py "$@" >> "$L" 2>&1
  echo "=== exit $? $(date +%H:%M:%S) ===" >> "$L"
}


# 2+3. Frozen-backbone probe: are ViT-B's PRETRAINED features better, or is
#      capacity just capacity? Both start from a COLD head, since the default
#      warm-start head was trained on ViT-S features and keeping it for ViT-S
#      only would compare the head's history rather than the backbones.
run probe_vits16 \
  --model-id facebook/dinov3-vits16-pretrain-lvd1689m --freeze-backbone \
  --init-head "" --epochs 10 --bb-lr 1e-4 --gamma 0.85 --hidden 1024 --seed 0 \
  --train-split train --val-split val --tag _probe_vits16

run probe_vitb16 \
  --model-id facebook/dinov3-vitb16-pretrain-lvd1689m --freeze-backbone \
  --init-head "" --epochs 6 --bb-lr 1e-4 --gamma 0.85 --hidden 1024 --seed 0 \
  --train-split train --val-split val --tag _probe_vitb16

echo "=== ALL DONE $(date) ===" >> "$L"
