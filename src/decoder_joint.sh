#!/bin/bash
# Decoder architecture under JOINT training - the comparison that actually matters.
#
# The frozen test gave progressive 0.6920 vs conv 0.6802 locally, but on target
# dec_prog scored 0.89623 against vits_lora_r16's 0.90063 - confounded, because
# dec_prog was frozen-backbone + cold head while 0.90063 was joint + warm head.
#
# Both runs here are JOINT (LoRA r16 all-6, the vits_lora_r16 recipe) and both use
# a COLD head, because the warm start is conv-shaped and progressive cannot load
# it. That makes decoder architecture the only difference between them. Params are
# matched: conv hidden 1024 = 11.09M, progressive hidden 832 = 11.19M.
#
# Neither is directly comparable to vits_lora_r16 (which had the warm head) - the
# comparison is A vs B.
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
L=../outputs/logs/decoder_joint.log
: > "$L"
ALL6="q_proj k_proj v_proj o_proj up_proj down_proj"
run () { echo "=== $1 START $(date '+%m-%d %H:%M:%S') ===" >> "$L"; shift
         python -u train_full_ft.py --lora-r 16 --lora-alpha 32 --lora-targets $ALL6 \
           --init-head "" --bb-lr 1e-4 --gamma 0.85 --bs 4 --seed 0 \
           --epochs 10 --save-epochs 10 --augment --aug-strength blackout \
           --train-split train12k --val-split holdout "$@" >> "$L" 2>&1
         echo "=== exit $? $(date '+%m-%d %H:%M:%S') ===" >> "$L"; }
run joint_conv         --head-arch conv        --hidden 1024 --tag _jnt_conv
run joint_progressive  --head-arch progressive --hidden 832  --tag _jnt_prog
echo "=== DECODER JOINT DONE $(date) ===" >> "$L"
