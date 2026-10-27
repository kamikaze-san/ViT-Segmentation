#!/bin/bash
# Does decoder ARCHITECTURE matter, isolated from capacity and from the backbone?
#
# Backbone: vits_lora_r16, FROZEN. It is in the winning 4-member ensemble, it is
# ViT-S (442s/epoch frozen), softmax, and its target score is known (0.90063), so
# a winner swaps straight in.
#
# The CONTROL matters: freezing a jointly-trained backbone and retraining any head
# from scratch scores worse than the shipped model, because the shipped head was
# co-trained. Comparing the new decoder against 0.90063 would measure the freeze,
# not the decoder. Both heads are therefore retrained cold on the same frozen
# features, and matched on parameters (11.09M vs 11.19M).
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
L=../outputs/logs/decoder.log
: > "$L"
BB=fullft_lr0.0001_g0.85_h1024_vits_lora_r16_ep10.pt
run () { echo "=== $1 START $(date '+%m-%d %H:%M:%S') ===" >> "$L"; shift
         python -u train_full_ft.py --freeze-backbone --backbone-ckpt "$BB" --init-head "" \
           --bb-lr 1e-4 --bs 4 --seed 0 --epochs 10 --save-epochs 10 \
           --augment --aug-strength blackout --train-split train12k --val-split holdout \
           "$@" >> "$L" 2>&1
         echo "=== exit $? $(date '+%m-%d %H:%M:%S') ===" >> "$L"; }
run control_conv       --head-arch conv        --hidden 1024 --tag _dec_conv
run test_progressive   --head-arch progressive --hidden 832  --tag _dec_prog
echo "=== DECODER TEST DONE $(date) ===" >> "$L"
