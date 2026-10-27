#!/bin/bash
# One-variable ablations. Waits for the ConvNeXt run to finish first.
#
# A is a missing CONTROL: there is no ViT-S full-FT model at 10 epochs on
# 12k+blackout with the standard taps, which is what breaks three comparisons at
# once (vs LoRA, vs alt-taps, vs the 16-epoch blackout).
#
# B is a rank curve at WIDE coverage on the current recipe. Every all-6 LoRA model
# we own is r=16; the r4/r8/r16 curve we have is q/k/v-only on the old 8.8k
# unaugmented recipe. alpha is set to 2r so the effective update scale alpha/r is
# held constant and rank is isolated - the convention the historical sweep used
# and that I failed to follow in the ViT-L sweep.
#
# Head init is left at the DEFAULT warm start for every run, matching blackout /
# 12k_ep16 / bce16, so these stay comparable to the existing ViT-S models too.
# No gradient checkpointing: ViT-S has no memory pressure and skipping it is ~30%
# faster.
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
L=../outputs/logs/ablations.log
ALL6="q_proj k_proj v_proj o_proj up_proj down_proj"

until grep -q "BEST val mean Dice" ../outputs/logs/convnext.log 2>/dev/null; do sleep 60; done
sleep 30
: > "$L"
echo "=== ConvNeXt finished; ablations starting $(date) ===" >> "$L"

common="--bb-lr 1e-4 --gamma 0.85 --hidden 1024 --bs 4 --seed 0 --epochs 10 \
        --save-epochs 10 --augment --aug-strength blackout \
        --train-split train12k --val-split holdout"

run () { echo "=== $1 START $(date '+%m-%d %H:%M:%S') ===" >> "$L"; shift
         python -u train_full_ft.py $common "$@" >> "$L" 2>&1
         echo "=== exit $? $(date '+%m-%d %H:%M:%S') ===" >> "$L"; }

# A. the control
run vits_ft10 --tag _vits_ft10

# B. rank curve at all-6 coverage, alpha = 2r
for R in 4 8 16 32; do
  run vits_lora_r$R --lora-r $R --lora-alpha $((2*R)) --lora-targets $ALL6 --tag _vits_lora_r$R
done
echo "=== ABLATIONS DONE $(date) ===" >> "$L"
