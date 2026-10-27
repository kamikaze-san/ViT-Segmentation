#!/bin/bash
# Waits for the LoRA rank curve, then:
#   1-2. SEED VARIANCE on the current recipe. results.json has seed variance only
#        for the OLD config (12 epochs, unaugmented). Every recent conclusion sits
#        in a +0.0005 to +0.002 band - LoRA r4 "beat" full FT by +0.0008, taps by
#        +0.0032 - and without a noise floor none of them are interpretable.
#        Identical to vits_ft10 except the shuffle seed.
#   3.   COPY-PASTE, one variable against vits_ft10: strong + blackout + copy-paste
#        vs strong + blackout. Pastes the defect's true mask shape (not its bbox)
#        with background-referenced brightness matching.
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
L=../outputs/logs/variance.log
until grep -q "ABLATIONS DONE" ../outputs/logs/ablations.log 2>/dev/null; do sleep 120; done
sleep 30
: > "$L"
run () { echo "=== $1 START $(date '+%m-%d %H:%M:%S') ===" >> "$L"; shift
         python -u train_full_ft.py "$@" >> "$L" 2>&1
         echo "=== exit $? $(date '+%m-%d %H:%M:%S') ===" >> "$L"; }

for S in 1 2; do
  run seed$S --bb-lr 1e-4 --gamma 0.85 --hidden 1024 --bs 4 --epochs 10 --save-epochs 10 \
      --augment --aug-strength blackout --train-split train12k --val-split holdout \
      --seed $S --tag _vits_ft10_s$S
done

run copypaste --bb-lr 1e-4 --gamma 0.85 --hidden 1024 --bs 4 --epochs 10 --save-epochs 10 \
    --augment --aug-strength copypaste --train-split train12k --val-split holdout \
    --seed 0 --tag _vits_copypaste

echo "=== VARIANCE+COPYPASTE DONE $(date) ===" >> "$L"
