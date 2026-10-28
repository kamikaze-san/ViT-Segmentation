#!/bin/bash
# E31 follow-up checks on C-RADIOv4-SO400M. Written for the Mac; also runs on the PC once its GPU is free.
#   fix     check_cradio_train_mode.py: the position fix is merged and works on this machine (must print PASS)
#   layout  job 1, cradio_layout_probe.py: which layout of C-RADIO's position grid suits 256x1600 strips (frozen probes)
#   drift   job 2: short stage-2 runs (1 epoch at constant peak LR from the stage-1 head, first N_TRAIN labelled train12k
#           strips, no pseudo labels, full 500-strip holdout before and after) for C-RADIO at each LR in CRADIO_LRS and
#           for ViT-B at 1e-4 as the reference (only if its stage-1 head is present), then feature_drift.py
# Usage: bash radio_checks.sh [all|fix|layout|drift]          (default all; drift always runs fix first)
#   long runs: nohup bash radio_checks.sh all > ../outputs/logs/radio_checks.log 2>&1 &
# Env: CRADIO_PATH  C-RADIO folder (default ../external/C-RADIOv4-SO400M)
#      N_TRAIN      strips per short run (default 1200 = 300 steps at bs 4)
#      CRADIO_LRS   default "1e-4 3e-5 1e-5"
#      VITB_ARM     auto (run if the ViT-B head is in CKPT_DIR) or 0 (skip)
#      EXTRA_ARGS   extra train_full_ft.py flags, e.g. a device flag if this machine's copy needs one
# Arms whose checkpoint and epoch-1 log line already exist are reused, so a rerun resumes.
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1
WHAT=${1:-all}
CRADIO=${CRADIO_PATH:-../external/C-RADIOv4-SO400M}
N=${N_TRAIN:-1200}
LRS=${CRADIO_LRS:-"1e-4 3e-5 1e-5"}
VITB=${VITB_ARM:-auto}
EXTRA=${EXTRA_ARGS:-}
LOGS=../outputs/logs
mkdir -p "$LOGS"
CK=$(python -c "from train_lora import CKPT_DIR; print(CKPT_DIR)")
CRADIO_HEAD=fullft_lr0.0001_g0.85_h1024_cradio_so_fh10_ep10.pt
VITB_HEAD=fullft_lr0.0001_g0.85_h1024_vitb_soft_fh10_ep10.pt
COMMON="--lora-r 16 --lora-alpha 32 --head-arch conv --hidden 1024 --gamma 0.85 --bs 4 --seed 0 --epochs 1
  --max-train $N --augment --aug-strength blackout --train-split train12k --val-split holdout --weight-power 0 --target soft"
stamp () { date '+%m-%d %H:%M:%S'; }

check_fix () {
  grep -q "ADAPTER.train_mode_fix()" train_full_ft.py && grep -q "def train_mode_fix" backbones.py ||
    { echo "REFUSING TO RUN: the C-RADIO position fix is not in train_full_ft.py / backbones.py (merge it first)"; exit 1; }
  [ -d "$CRADIO" ] || { echo "REFUSING TO RUN: C-RADIO folder $CRADIO not found (set CRADIO_PATH)"; exit 1; }
  echo "=== fix check $(stamp) ==="
  python -u check_cradio_train_mode.py "$CRADIO" 2>&1 | tee "$LOGS/radio_checks_fix.log" | grep -v -i "warn"
  grep -q "^PASS" "$LOGS/radio_checks_fix.log" || { echo "REFUSING TO CONTINUE: fix check did not PASS"; exit 1; }
}

run_layout () {
  echo "=== layout probe START $(stamp) ==="
  python -u cradio_layout_probe.py --backbone-path "$CRADIO" > "$LOGS/radio_checks_layout.log" 2>&1
  echo "=== layout probe exit $? $(stamp) ==="
  tail -8 "$LOGS/radio_checks_layout.log"
}

run_arm () {                                          # run_arm <tag> <train_full_ft args...>
  local tag=$1; shift
  local log="$LOGS/lrdrift${tag}.log"
  if ls "$CK"/fullft_lr*"${tag}".pt >/dev/null 2>&1 && grep -q "^ep  1 " "$log" 2>/dev/null; then
    echo "arm $tag: already done, reusing"; return 0
  fi
  echo "=== arm $tag START $(stamp) ===" | tee "$log"
  python -u train_full_ft.py "$@" $COMMON $EXTRA --tag "$tag" >> "$log" 2>&1
  local rc=$?
  echo "=== arm $tag exit $rc $(stamp) ===" | tee -a "$log"
  grep -E "epoch 0 \(|^ep  1 |Error" "$log" | tail -3
  return $rc
}

run_drift () {
  [ -f "$CK/$CRADIO_HEAD" ] || { echo "REFUSING TO RUN: $CRADIO_HEAD not in $CK"; exit 1; }
  local arms=()
  for lr in $LRS; do
    run_arm "_lrdrift_cradio_$lr" --backbone cradio --backbone-path "$CRADIO" --init-head $CRADIO_HEAD --bb-lr "$lr" &&
      arms+=(--arm "cradio:$lr:_lrdrift_cradio_$lr")
  done
  if [ "$VITB" != 0 ] && [ -f "$CK/$VITB_HEAD" ]; then
    run_arm _lrdrift_vitb_1e-4 --model-id facebook/dinov3-vitb16-pretrain-lvd1689m \
      --lora-targets q_proj k_proj v_proj o_proj up_proj down_proj --init-head $VITB_HEAD --bb-lr 1e-4 &&
      arms+=(--arm "dinov3:1e-4:_lrdrift_vitb_1e-4")
  else
    echo "ViT-B reference arm skipped ($VITB_HEAD not in $CK, or VITB_ARM=0)"
  fi
  [ ${#arms[@]} -gt 0 ] || { echo "no arm finished: nothing to analyse"; exit 1; }
  echo "=== feature drift START $(stamp) ==="
  python -u feature_drift.py "${arms[@]}" 2>&1 | tee "$LOGS/radio_checks_drift.log" | grep -v -i "warn"
  echo "=== feature drift exit $? $(stamp) ==="
}

echo "=== radio_checks $WHAT START $(stamp) | C-RADIO $CRADIO | CKPT_DIR $CK | N_TRAIN $N | LRs $LRS | extra '$EXTRA' ==="
case "$WHAT" in
  fix)    check_fix ;;
  layout) check_fix; run_layout ;;
  drift)  check_fix; run_drift ;;
  all)    check_fix; run_layout; run_drift ;;
  *)      echo "usage: radio_checks.sh [all|fix|layout|drift]"; exit 2 ;;
esac
echo "=== radio_checks $WHAT DONE $(stamp) ==="
