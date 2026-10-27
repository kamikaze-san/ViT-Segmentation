#!/usr/bin/env bash
# Cheap weight_power calibration for the SPATIAL cache.
#
# The spatial cache keeps ALL 1600 patches/image, so its class balance is
# ~9x more skewed than the old flat cache (bg:c2 2631:1 vs 282:1). The
# weight_power tuned on the flat cache (-0.5) is therefore badly wrong here -
# it collapsed c2 to 0.0000 in increment A.
#
# Analytic equivalent works out to ~ -0.074, but that only reproduces what the
# flat cache did; the true optimum on this distribution may differ. So: rank
# candidates on a 2000-image subset with few epochs (fast), then run the real
# increments on the full set at the winner.
set -u
cd "$(dirname "$0")"
LOG_DIR="../outputs/increment_logs"
mkdir -p "$LOG_DIR"

for wp in 0.45 0.60 0.75 0.90; do
  echo "### calibrating weight_power=$wp ###"
  python -u train_spatial.py --head mlp --upscale 1 --weight-power "$wp" \
      --epochs 5 --max-train 2000 --tag "_cal$wp" > "$LOG_DIR/cal_wp$wp.log" 2>&1
  echo "  exit $?"
  grep -E "class weights|BEST val mean Dice" "$LOG_DIR/cal_wp$wp.log" || true
  tail -3 "$LOG_DIR/cal_wp$wp.log" | grep "^ep " || true
done

echo ""
echo "============ CALIBRATION SUMMARY ============"
for wp in 0.45 0.60 0.75 0.90; do
  printf "wp=%-6s %s\n" "$wp" "$(grep 'BEST val mean Dice' "$LOG_DIR/cal_wp$wp.log" 2>/dev/null | tail -1)"
done
echo "CALIBRATION DONE"
