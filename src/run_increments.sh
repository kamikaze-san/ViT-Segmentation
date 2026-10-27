#!/usr/bin/env bash
# Overnight increment chain.
#
# Every step is compared on VAL only. The TEST split is deliberately never
# touched here - one clean test evaluation happens afterwards, on the winner.
#
# Ablation is designed so each step changes exactly one thing:
#   A  mlp      up1        per-patch baseline (reproduces the B3 head)
#   A1 mlp_deep up1        + one extra layer, still 1x1  -> DEPTH control
#   A2 conv     up1        + 3x3 middle kernel           -> SPATIAL CONTEXT
#   B  conv     up4        + PixelShuffle x4             -> OUTPUT RESOLUTION
#   C  conv     up4 +dice  + soft-Dice loss term         -> LOSS FUNCTION
#
# weight_power=0.30 was calibrated empirically on this cache (2000-image
# subset, 5 epochs). The spatial cache keeps ALL 1600 patches/image so it is
# ~9x more imbalanced than the old flat cache (bg:c2 2631:1 vs 282:1); the
# flat cache's -0.5 is badly wrong here and collapsed c2 to zero. Measured
# curve: -0.15:0.314  -0.05:0.337  0.05:0.371  0.15:0.454  0.30:0.535
#         0.45:0.481  0.60:0.413  0.75:0.343   -> clean interior peak at 0.30
set -u
cd "$(dirname "$0")"
LOG_DIR="../outputs/increment_logs"
mkdir -p "$LOG_DIR"

run () {
  local name="$1"; shift
  echo "### INCREMENT $name :: $* ###"
  python -u train_spatial.py "$@" --tag "_$name" > "$LOG_DIR/$name.log" 2>&1
  echo "### $name finished, exit $? ###"
  grep -E "trainable params|BEST val mean Dice" "$LOG_DIR/$name.log" || true
}

if [ ! -f "../data/processed/spatial_cache/train_order.csv" ]; then
  echo "### Caching train spatial features (43 GB, ~65 min) ###"
  python -u cache_spatial.py train > "$LOG_DIR/cache_train.log" 2>&1
  echo "### cache exit $? ###"
  tail -2 "$LOG_DIR/cache_train.log"
fi

run A  --head mlp      --upscale 1 --weight-power 0.30
run A1 --head mlp_deep --upscale 1 --weight-power 0.30
run A2 --head conv     --upscale 1 --weight-power 0.30
run B  --head conv     --upscale 4 --weight-power 0.30
run C  --head conv     --upscale 4 --weight-power 0.30 --dice 1.0

echo ""
echo "================ SUMMARY (val mean Dice c1-4) ================"
for f in "$LOG_DIR"/A.log "$LOG_DIR"/A1.log "$LOG_DIR"/A2.log "$LOG_DIR"/B.log "$LOG_DIR"/C.log; do
  [ -f "$f" ] && printf "%-6s %s\n" "$(basename "$f" .log)" "$(grep 'BEST val mean Dice' "$f" | tail -1)"
done
echo "ALL DONE"
