#!/bin/bash
# Frozen-encoder comparison (E29 extended): one process per encoder (a CUDA error cannot poison the next one),
# new encoders first, then the three E29 encoders again so every encoder also has test-set scores.
cd "$(dirname "$0")"
L=../outputs/logs/encoder_probe.log
export HF_HUB_OFFLINE=1
for e in cradio_h dinov3h qwen38 cradio_so dinov3 dinov3l qwenvit; do
  echo "=== $e START $(date '+%m-%d %H:%M:%S') ===" >> "$L"
  python -u encoder_probe.py --encoder $e >> "$L" 2>&1
  echo "=== $e exit $? $(date '+%m-%d %H:%M:%S') ===" >> "$L"
done
python -u encoder_test_compare.py >> "$L" 2>&1
echo "=== ALL ENCODERS DONE $(date '+%m-%d %H:%M:%S') ===" >> "$L"
