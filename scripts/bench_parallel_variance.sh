#!/usr/bin/env bash
# Re-run parallel-5 cold a few times to measure run-to-run variance.
# Each iteration starts with a fresh disk cache so all 5 builds are
# genuinely cold; CDN state varies between iterations and that's
# the dominant source of spread.
set -uo pipefail
BIN=/home/users/marcyin/surface_prior/surface_priors_rs/target/release/spx-build
CACHE=/tmp/spx-pvariance
YEARS=(2020 2021 2022 2023 2024)
echo "node: $(hostname), cpus: $(nproc)"

for iter in 1 2 3 4; do
  rm -rf "$CACHE" /tmp/spx-batch-*.log
  echo ""
  echo "=== Iteration $iter ==="
  start=$(date +%s.%N)
  for y in "${YEARS[@]}"; do
    $BIN --datetime "${y}-07-01/${y}-07-31" --top-k 3 --concurrency 120 \
      --no-write --disk-cache "$CACHE" \
      > /tmp/spx-batch-$y.log 2>&1 &
  done
  wait
  end=$(date +%s.%N)
  wall=$(python3 -c "print(f'{$end - $start:.2f}')")
  echo "  wall: ${wall}s"
  for y in "${YEARS[@]}"; do
    t=$(tail -1 /tmp/spx-batch-$y.log | python3 -c "
import json,sys
try:
    d=json.loads(sys.stdin.read())
    print(f'{d[\"timing\"][\"total\"]:.2f}s  fetch={d[\"timing\"][\"fetch\"]:.2f}  scout={d[\"timing\"][\"scout\"]:.2f}')
except Exception as e: print(f'FAIL: {e}')
" 2>/dev/null)
    echo "  $y: $t"
  done
done
