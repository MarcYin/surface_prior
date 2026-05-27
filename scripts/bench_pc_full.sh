#!/usr/bin/env bash
# Production-realistic PC benchmark: parallel-5 with full GeoTIFF writes.
set -uo pipefail
BIN=/home/users/marcyin/surface_prior/surface_priors_rs/target/release/spx-build
CACHE=/tmp/spx-pc-full
YEARS=(2020 2021 2022 2023 2024)
echo "node: $(hostname), cpus: $(nproc)"

extract() { python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(f'{d[\"timing\"][\"total\"]:.2f}s  fetch={d[\"timing\"][\"fetch\"]:.2f}  compose={d[\"timing\"][\"compose\"]:.2f}  write={d[\"timing\"][\"write\"]:.2f}')"; }

for iter in 1 2 3; do
  rm -rf "$CACHE" /tmp/spx-pc-full-out-*
  echo ""
  echo "=== iter $iter: PC parallel-5 cold WITH WRITE ==="
  start=$(date +%s.%N)
  for y in "${YEARS[@]}"; do
    $BIN --datetime "${y}-07-01/${y}-07-31" --top-k 3 --concurrency 120 \
      --endpoint pc --disk-cache "$CACHE" \
      --out-dir "/tmp/spx-pc-full-out-$y" \
      > "/tmp/spx-pc-batch-$y.log" 2>&1 &
  done
  wait
  end=$(date +%s.%N)
  echo "  wall: $(python3 -c "print(f'{$end - $start:.2f}')")s"
  for y in "${YEARS[@]}"; do
    t=$(tail -1 "/tmp/spx-pc-batch-$y.log" | extract 2>/dev/null || echo FAIL)
    n=$(ls /tmp/spx-pc-full-out-$y/*.tif 2>/dev/null | wc -l)
    echo "  $y: $t  (tifs=$n)"
  done
done
