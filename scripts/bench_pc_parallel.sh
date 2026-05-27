#!/usr/bin/env bash
set -uo pipefail
BIN=/home/users/marcyin/surface_prior/surface_priors_rs/target/release/spx-build
CACHE=/tmp/spx-pc-parallel
YEARS=(2020 2021 2022 2023 2024)
echo "node: $(hostname), cpus: $(nproc)"

extract() { python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(f'{d[\"timing\"][\"total\"]:.2f}s')"; }

for iter in 1 2 3; do
  rm -rf "$CACHE" /tmp/spx-pc-batch-*.log
  echo ""
  echo "=== iter $iter: PC parallel-5 cold ==="
  start=$(date +%s.%N)
  for y in "${YEARS[@]}"; do
    $BIN --datetime "${y}-07-01/${y}-07-31" --top-k 3 --concurrency 120 \
      --endpoint pc --no-write --disk-cache "$CACHE" \
      > /tmp/spx-pc-batch-$y.log 2>&1 &
  done
  wait
  end=$(date +%s.%N)
  echo "  wall: $(python3 -c "print(f'{$end - $start:.2f}')")s"
  for y in "${YEARS[@]}"; do
    t=$(tail -1 /tmp/spx-pc-batch-$y.log | extract 2>/dev/null || echo FAIL)
    echo "  $y: $t"
  done
done
