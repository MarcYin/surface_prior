#!/usr/bin/env bash
# Cold then warm parallel-process 5-year batch.
set -uo pipefail
BIN=/home/users/marcyin/surface_prior/surface_priors_rs/target/release/spx-build
CACHE=/tmp/spx-rust-parallel-cache
YEARS=(2020 2021 2022 2023 2024)
echo "node: $(hostname), cpus: $(nproc)"

extract() { python3 -c "import json,sys; print(f'{json.loads(sys.stdin.read())[\"timing\"][\"total\"]:.2f}')"; }

rm -rf "$CACHE" /tmp/spx-batch-*.log

echo "=== Pass 1 (cold, parallel 5) ==="
start=$(date +%s.%N)
for y in "${YEARS[@]}"; do
  $BIN --datetime "${y}-07-01/${y}-07-31" --top-k 3 --concurrency 120 \
    --no-write --disk-cache "$CACHE" \
    > /tmp/spx-batch-$y.log 2>&1 &
done
wait
end=$(date +%s.%N)
echo "  wall: $(python3 -c "print(f'{$end - $start:.2f}')")s"
for y in "${YEARS[@]}"; do
  t=$(tail -1 /tmp/spx-batch-$y.log | extract 2>/dev/null || echo FAIL)
  echo "  $y: ${t}s"
done

echo ""
echo "=== Pass 2 (warm, parallel 5) ==="
start=$(date +%s.%N)
for y in "${YEARS[@]}"; do
  $BIN --datetime "${y}-07-01/${y}-07-31" --top-k 3 --concurrency 120 \
    --no-write --disk-cache "$CACHE" \
    > /tmp/spx-batch-$y.log 2>&1 &
done
wait
end=$(date +%s.%N)
echo "  wall: $(python3 -c "print(f'{$end - $start:.2f}')")s"
for y in "${YEARS[@]}"; do
  t=$(tail -1 /tmp/spx-batch-$y.log | extract 2>/dev/null || echo FAIL)
  echo "  $y: ${t}s"
done
