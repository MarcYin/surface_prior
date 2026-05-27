#!/usr/bin/env bash
# Run a back-to-back Rust benchmark suite on a compute node.
# Caller is expected to wrap this in srun --partition=debug ...
set -uo pipefail

CACHE=/tmp/spx-rust-batch-cache
BIN=/home/users/marcyin/surface_prior/surface_priors_rs/target/release/spx-build
YEARS=(2020 2021 2022 2023 2024)

extract_total() {
  python3 -c "import json,sys; print(f'{json.loads(sys.stdin.read())[\"timing\"][\"total\"]:.2f}')"
}

rm -rf "$CACHE" /tmp/spx-batch-*.log /tmp/spx-rust-out-*
echo "node: $(hostname), cpus: $(nproc), mem: $(free -h | awk 'NR==2{print $2}')"

echo "=== A: Rust sequential cold ==="
start=$(date +%s.%N)
for y in "${YEARS[@]}"; do
  t=$($BIN --datetime "${y}-07-01/${y}-07-31" --top-k 3 --concurrency 600 \
       --no-write --disk-cache "$CACHE" 2>/dev/null | tail -1 | extract_total)
  echo "  $y: ${t}s"
done
echo "  wall: $(python3 -c "import time; print(f'{time.time()-$start:.2f}')")s"

echo ""
echo "=== B: Rust sequential warm ==="
start=$(date +%s.%N)
for y in "${YEARS[@]}"; do
  t=$($BIN --datetime "${y}-07-01/${y}-07-31" --top-k 3 --concurrency 600 \
       --no-write --disk-cache "$CACHE" 2>/dev/null | tail -1 | extract_total)
  echo "  $y: ${t}s"
done
echo "  wall: $(python3 -c "import time; print(f'{time.time()-$start:.2f}')")s"

echo ""
echo "=== C: Rust parallel-process cold (xargs -P 5, fresh cache) ==="
rm -rf "$CACHE"
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
  t=$(tail -1 /tmp/spx-batch-$y.log | extract_total 2>/dev/null || echo "FAIL")
  echo "  $y: ${t}s"
done
