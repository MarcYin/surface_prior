#!/usr/bin/env bash
# Compare Planetary Computer vs Element84 from this worker node.
set -uo pipefail
BIN=/home/users/marcyin/surface_prior/surface_priors_rs/target/release/spx-build
YEARS=(2020 2021 2022 2023 2024)
echo "node: $(hostname), cpus: $(nproc)"

extract() { python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(f'{d[\"timing\"][\"total\"]:.2f}s  fetch={d[\"timing\"][\"fetch\"]:.2f}  scout={d[\"timing\"][\"scout\"]:.2f}  list={d[\"timing\"][\"list_scenes\"]:.2f}')"; }

for ep in pc earth-search; do
  CACHE=/tmp/spx-ep-$ep
  rm -rf "$CACHE"
  echo ""
  echo "============================================"
  echo "  Endpoint: $ep"
  echo "============================================"
  echo "--- cold (5 sequential builds, fresh cache) ---"
  start=$(date +%s.%N)
  for y in "${YEARS[@]}"; do
    out=$($BIN --datetime "${y}-07-01/${y}-07-31" --top-k 3 --concurrency 600 \
          --endpoint "$ep" --no-write --disk-cache "$CACHE" 2>/dev/null | tail -1)
    t=$(echo "$out" | extract 2>/dev/null || echo "FAIL")
    echo "  $y: $t"
  done
  end=$(date +%s.%N)
  echo "  wall: $(python3 -c "print(f'{$end - $start:.2f}')")s"

  echo "--- warm (same cache, 5 builds) ---"
  start=$(date +%s.%N)
  for y in "${YEARS[@]}"; do
    out=$($BIN --datetime "${y}-07-01/${y}-07-31" --top-k 3 --concurrency 600 \
          --endpoint "$ep" --no-write --disk-cache "$CACHE" 2>/dev/null | tail -1)
    t=$(echo "$out" | extract 2>/dev/null || echo "FAIL")
    echo "  $y: $t"
  done
  end=$(date +%s.%N)
  echo "  wall: $(python3 -c "print(f'{$end - $start:.2f}')")s"
done
