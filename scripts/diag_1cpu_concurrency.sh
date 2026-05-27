#!/usr/bin/env bash
# Sweep concurrency on 1 CPU. If lower concurrency reduces errors and
# total time, the bottleneck is reactor starvation under load, not
# CPU-bound decode.
set -uo pipefail
BIN=/home/users/marcyin/surface_prior/surface_priors_rs/target/release/spx-build
CACHE=/tmp/spx-diag-conc
echo "node: $(hostname), cpus: $(nproc)"

for c in 600 200 50 20 10; do
  rm -rf "$CACHE"
  echo "=== concurrency=$c ==="
  out=$($BIN --datetime 2024-07-01/2024-07-31 --top-k 3 --concurrency $c \
        --no-write --disk-cache "$CACHE" 2>&1)
  errs=$(echo "$out" | grep -c "fetch error")
  json=$(echo "$out" | tail -1)
  total=$(echo "$json" | python3 -c "import json,sys; print(f'{json.loads(sys.stdin.read())[\"timing\"][\"total\"]:.2f}')" 2>/dev/null || echo "FAIL")
  obs=$(echo "$json" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['observations'])" 2>/dev/null || echo "?")
  echo "  total=${total}s  observations=${obs}  errors=${errs}"
done
