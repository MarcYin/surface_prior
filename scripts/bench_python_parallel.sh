#!/usr/bin/env bash
# Python parallel-process 5-year benchmark on a worker.
set -uo pipefail
cd /home/users/marcyin/surface_prior
CACHE=/tmp/spx-py-parallel-cache
YEARS=(2020 2021 2022 2023 2024)
echo "node: $(hostname), cpus: $(nproc)"

# Use the existing parallel-years flag inside the Python script which
# spawns ThreadPoolExecutor of years; that mirrors xargs-P but in one
# process. (Python's process-level parallel would need separate
# invocations; the script's --parallel-years is the closest equivalent.)
rm -rf "$CACHE"
echo "=== Python --parallel-years=5 (cold + warm) ==="
PYTHONPATH=src python scripts/benchmark_multi_year.py \
  --years "${YEARS[@]}" --target-month 7 --target-only \
  --passes 2 --parallel-years 5 --cache-dir "$CACHE" 2>&1 | grep -E "wall="
