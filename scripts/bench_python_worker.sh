#!/usr/bin/env bash
# Python comparison on a compute node — same AOI / years / pattern as Rust.
set -uo pipefail
cd /home/users/marcyin/surface_prior

CACHE=/tmp/spx-py-worker-cache
rm -rf "$CACHE"
echo "node: $(hostname), cpus: $(nproc)"

PYTHONPATH=src python scripts/benchmark_multi_year.py \
  --years 2020 2021 2022 2023 2024 \
  --target-month 7 --target-only \
  --passes 2 --cache-dir "$CACHE" 2>&1 | grep -E "year=|wall=" | head -25
