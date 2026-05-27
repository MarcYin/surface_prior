#!/usr/bin/env bash
# Compare three transport paths to the same Sentinel-2 COG byte range:
#   1. HTTPS via virtual-hosted-style URL (what spx-build uses)
#   2. HTTPS via path-style URL
#   3. AWS CLI (`aws s3api get-object`) — the "S3-native" route
#
# All three should hit the same S3 backend over HTTPS; differences are
# in the client (signing, pooling, multiplexing).
set -uo pipefail
echo "node: $(hostname)"
echo "aws cli: $(aws --version 2>&1)"

URL_VHOST="https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/36/R/UU/2024/7/S2B_36RUU_20240701_0_L2A/B04.tif"
URL_PATH="https://s3.us-west-2.amazonaws.com/sentinel-cogs/sentinel-s2-l2a-cogs/36/R/UU/2024/7/S2B_36RUU_20240701_0_L2A/B04.tif"
S3_URI="s3://sentinel-cogs/sentinel-s2-l2a-cogs/36/R/UU/2024/7/S2B_36RUU_20240701_0_L2A/B04.tif"

echo ""
echo "=== HTTPS virtual-hosted ==="
for i in 1 2 3; do
  curl -s -o /dev/null -w "  req$i  ttfb=%{time_starttransfer}s  total=%{time_total}s  size=%{size_download}B\n" \
    -r 0-65535 "$URL_VHOST"
done

echo ""
echo "=== HTTPS path-style ==="
for i in 1 2 3; do
  curl -s -o /dev/null -w "  req$i  ttfb=%{time_starttransfer}s  total=%{time_total}s  size=%{size_download}B\n" \
    -r 0-65535 "$URL_PATH"
done

echo ""
echo "=== AWS CLI (s3api get-object, --no-sign-request) ==="
for i in 1 2 3; do
  start=$(date +%s.%N)
  aws s3api get-object \
    --no-sign-request \
    --bucket sentinel-cogs \
    --key sentinel-s2-l2a-cogs/36/R/UU/2024/7/S2B_36RUU_20240701_0_L2A/B04.tif \
    --range bytes=0-65535 \
    /tmp/s3-probe-$i.bin > /dev/null 2>&1
  end=$(date +%s.%N)
  size=$(stat -c%s /tmp/s3-probe-$i.bin 2>/dev/null || echo "?")
  echo "  req$i  total=$(python3 -c "print(f'{$end - $start:.3f}')")s  size=${size}B"
done
rm -f /tmp/s3-probe-*.bin

echo ""
echo "=== 20 parallel HTTPS (vhost) GETs ==="
start=$(date +%s.%N)
for i in $(seq 1 20); do
  (curl -s -o /dev/null --http2 -r 0-65535 "$URL_VHOST") &
done; wait
end=$(date +%s.%N)
echo "  wall: $(python3 -c "print(f'{$end - $start:.2f}')")s for 20 parallel"

echo ""
echo "=== 20 parallel AWS CLI GETs ==="
start=$(date +%s.%N)
for i in $(seq 1 20); do
  (aws s3api get-object \
    --no-sign-request \
    --bucket sentinel-cogs \
    --key sentinel-s2-l2a-cogs/36/R/UU/2024/7/S2B_36RUU_20240701_0_L2A/B04.tif \
    --range bytes=0-65535 \
    /tmp/s3-probe-$i.bin > /dev/null 2>&1) &
done; wait
end=$(date +%s.%N)
echo "  wall: $(python3 -c "print(f'{$end - $start:.2f}')")s for 20 parallel"
rm -f /tmp/s3-probe-*.bin
