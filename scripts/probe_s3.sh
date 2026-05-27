#!/usr/bin/env bash
# Probe network latency + S3 throughput from this node.
set -uo pipefail
echo "node: $(hostname)"

URL="https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/36/R/UU/2024/7/S2B_36RUU_20240701_0_L2A/B04.tif"

echo ""
echo "=== Single GET, full timing (curl) ==="
curl -s -o /dev/null -w "  dns=%{time_namelookup}s  tcp=%{time_connect}s  tls=%{time_appconnect}s  ttfb=%{time_starttransfer}s  total=%{time_total}s  size=%{size_download}B  speed=%{speed_download}B/s\n" \
  -r 0-65535 "$URL"

echo ""
echo "=== 5 sequential range GETs (connection should reuse keep-alive) ==="
for i in 1 2 3 4 5; do
  curl -s -o /dev/null -w "  req$i: ttfb=%{time_starttransfer}s  total=%{time_total}s  size=%{size_download}B\n" \
    -r 0-65535 "$URL"
done

echo ""
echo "=== 5 parallel range GETs (HTTP/2 multiplexed, --http2) ==="
{
  for i in 1 2 3 4 5; do
    (curl -s -o /dev/null -w "  par_req$i: ttfb=%{time_starttransfer}s  total=%{time_total}s\n" \
       --http2 -r 0-65535 "$URL") &
  done
  wait
} 2>&1

echo ""
echo "=== 20 parallel range GETs (test concurrency cap) ==="
start=$(date +%s.%N)
for i in $(seq 1 20); do
  (curl -s -o /dev/null --http2 -r 0-65535 "$URL") &
done
wait
end=$(date +%s.%N)
echo "  wall: $(python3 -c "print(f'{$end - $start:.2f}')")s for 20 parallel"

echo ""
echo "=== 100 parallel range GETs ==="
start=$(date +%s.%N)
for i in $(seq 1 100); do
  (curl -s -o /dev/null --http2 -r 0-65535 "$URL") &
done
wait
end=$(date +%s.%N)
echo "  wall: $(python3 -c "print(f'{$end - $start:.2f}')")s for 100 parallel"
