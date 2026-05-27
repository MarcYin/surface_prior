#!/usr/bin/env bash
# Same range-GET via boto3 (anonymous S3 client) vs requests (HTTPS).
# Both should bottom-out at HTTPS-over-TLS to the same S3 endpoint;
# only the client-side overhead differs.
set -uo pipefail
echo "node: $(hostname)"

cd /home/users/marcyin/surface_prior
PYTHONPATH=src python3 - <<'PY'
import time, os
import urllib.request

URL_VHOST = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/36/R/UU/2024/7/S2B_36RUU_20240701_0_L2A/B04.tif"
BUCKET = "sentinel-cogs"
KEY = "sentinel-s2-l2a-cogs/36/R/UU/2024/7/S2B_36RUU_20240701_0_L2A/B04.tif"

# --- 1. requests (HTTPS) ---
try:
    import requests
    s = requests.Session()
    print("=== requests (HTTPS) ===")
    for i in range(3):
        t0 = time.perf_counter()
        r = s.get(URL_VHOST, headers={"Range": "bytes=0-65535"})
        r.raise_for_status()
        _ = r.content
        print(f"  req{i+1}  total={time.perf_counter()-t0:.3f}s  size={len(r.content)}B")
except Exception as e:
    print(f"  requests skipped: {e}")

# --- 2. boto3 (s3:// via AWS SDK, anonymous) ---
try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    s3 = boto3.client("s3", region_name="us-west-2",
                      config=Config(signature_version=UNSIGNED, s3={"addressing_style": "virtual"}))
    print("\n=== boto3 (s3, --no-sign-request) ===")
    for i in range(3):
        t0 = time.perf_counter()
        resp = s3.get_object(Bucket=BUCKET, Key=KEY, Range="bytes=0-65535")
        body = resp["Body"].read()
        print(f"  req{i+1}  total={time.perf_counter()-t0:.3f}s  size={len(body)}B")
except Exception as e:
    print(f"  boto3 skipped: {e}")

# --- 3. 20 parallel via boto3 ---
try:
    import boto3, concurrent.futures
    from botocore import UNSIGNED
    from botocore.config import Config
    s3 = boto3.client("s3", region_name="us-west-2",
                      config=Config(signature_version=UNSIGNED,
                                    s3={"addressing_style": "virtual"},
                                    max_pool_connections=64))
    def one():
        resp = s3.get_object(Bucket=BUCKET, Key=KEY, Range="bytes=0-65535")
        return len(resp["Body"].read())
    print("\n=== 20 parallel boto3 GETs ===")
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
        sizes = list(pool.map(lambda _: one(), range(20)))
    print(f"  wall: {time.perf_counter()-t0:.2f}s  total_bytes={sum(sizes)}")
except Exception as e:
    print(f"  boto3 parallel skipped: {e}")

# --- 4. 20 parallel via requests ---
try:
    import requests, concurrent.futures
    sess = requests.Session()
    sess.mount("https://", requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64))
    def one():
        r = sess.get(URL_VHOST, headers={"Range": "bytes=0-65535"})
        return len(r.content)
    print("\n=== 20 parallel requests HTTPS GETs ===")
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
        sizes = list(pool.map(lambda _: one(), range(20)))
    print(f"  wall: {time.perf_counter()-t0:.2f}s  total_bytes={sum(sizes)}")
except Exception as e:
    print(f"  requests parallel skipped: {e}")
PY
