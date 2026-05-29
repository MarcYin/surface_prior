//! Minimal COG (TIFF) parser focused on Sentinel-2 L2A on Element84.
//!
//! S2 L2A COGs in `sentinel-cogs.s3.us-west-2.amazonaws.com` share a
//! consistent profile:
//!   - little-endian classic TIFF (not BigTIFF)
//!   - DEFLATE compression (tag 259 == 8)
//!   - tiled (TileWidth/TileLength = 1024 typical, sometimes 256)
//!   - 4 overviews (factor 2/4/8/16) addressed via SubIFDs
//!   - dtype uint16 for SR bands, uint8 for SCL
//!
//! We deliberately parse only the tags we need; everything else is
//! ignored. The reader is HTTP-backed: a single 64 KiB range-GET at
//! byte 0 typically covers header + all IFD entries + the SubIFD chain
//! for these COGs, so opening a COG costs one HTTP round-trip.

use anyhow::{anyhow, Context, Result};
use byteorder::{ByteOrder, LittleEndian};
use bytes::Bytes;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

const HEADER_RANGE: u64 = 64 * 1024;

const TAG_IMAGE_WIDTH: u16 = 256;
const TAG_IMAGE_LENGTH: u16 = 257;
const TAG_BITS_PER_SAMPLE: u16 = 258;
const TAG_COMPRESSION: u16 = 259;
const TAG_SAMPLES_PER_PIXEL: u16 = 277;
const TAG_TILE_WIDTH: u16 = 322;
const TAG_TILE_LENGTH: u16 = 323;
const TAG_TILE_OFFSETS: u16 = 324;
const TAG_TILE_BYTE_COUNTS: u16 = 325;
const TAG_SUB_IFDS: u16 = 330;
const TAG_SAMPLE_FORMAT: u16 = 339;
const TAG_GEO_TIFF_KEYS: u16 = 34735;
const TAG_PIXEL_SCALE: u16 = 33550;
const TAG_TIE_POINT: u16 = 33922;
const TAG_NODATA: u16 = 42113;
const TAG_PREDICTOR: u16 = 317;
const TAG_PLANAR_CONFIG: u16 = 284;

const COMPRESSION_NONE: u32 = 1;
const COMPRESSION_DEFLATE: u32 = 8;
const COMPRESSION_ADOBE_DEFLATE: u32 = 32946;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SampleFormat {
    UInt8,
    UInt16,
    /// Signed 16-bit. HLS surface reflectance is stored this way (fill
    /// -9999, scale 1e-4); the bytes are decoded identically to UInt16,
    /// but callers must reinterpret as `i16` and clamp negatives/fill.
    Int16,
}

impl SampleFormat {
    pub fn bytes(self) -> usize {
        match self {
            SampleFormat::UInt8 => 1,
            SampleFormat::UInt16 | SampleFormat::Int16 => 2,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OverviewLevel {
    pub width: u32,
    pub height: u32,
    pub tile_width: u32,
    pub tile_height: u32,
    pub tile_offsets: Vec<u64>,
    pub tile_byte_counts: Vec<u64>,
    pub compression: u32,
    /// 1 = none, 2 = horizontal differencing. Anything else is unsupported.
    pub predictor: u16,
    pub bytes_per_sample: u32,
}

impl OverviewLevel {
    pub fn tiles_x(&self) -> u32 {
        (self.width + self.tile_width - 1) / self.tile_width
    }
    pub fn tiles_y(&self) -> u32 {
        (self.height + self.tile_height - 1) / self.tile_height
    }
    /// Pixel-space window → indices of tiles that intersect it.
    pub fn tiles_for_window(&self, win: PixelWindow) -> Vec<TileRequest> {
        let mut tiles = Vec::new();
        let x0 = win.col_off / self.tile_width;
        let y0 = win.row_off / self.tile_height;
        let x1 = ((win.col_off + win.width).saturating_sub(1)) / self.tile_width;
        let y1 = ((win.row_off + win.height).saturating_sub(1)) / self.tile_height;
        for ty in y0..=y1 {
            for tx in x0..=x1 {
                let idx = (ty * self.tiles_x() + tx) as usize;
                if idx < self.tile_offsets.len() {
                    tiles.push(TileRequest {
                        tile_x: tx,
                        tile_y: ty,
                        index: idx,
                        offset: self.tile_offsets[idx],
                        byte_count: self.tile_byte_counts[idx],
                    });
                }
            }
        }
        tiles
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PixelWindow {
    pub col_off: u32,
    pub row_off: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct TileRequest {
    pub tile_x: u32,
    pub tile_y: u32,
    pub index: usize,
    pub offset: u64,
    pub byte_count: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CogProfile {
    pub url: String,
    pub width: u32,
    pub height: u32,
    pub sample_format: SampleFormat,
    pub pixel_scale: Option<[f64; 3]>,
    pub tie_point: Option<[f64; 6]>,
    pub nodata: Option<f64>,
    /// Index 0 = full resolution; later entries are progressively coarser overviews.
    pub levels: Vec<OverviewLevel>,
}

impl CogProfile {
    /// Choose the overview level with the smallest downsample factor that still
    /// produces at most `target_max_dim` pixels along its longer axis after
    /// covering the original image. In practice this picks the closest overview
    /// to the requested output resolution.
    pub fn level_for_decimation(&self, decimation: f64) -> &OverviewLevel {
        let target_width = (self.width as f64 / decimation).max(1.0);
        let mut best_idx = 0;
        let mut best_ratio = f64::INFINITY;
        for (i, lvl) in self.levels.iter().enumerate() {
            let lvl_ratio = self.width as f64 / lvl.width as f64;
            // We want the coarsest level that is still finer than (or equal to)
            // the requested decimation. Smaller pixels per unit area mean
            // less post-decode resampling.
            if lvl_ratio <= decimation {
                let dist = (decimation - lvl_ratio).abs();
                if dist < best_ratio {
                    best_ratio = dist;
                    best_idx = i;
                }
            }
        }
        // Fallback: if no level is finer than target, the coarsest available wins.
        if best_ratio.is_infinite() {
            best_idx = self.levels.len() - 1;
        }
        let _ = target_width;
        &self.levels[best_idx]
    }
}

/// Process-wide cache of parsed COG profiles, keyed by URL path
/// (no query string). Eliminates duplicate header GETs across phases
/// (e.g. scout opens SCL coarse, fetch_quality opens it full-res; both
/// share the parsed profile). Keyed by path-only so SAS-token rotation
/// doesn't bust the cache.
fn cog_profile_cache() -> &'static dashmap::DashMap<String, std::sync::Arc<CogProfile>> {
    static CACHE: std::sync::OnceLock<dashmap::DashMap<String, std::sync::Arc<CogProfile>>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(dashmap::DashMap::new)
}

fn url_path_key(url: &str) -> &str {
    url.split_once('?').map(|(p, _)| p).unwrap_or(url)
}

/// Process-global disk cache backend for parsed COG headers. Set by
/// the pipeline's bootstrap (run_build / run_build_periods) when the
/// caller passes a `disk_cache` directory; null otherwise.
fn cog_disk_cache_slot() -> &'static parking_lot::RwLock<Option<crate::disk_cache::DiskCache>> {
    static SLOT: std::sync::OnceLock<
        parking_lot::RwLock<Option<crate::disk_cache::DiskCache>>,
    > = std::sync::OnceLock::new();
    SLOT.get_or_init(|| parking_lot::RwLock::new(None))
}

/// Install (or remove) the disk-cache backend that `open_cog` should
/// consult before issuing a header fetch. Idempotent; calling with
/// `None` disables disk persistence (in-memory cache still works).
pub fn set_cog_disk_cache(cache: Option<crate::disk_cache::DiskCache>) {
    *cog_disk_cache_slot().write() = cache;
}

/// Open a COG over HTTP, returning the parsed profile. Three-tier
/// cache: in-memory dashmap → disk (if configured) → HTTP fetch.
pub async fn open_cog(http: &reqwest::Client, url: &str) -> Result<std::sync::Arc<CogProfile>> {
    let cache = cog_profile_cache();
    let key = url_path_key(url).to_string();
    if let Some(profile) = cache.get(&key) {
        return Ok(profile.clone());
    }
    // Try the on-disk cache before going to the network.
    if let Some(dc) = cog_disk_cache_slot().read().clone() {
        let disk_key = dc.cog_profile_key(url);
        if let Ok(Some(mut p)) = dc.load_cog_profile(&disk_key) {
            // The stored profile keeps the URL it was first saved
            // with; refresh it so callers always see the current href.
            p.url = url.to_string();
            let arc = std::sync::Arc::new(p);
            cache.insert(key.clone(), arc.clone());
            return Ok(arc);
        }
    }
    // Header fetch — most COGs fit IFD0 + SubIFDs in the first 64 KiB.
    let header = http_range_get(http, url, 0, HEADER_RANGE - 1).await?;
    let profile = std::sync::Arc::new(parse_cog(http, url, header).await?);
    cache.insert(key, profile.clone());
    // Best-effort write-through to disk; tolerate failures silently
    // since the in-memory cache still works.
    if let Some(dc) = cog_disk_cache_slot().read().clone() {
        let disk_key = dc.cog_profile_key(url);
        let _ = dc.store_cog_profile(&disk_key, &profile);
    }
    Ok(profile)
}

/// Clear the profile cache; primarily for tests / long-running batch
/// jobs that want to bound memory.
pub fn clear_cog_profile_cache() {
    cog_profile_cache().clear();
}

async fn parse_cog(http: &reqwest::Client, url: &str, header: Bytes) -> Result<CogProfile> {
    let header_arc: Arc<Bytes> = Arc::new(header);
    let header_ref = header_arc.as_ref();
    if header_ref.len() < 8 {
        return Err(anyhow!("TIFF header truncated"));
    }
    if &header_ref[0..2] != b"II" {
        return Err(anyhow!(
            "non-little-endian TIFF ({:?}); not supported",
            &header_ref[0..2]
        ));
    }
    let magic = LittleEndian::read_u16(&header_ref[2..4]);
    if magic != 42 {
        return Err(anyhow!("BigTIFF or unknown magic {}; not supported", magic));
    }
    let ifd0_offset = LittleEndian::read_u32(&header_ref[4..8]) as u64;

    // Build a small "extended" buffer: we already have the first 64 KiB; if
    // any tag's value array sits outside that range, we'll do a follow-up
    // range-GET. To minimise round-trips we batch follow-up reads.
    let mut buffer = HeaderBuffer::new(http, url, header_arc);

    let ifd0 = parse_ifd(&mut buffer, ifd0_offset).await?;
    let sub_ifd_offsets = ifd0.sub_ifds.clone();

    let mut levels = Vec::new();
    levels.push(ifd_to_level(&ifd0));
    for off in sub_ifd_offsets {
        let ifd = parse_ifd(&mut buffer, off).await?;
        levels.push(ifd_to_level(&ifd));
    }

    // SampleFormat tag (339): 2 = signed int. HLS reflectance is signed
    // Int16; without this the bytes get read as UInt16 and the -9999 fill
    // (and any negative reflectance) becomes a huge positive value.
    let signed = ifd0.sample_format_code == 2;
    let sample_format = match (ifd0.bits_per_sample, ifd0.samples_per_pixel, signed) {
        (8, _, _) => SampleFormat::UInt8,
        (16, _, true) => SampleFormat::Int16,
        (16, _, false) => SampleFormat::UInt16,
        (bps, spp, _) => {
            return Err(anyhow!("unsupported sample shape {:?}", (bps, spp)))
        }
    };

    Ok(CogProfile {
        url: url.to_string(),
        width: ifd0.width,
        height: ifd0.height,
        sample_format,
        pixel_scale: ifd0.pixel_scale,
        tie_point: ifd0.tie_point,
        nodata: ifd0.nodata,
        levels,
    })
}

fn ifd_to_level(ifd: &Ifd) -> OverviewLevel {
    OverviewLevel {
        width: ifd.width,
        height: ifd.height,
        tile_width: ifd.tile_width,
        tile_height: ifd.tile_height,
        tile_offsets: ifd.tile_offsets.clone(),
        tile_byte_counts: ifd.tile_byte_counts.clone(),
        compression: ifd.compression,
        predictor: if ifd.predictor == 0 { 1 } else { ifd.predictor },
        bytes_per_sample: (ifd.bits_per_sample / 8).max(1),
    }
}

/// Fetch many tiles in parallel. Returns each tile as (index, decompressed bytes).
///
/// Consecutive tile ranges in the COG file get **coalesced into one HTTP
/// request** (similar to GDAL's `GDAL_HTTP_MERGE_CONSECUTIVE_RANGES`)
/// to amortise per-request HTTP overhead. The configurable gap allows
/// a small amount of "wasted" intermediate bytes to be transferred if
/// it spares us a round-trip.
/// Byte gap below which adjacent tiles are merged into one range GET.
/// Defaults to 4 KiB: measured sweeps (4 KiB → 1 MiB) showed bigger gaps
/// barely cut the GET count and didn't improve wall — at high concurrency
/// HTTP/2 multiplexing already hides per-request latency, and over-merging
/// re-reads bytes and costs parallelism. Override via `SPX_MERGE_GAP`.
fn merge_gap() -> u64 {
    static G: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *G.get_or_init(|| {
        std::env::var("SPX_MERGE_GAP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4 * 1024)
    })
}

pub async fn read_tiles(
    http: Arc<reqwest::Client>,
    url: Arc<String>,
    tiles: Vec<TileRequest>,
    level: &OverviewLevel,
    expected_size: usize,
    semaphore: Arc<tokio::sync::Semaphore>,
) -> Result<HashMap<usize, Vec<u8>>> {
    use futures::stream::{FuturesUnordered, StreamExt};
    let compression = level.compression;
    let predictor = level.predictor;
    let tile_w = level.tile_width as usize;
    let tile_h = level.tile_height as usize;
    let bps = level.bytes_per_sample as usize;

    // Maximum number of bytes to "tolerate" between two tiles in a
    // merged range request. Bigger gap => fewer, larger HTTP GETs (fewer
    // round trips, the dominant cost on RTT-bound links) at the cost of
    // re-reading some inter-tile bytes and less request parallelism.
    // Override with SPX_MERGE_GAP (bytes) to tune per network.
    let groups = coalesce_ranges(&tiles, merge_gap());
    // n_groups = actual HTTP range-GETs after coalescing contiguous tiles;
    // this (not n_tiles) is the RTT-bound request count.
    tracing::debug!(n_tiles = tiles.len(), n_groups = groups.len(), "read_tiles coalesce");
    // n_groups = actual HTTP range-GETs after coalescing contiguous tiles;
    // this (not n_tiles) is the RTT-bound request count.
    tracing::debug!(n_tiles = tiles.len(), n_groups = groups.len(), "read_tiles coalesce");

    // Detect runtime shape once. On the current-thread runtime we run
    // decode inline (blocking the reactor briefly is harmless on 1 CPU
    // because nothing else can run anyway). On the multi-thread
    // runtime we keep `spawn_blocking` so the reactor stays hot while
    // tile decode happens on a separate thread.
    let single_threaded = matches!(
        tokio::runtime::Handle::current().runtime_flavor(),
        tokio::runtime::RuntimeFlavor::CurrentThread
    );
    let mut tasks = FuturesUnordered::new();
    for group in groups {
        let sem = semaphore.clone();
        let http = http.clone();
        let url = url.clone();
        tasks.push(tokio::spawn(async move {
            let _permit = sem.acquire_owned().await.unwrap();
            let raw = http_range_get(&http, &url, group.start, group.end).await?;
            let group_tiles = group.tiles;
            let group_start = group.start;
            let raw_for_decode = raw;
            let decode = move || -> Result<Vec<(usize, Vec<u8>)>> {
                let raw = raw_for_decode.as_ref();
                let mut out: Vec<(usize, Vec<u8>)> = Vec::with_capacity(group_tiles.len());
                for tile in &group_tiles {
                    let local_start = (tile.offset - group_start) as usize;
                    let local_end = local_start + tile.byte_count as usize;
                    if local_end > raw.len() {
                        return Err(anyhow!(
                            "merged range short: tile {} wants {}..{}, got {}",
                            tile.index,
                            local_start,
                            local_end,
                            raw.len()
                        ));
                    }
                    let mut buf =
                        decode_tile(&raw[local_start..local_end], compression, expected_size)?;
                    apply_predictor(&mut buf, predictor, tile_w, tile_h, bps)?;
                    out.push((tile.index, buf));
                }
                Ok(out)
            };
            let decoded = if single_threaded {
                decode()?
            } else {
                tokio::task::spawn_blocking(decode)
                    .await
                    .context("decode join")??
            };
            Ok::<Vec<(usize, Vec<u8>)>, anyhow::Error>(decoded)
        }));
    }
    let mut out: HashMap<usize, Vec<u8>> = HashMap::with_capacity(64);
    while let Some(res) = tasks.next().await {
        let group_results = res.context("join")??;
        for (idx, bytes) in group_results {
            out.insert(idx, bytes);
        }
    }
    Ok(out)
}

/// One contiguous HTTP range covering 1+ tiles.
#[derive(Debug)]
struct RangeGroup {
    start: u64,
    end: u64,
    tiles: Vec<TileRequest>,
}

/// Merge sorted tiles into ranges. Tiles whose byte ranges are within
/// `max_gap` of the previous tile join the same group, capping the
/// total range at a few MiB to keep per-request cost reasonable.
fn coalesce_ranges(tiles: &[TileRequest], max_gap: u64) -> Vec<RangeGroup> {
    if tiles.is_empty() {
        return Vec::new();
    }
    let mut sorted = tiles.to_vec();
    sorted.sort_by_key(|t| t.offset);
    let mut groups: Vec<RangeGroup> = Vec::new();
    const MAX_GROUP_BYTES: u64 = 1024 * 1024; // 1 MiB cap per request
    for tile in sorted {
        let tile_end = tile.offset + tile.byte_count - 1;
        if let Some(last) = groups.last_mut() {
            let gap = tile.offset.saturating_sub(last.end + 1);
            let merged_end = tile_end.max(last.end);
            let merged_size = merged_end - last.start + 1;
            if gap <= max_gap && merged_size <= MAX_GROUP_BYTES {
                last.end = merged_end;
                last.tiles.push(tile);
                continue;
            }
        }
        groups.push(RangeGroup {
            start: tile.offset,
            end: tile_end,
            tiles: vec![tile],
        });
    }
    groups
}

/// DEFLATE decompression via libdeflate (FFI to libdeflate C library).
///
/// Benchmarks libdeflate at ~2× the throughput of flate2 (pure Rust)
/// for the kinds of small zlib streams S2 L2A COGs emit. On 1 CPU this
/// is the single biggest CPU win available; on 16 CPUs it still pays
/// because we have ~150-800 decompress calls per build.
///
/// libdeflater takes a pre-allocated output buffer rather than growing
/// one; we size to the expected tile bytes (tile_w × tile_h × bps).
fn decode_tile(raw: &[u8], compression: u32, expected: usize) -> Result<Vec<u8>> {
    match compression {
        COMPRESSION_NONE => Ok(raw.to_vec()),
        COMPRESSION_DEFLATE | COMPRESSION_ADOBE_DEFLATE => {
            // libdeflate expects exact output capacity; tile bytes are
            // deterministic from the IFD, so `expected` is reliable.
            let mut out = vec![0u8; expected];
            let mut decompressor = libdeflater::Decompressor::new();
            let written = decompressor
                .zlib_decompress(raw, &mut out)
                .map_err(|e| anyhow!("DEFLATE (libdeflate): {e:?}"))?;
            out.truncate(written);
            Ok(out)
        }
        other => Err(anyhow!("unsupported compression {other}")),
    }
}

/// Horizontal differencing predictor (TIFF tag 317 = 2). Each row was
/// stored as deltas from the previous pixel and we cumulate left-to-
/// right with modular wrap matching the sample's natural integer width.
///
/// Hot path. Rows are independent → rayon-parallel across rows. The
/// inner u16 prefix-sum uses an 8-lane SIMD reduction; the u8 path is
/// kept scalar because S2 SCL tiles are small enough that SIMD doesn't
/// pay off after stride overhead.
fn apply_predictor(buf: &mut [u8], predictor: u16, tile_w: usize, tile_h: usize, bps: usize) -> Result<()> {
    if predictor != 2 {
        return Ok(());
    }
    let row_bytes = tile_w * bps;
    let usable_rows = buf.len() / row_bytes.max(1);
    let rows_to_process = tile_h.min(usable_rows);
    // Per-row work is small (1-2 KiB) so we DON'T rayon-parallelise
    // here — calling rayon from inside a tokio task pings the rayon
    // thread pool for every tile and the dispatch overhead exceeds
    // the work. Tile-level parallelism happens at the read_tiles
    // layer where each tile becomes its own tokio task; the per-row
    // loop stays sequential with a SIMD inner kernel.
    let rows = &mut buf[..rows_to_process * row_bytes];
    match bps {
        1 => {
            for row in rows.chunks_mut(row_bytes) {
                predictor_invert_u8(row);
            }
        }
        2 => {
            for row in rows.chunks_mut(row_bytes) {
                predictor_invert_u16(row);
            }
        }
        _ => return Err(anyhow!("predictor=2 with {} byte samples not supported", bps)),
    }
    Ok(())
}

#[inline]
fn predictor_invert_u8(row: &mut [u8]) {
    for c in 1..row.len() {
        row[c] = row[c].wrapping_add(row[c - 1]);
    }
}

#[inline]
fn predictor_invert_u16(row: &mut [u8]) {
    let n = row.len() / 2;
    if n == 0 {
        return;
    }
    // 8-lane SIMD prefix sum (Sklansky pattern) reduces the per-row
    // serial dependency from O(n) to O(n/8 + log2(8)). The kernel runs
    // inside a `tokio::task::spawn_blocking` so it doesn't fight the
    // tokio reactor; the inner work is purely CPU-bound now.
    const LANES: usize = 8;
    let blocks = n / LANES;
    let rem = n % LANES;
    let mut carry: u16 = 0;
    let mut staging = [0u16; LANES];
    for blk in 0..blocks {
        let off = blk * LANES * 2;
        for i in 0..LANES {
            let bi = off + i * 2;
            staging[i] = u16::from_le_bytes([row[bi], row[bi + 1]]);
        }
        let v0 = wide::u16x8::new(staging);
        let s1 = {
            let mut sh = [0u16; LANES];
            sh[1..].copy_from_slice(&staging[..LANES - 1]);
            v0 + wide::u16x8::new(sh)
        };
        let a1: [u16; LANES] = s1.into();
        let s2 = {
            let mut sh = [0u16; LANES];
            sh[2..].copy_from_slice(&a1[..LANES - 2]);
            s1 + wide::u16x8::new(sh)
        };
        let a2: [u16; LANES] = s2.into();
        let s3 = {
            let mut sh = [0u16; LANES];
            sh[4..].copy_from_slice(&a2[..LANES - 4]);
            s2 + wide::u16x8::new(sh)
        };
        let mut out: [u16; LANES] = s3.into();
        if carry != 0 {
            for v in out.iter_mut() {
                *v = v.wrapping_add(carry);
            }
        }
        for i in 0..LANES {
            let bi = off + i * 2;
            row[bi] = out[i] as u8;
            row[bi + 1] = (out[i] >> 8) as u8;
        }
        carry = out[LANES - 1];
    }
    // Scalar tail for the last partial block.
    let base = blocks * LANES * 2;
    let mut prev = carry;
    for c in 0..rem {
        let bi = base + c * 2;
        let delta = u16::from_le_bytes([row[bi], row[bi + 1]]);
        let v = prev.wrapping_add(delta);
        row[bi] = v as u8;
        row[bi + 1] = (v >> 8) as u8;
        prev = v;
    }
}

/// Process-wide LRU of raw HTTP range responses keyed by (url, start, end).
/// Sized to ~200 MB so warm reruns within one process avoid re-issuing
/// network requests for tile bytes the binary already pulled. Matches
/// GDAL's `VSI_CACHE` shape; doesn't help across processes (use the
/// disk cache for that), but does help any workflow that rebuilds the
/// same scene set more than once in one invocation.
fn tile_cache() -> &'static moka::sync::Cache<(String, u64, u64), Arc<Bytes>> {
    static CACHE: std::sync::OnceLock<moka::sync::Cache<(String, u64, u64), Arc<Bytes>>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(|| {
        moka::sync::Cache::builder()
            .max_capacity(200 * 1024 * 1024)
            .weigher(|_, v: &Arc<Bytes>| v.len().try_into().unwrap_or(u32::MAX))
            .build()
    })
}

pub fn clear_tile_cache() {
    tile_cache().invalidate_all();
}

/// Range-GET with exponential-backoff retries on throttle (429), service-unavailable
/// (502/503/504), and transient connection errors. Honours the `Retry-After`
/// header when the server sends one. Per-attempt delays start at 200 ms and
/// double, with jitter, up to 5 attempts. Total worst-case wait is ~6 s before
/// a hard failure, which is well below the per-task budget but long enough to
/// ride out the typical AWS S3 cool-down window.
pub async fn http_range_get(
    http: &reqwest::Client,
    url: &str,
    start: u64,
    end: u64,
) -> Result<Bytes> {
    let cache_key = (url.to_string(), start, end);
    if let Some(cached) = tile_cache().get(&cache_key) {
        return Ok(cached.as_ref().clone());
    }

    const MAX_ATTEMPTS: u32 = 5;
    const BASE_DELAY_MS: u64 = 200;
    let mut attempt: u32 = 0;

    loop {
        let send_result = http
            .get(url)
            .header("Range", format!("bytes={start}-{end}"))
            .send()
            .await;

        let response_handling: Result<Bytes, RetryDecision> = match send_result {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    match resp.bytes().await {
                        Ok(b) => Ok(b),
                        Err(e) => Err(RetryDecision::Transient(format!("body read: {e}"), None)),
                    }
                } else if is_retryable_status(status) {
                    // Optional Retry-After hint (seconds or HTTP-date).
                    let retry_after_secs = resp
                        .headers()
                        .get(reqwest::header::RETRY_AFTER)
                        .and_then(|v| v.to_str().ok())
                        .and_then(|s| s.trim().parse::<u64>().ok());
                    Err(RetryDecision::Transient(
                        format!("status {status}"),
                        retry_after_secs,
                    ))
                } else {
                    Err(RetryDecision::Fatal(format!("status {status}")))
                }
            }
            Err(e) if is_retryable_error(&e) => {
                Err(RetryDecision::Transient(format!("send: {e}"), None))
            }
            Err(e) => Err(RetryDecision::Fatal(format!("send: {e}"))),
        };

        match response_handling {
            Ok(bytes) => {
                tile_cache().insert(cache_key, Arc::new(bytes.clone()));
                return Ok(bytes);
            }
            Err(RetryDecision::Fatal(msg)) => {
                return Err(anyhow!("range read {url} [{start}..={end}]: {msg}"));
            }
            Err(RetryDecision::Transient(msg, retry_after)) => {
                attempt += 1;
                if attempt >= MAX_ATTEMPTS {
                    return Err(anyhow!(
                        "range read {url} [{start}..={end}] gave up after {} attempts: {msg}",
                        attempt
                    ));
                }
                let delay_ms = retry_after
                    .map(|s| s.saturating_mul(1000))
                    .unwrap_or_else(|| {
                        let base = BASE_DELAY_MS << (attempt - 1).min(6);
                        // Decorrelated jitter: in [base/2, base+base/2)
                        let jitter = pseudo_jitter_ms() % base.max(1);
                        (base / 2) + jitter
                    });
                tracing::debug!(
                    url,
                    attempt,
                    delay_ms,
                    "retryable HTTP failure: {}",
                    msg
                );
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            }
        }
    }
}

enum RetryDecision {
    /// Stop, propagate the error.
    Fatal(String),
    /// Backoff and retry. Optional Retry-After (seconds) overrides the default schedule.
    Transient(String, Option<u64>),
}

fn is_retryable_status(s: reqwest::StatusCode) -> bool {
    matches!(s.as_u16(), 408 | 425 | 429 | 500 | 502 | 503 | 504)
}

fn is_retryable_error(e: &reqwest::Error) -> bool {
    e.is_timeout()
        || e.is_connect()
        || e.is_request()
        // reqwest classifies premature stream resets as decode errors.
        || e.to_string().contains("connection reset")
        || e.to_string().contains("stream error")
}

/// Cheap jittering: take system nanos modulo something. We're not seeding
/// rand; just want a small unpredictable offset so concurrent retries don't
/// thunder back at the server in lockstep.
fn pseudo_jitter_ms() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static CTR: AtomicU64 = AtomicU64::new(0);
    let salt = CTR.fetch_add(0x9E37_79B9, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0);
    (salt ^ nanos) >> 8
}

#[derive(Debug, Default)]
struct Ifd {
    width: u32,
    height: u32,
    bits_per_sample: u32,
    samples_per_pixel: u32,
    /// TIFF SampleFormat (tag 339): 1 = unsigned int (default when
    /// absent), 2 = signed int, 3 = IEEE float. 0 means the tag was not
    /// present, treated as unsigned.
    sample_format_code: u32,
    compression: u32,
    tile_width: u32,
    tile_height: u32,
    tile_offsets: Vec<u64>,
    tile_byte_counts: Vec<u64>,
    sub_ifds: Vec<u64>,
    pixel_scale: Option<[f64; 3]>,
    tie_point: Option<[f64; 6]>,
    nodata: Option<f64>,
    predictor: u16,
    planar_config: u16,
}

async fn parse_ifd(buffer: &mut HeaderBuffer<'_>, offset: u64) -> Result<Ifd> {
    buffer.ensure(offset, 2).await?;
    let n_entries = LittleEndian::read_u16(buffer.slice(offset, 2)?) as usize;
    let entries_start = offset + 2;
    let entries_size = (n_entries as u64) * 12 + 4;
    buffer.ensure(entries_start, entries_size as usize).await?;
    // Copy the 12-byte entries out of the buffer so subsequent
    // `read_*` calls (which may re-borrow buffer mutably) don't conflict.
    let mut raw_entries: Vec<[u8; 12]> = Vec::with_capacity(n_entries);
    for i in 0..n_entries {
        let entry_offset = entries_start + (i as u64) * 12;
        let mut e = [0u8; 12];
        e.copy_from_slice(buffer.slice(entry_offset, 12)?);
        raw_entries.push(e);
    }

    let mut ifd = Ifd::default();
    for raw in &raw_entries {
        let tag = LittleEndian::read_u16(&raw[0..2]);
        let typ = LittleEndian::read_u16(&raw[2..4]);
        let count = LittleEndian::read_u32(&raw[4..8]) as usize;
        let value_or_offset = &raw[8..12];

        match tag {
            TAG_IMAGE_WIDTH => ifd.width = read_scalar_u32(buffer, typ, count, value_or_offset).await?,
            TAG_IMAGE_LENGTH => ifd.height = read_scalar_u32(buffer, typ, count, value_or_offset).await?,
            TAG_BITS_PER_SAMPLE => {
                ifd.bits_per_sample = read_scalar_u32(buffer, typ, count, value_or_offset).await?
            }
            TAG_SAMPLES_PER_PIXEL => {
                ifd.samples_per_pixel = read_scalar_u32(buffer, typ, count, value_or_offset).await?
            }
            TAG_SAMPLE_FORMAT => {
                ifd.sample_format_code =
                    read_scalar_u32(buffer, typ, count, value_or_offset).await?
            }
            TAG_COMPRESSION => {
                ifd.compression = read_scalar_u32(buffer, typ, count, value_or_offset).await?
            }
            TAG_TILE_WIDTH => ifd.tile_width = read_scalar_u32(buffer, typ, count, value_or_offset).await?,
            TAG_TILE_LENGTH => {
                ifd.tile_height = read_scalar_u32(buffer, typ, count, value_or_offset).await?
            }
            TAG_TILE_OFFSETS => {
                ifd.tile_offsets = read_array_u64(buffer, typ, count, value_or_offset).await?;
            }
            TAG_TILE_BYTE_COUNTS => {
                ifd.tile_byte_counts = read_array_u64(buffer, typ, count, value_or_offset).await?;
            }
            TAG_SUB_IFDS => {
                ifd.sub_ifds = read_array_u64(buffer, typ, count, value_or_offset).await?;
            }
            TAG_PIXEL_SCALE => {
                let arr = read_array_f64(buffer, typ, count, value_or_offset).await?;
                if arr.len() >= 3 {
                    ifd.pixel_scale = Some([arr[0], arr[1], arr[2]]);
                }
            }
            TAG_TIE_POINT => {
                let arr = read_array_f64(buffer, typ, count, value_or_offset).await?;
                if arr.len() >= 6 {
                    ifd.tie_point = Some([arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]]);
                }
            }
            TAG_NODATA => {
                let s = read_ascii(buffer, typ, count, value_or_offset).await?;
                ifd.nodata = s.trim_end_matches('\0').trim().parse().ok();
            }
            TAG_PREDICTOR => {
                ifd.predictor = read_scalar_u32(buffer, typ, count, value_or_offset).await? as u16;
            }
            TAG_PLANAR_CONFIG => {
                ifd.planar_config = read_scalar_u32(buffer, typ, count, value_or_offset).await? as u16;
            }
            TAG_GEO_TIFF_KEYS => {}
            _ => {}
        }
    }
    Ok(ifd)
}

async fn read_scalar_u32(
    buffer: &mut HeaderBuffer<'_>,
    typ: u16,
    count: usize,
    value: &[u8],
) -> Result<u32> {
    let arr = read_array_u32(buffer, typ, count, value).await?;
    arr.first()
        .copied()
        .ok_or_else(|| anyhow!("expected at least one value for tag"))
}

async fn read_array_u32(
    buffer: &mut HeaderBuffer<'_>,
    typ: u16,
    count: usize,
    value_or_offset: &[u8],
) -> Result<Vec<u32>> {
    let elem_size = type_size(typ)?;
    let total_size = elem_size * count;
    let bytes = read_inline_or_offset(buffer, total_size, value_or_offset).await?;
    let mut out = Vec::with_capacity(count);
    match typ {
        1 => {
            for i in 0..count {
                out.push(bytes[i] as u32);
            }
        }
        3 => {
            for i in 0..count {
                out.push(LittleEndian::read_u16(&bytes[i * 2..(i + 1) * 2]) as u32);
            }
        }
        4 => {
            for i in 0..count {
                out.push(LittleEndian::read_u32(&bytes[i * 4..(i + 1) * 4]));
            }
        }
        _ => return Err(anyhow!("unsupported type {typ} for u32 read")),
    }
    Ok(out)
}

async fn read_array_u64(
    buffer: &mut HeaderBuffer<'_>,
    typ: u16,
    count: usize,
    value_or_offset: &[u8],
) -> Result<Vec<u64>> {
    // Type 4 (LONG) is u32 in classic TIFF; we widen.
    let arr = read_array_u32(buffer, typ, count, value_or_offset).await?;
    Ok(arr.into_iter().map(|v| v as u64).collect())
}

async fn read_array_f64(
    buffer: &mut HeaderBuffer<'_>,
    typ: u16,
    count: usize,
    value_or_offset: &[u8],
) -> Result<Vec<f64>> {
    let elem_size = type_size(typ)?;
    let total_size = elem_size * count;
    let bytes = read_inline_or_offset(buffer, total_size, value_or_offset).await?;
    let mut out = Vec::with_capacity(count);
    match typ {
        11 => {
            // FLOAT (f32) — promote to f64.
            for i in 0..count {
                let v = LittleEndian::read_f32(&bytes[i * 4..(i + 1) * 4]);
                out.push(v as f64);
            }
        }
        12 => {
            for i in 0..count {
                out.push(LittleEndian::read_f64(&bytes[i * 8..(i + 1) * 8]));
            }
        }
        _ => return Err(anyhow!("unsupported type {typ} for f64 read")),
    }
    Ok(out)
}

async fn read_ascii(
    buffer: &mut HeaderBuffer<'_>,
    typ: u16,
    count: usize,
    value_or_offset: &[u8],
) -> Result<String> {
    if typ != 2 {
        return Err(anyhow!("expected ASCII type"));
    }
    let bytes = read_inline_or_offset(buffer, count, value_or_offset).await?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

async fn read_inline_or_offset(
    buffer: &mut HeaderBuffer<'_>,
    total_size: usize,
    value_or_offset: &[u8],
) -> Result<Vec<u8>> {
    if total_size <= 4 {
        Ok(value_or_offset[..total_size].to_vec())
    } else {
        let off = LittleEndian::read_u32(value_or_offset) as u64;
        buffer.ensure(off, total_size).await?;
        Ok(buffer.slice(off, total_size)?.to_vec())
    }
}

fn type_size(typ: u16) -> Result<usize> {
    Ok(match typ {
        1 | 2 | 6 | 7 => 1,
        3 | 8 => 2,
        4 | 9 | 11 => 4,
        5 | 10 | 12 => 8,
        _ => return Err(anyhow!("unknown TIFF type code {typ}")),
    })
}

/// Owns a contiguous byte buffer keyed by offsets into the TIFF file.
/// On miss we issue another range-GET and splice the result in. In
/// practice, S2 L2A COGs return everything we need in the first 64 KiB
/// so the splice path is rare.
struct HeaderBuffer<'a> {
    http: &'a reqwest::Client,
    url: &'a str,
    chunks: Vec<(u64, Arc<Bytes>)>, // (start_offset, bytes)
}

impl<'a> HeaderBuffer<'a> {
    fn new(http: &'a reqwest::Client, url: &'a str, header: Arc<Bytes>) -> Self {
        let chunks = vec![(0u64, header)];
        Self { http, url, chunks }
    }
    async fn ensure(&mut self, offset: u64, length: usize) -> Result<()> {
        if self.has(offset, length) {
            return Ok(());
        }
        // Fetch a 64 KiB block aligned around the requested region.
        let block_start = (offset / 16_384) * 16_384;
        let block_end = block_start + 65_535;
        let new_chunk = http_range_get(self.http, self.url, block_start, block_end).await?;
        self.chunks.push((block_start, Arc::new(new_chunk)));
        if !self.has(offset, length) {
            return Err(anyhow!(
                "byte range [{offset}..{}] not covered by IFD fetch",
                offset + length as u64
            ));
        }
        Ok(())
    }
    fn has(&self, offset: u64, length: usize) -> bool {
        self.chunks
            .iter()
            .any(|(start, bytes)| *start <= offset && offset + length as u64 <= start + bytes.len() as u64)
    }
    fn slice(&self, offset: u64, length: usize) -> Result<&[u8]> {
        for (start, bytes) in &self.chunks {
            if *start <= offset && offset + length as u64 <= start + bytes.len() as u64 {
                let local = (offset - start) as usize;
                return Ok(&bytes[local..local + length]);
            }
        }
        Err(anyhow!("offset {offset} +{length} not in any chunk"))
    }
}

/// Stitch a set of decoded tiles into a single rectangular array.
/// `output` is row-major (height × width × elem). Tiles writing into
/// non-overlapping output regions are written in parallel via rayon.
pub fn stitch_tiles(
    output: &mut [u8],
    window: PixelWindow,
    level: &OverviewLevel,
    tiles: &HashMap<usize, Vec<u8>>,
    elem_size: usize,
) -> Result<()> {
    let out_stride = window.width as usize * elem_size;
    let tile_stride = level.tile_width as usize * elem_size;
    let tile_rows = level.tile_height as usize;
    // For each tile, compute the destination row-range and slice
    // `output` into row chunks. Tiles don't overlap, so even though
    // we hand each tile its own &mut subslice, no two tiles touch the
    // same byte. We use chunked row spans via rayon's `par_iter` over
    // the tile list and `split_at_mut` chains to hand out exclusive
    // mutable views.
    let tiles_x = level.tiles_x();
    let mut plans: Vec<TilePlan> = Vec::with_capacity(tiles.len());
    for (&idx, decoded) in tiles {
        let tx = (idx as u32) % tiles_x;
        let ty = (idx as u32) / tiles_x;
        let tile_pixel_x = tx * level.tile_width;
        let tile_pixel_y = ty * level.tile_height;
        let x0 = tile_pixel_x.max(window.col_off);
        let y0 = tile_pixel_y.max(window.row_off);
        let x1 = (tile_pixel_x + level.tile_width).min(window.col_off + window.width);
        let y1 = (tile_pixel_y + level.tile_height).min(window.row_off + window.height);
        if x1 <= x0 || y1 <= y0 {
            continue;
        }
        let tile_local_x0 = (x0 - tile_pixel_x) as usize;
        let tile_local_y0 = (y0 - tile_pixel_y) as usize;
        let copy_w = (x1 - x0) as usize;
        let copy_h = (y1 - y0) as usize;
        if decoded.len() < tile_rows * tile_stride {
            continue;
        }
        let out_x = (x0 - window.col_off) as usize;
        let out_y = (y0 - window.row_off) as usize;
        plans.push(TilePlan {
            decoded,
            tile_local_x0,
            tile_local_y0,
            copy_w,
            copy_h,
            out_x,
            out_y,
            tile_stride,
        });
    }
    // Sort by dst out_y so the disjoint mutable row ranges can be
    // partitioned safely. For tiles writing into the same destination
    // row range (rare; only happens if a level's tiling overlaps the
    // window in unusual ways) we fall back to sequential within those.
    // Sequential per-tile copy. Each tile typically writes a small
    // contiguous rectangle (a few rows × tile_width bytes); rayon
    // overhead exceeds the win when many `fetch_band` calls are in
    // flight concurrently.
    for plan in &plans {
        let bytes_per_pixel = elem_size;
        let len = plan.copy_w * bytes_per_pixel;
        for r in 0..plan.copy_h {
            let src_row = plan.tile_local_y0 + r;
            let src_off = src_row * plan.tile_stride + plan.tile_local_x0 * bytes_per_pixel;
            let dst_off = (plan.out_y + r) * out_stride + plan.out_x * bytes_per_pixel;
            output[dst_off..dst_off + len]
                .copy_from_slice(&plan.decoded[src_off..src_off + len]);
        }
    }
    Ok(())
}

struct TilePlan<'a> {
    decoded: &'a Vec<u8>,
    tile_local_x0: usize,
    tile_local_y0: usize,
    copy_w: usize,
    copy_h: usize,
    out_x: usize,
    out_y: usize,
    tile_stride: usize,
}
