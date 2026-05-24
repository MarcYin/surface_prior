//! End-to-end orchestration: scout → select → fetch → compose → write.
//!
//! Mirrors `surface_priors`' StacApiSource + ChunkedCompositor flow.
//! Single-tile selection (top-k by clearness across all picked scenes) is
//! the MVP — tile-aware partitioning is omitted because the goal here
//! is to compare the network-layer + composite cost to Python end-to-end.

use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::cog::{open_cog, read_tiles, stitch_tiles, CogProfile, PixelWindow, SampleFormat};
use crate::grid::{cog_window_for_utm, resample_u16_to_u16, resample_u8_to_u8, GridSpec};
use crate::stac::StacItem;

/// Per-(scene, chunk) statistics; cached on disk via [`crate::disk_cache`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneChunkStat {
    pub chunk_id: u32,
    pub usable_fraction: f32,
    pub mean_clear: f32,
}

/// All the bands the surface_priors Python pipeline reads for S2 L2A
/// on Element84, in stable order. Matches `EARTH_SEARCH_S2_ALIASES`.
pub const S2_BAND_ASSETS: [&str; 12] = [
    "coastal",
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir",
    "nir08",
    "nir09",
    "swir16",
    "swir22",
];

#[derive(Debug, Clone)]
pub struct SceneStats {
    pub item_id: String,
    pub usable_fraction: f32,
    pub mean_clear: f32,
}

/// SCL classes considered "clear sky over land/water" (matches Python).
const SCL_CLEAR: [u8; 4] = [4, 5, 6, 11];
const SCL_NODATA: u8 = 0;

/// Scout a single scene: open its SCL COG, read at coarse resolution,
/// compute per-AOI usable_fraction + mean_clear.
pub async fn scout_scene(
    http: Arc<reqwest::Client>,
    scene: &StacItem,
    grid: &GridSpec,
    coarse_resolution: f64,
    semaphore: Arc<tokio::sync::Semaphore>,
    scl_asset: &str,
) -> Result<SceneStats> {
    let url = scene
        .assets
        .get(scl_asset)
        .ok_or_else(|| anyhow::anyhow!("scene {} missing {} asset", scene.id, scl_asset))?
        .clone();
    let cog = open_cog(&http, &url)
        .await
        .with_context(|| format!("open SCL {url}"))?;
    // SCL native is 20 m; we want coarse_resolution metres per pixel.
    let scl_native_res = cog
        .pixel_scale
        .map(|s| s[0])
        .unwrap_or(20.0);
    let decimation = (coarse_resolution / scl_native_res).max(1.0);
    let level = cog.level_for_decimation(decimation);
    let pixel_size = scl_native_res * (cog.width as f64 / level.width as f64);
    let origin = cog
        .tie_point
        .map(|t| [t[3], t[4]])
        .unwrap_or([0.0, 0.0]);
    let win = match cog_window_for_utm(
        origin,
        pixel_size,
        level.width,
        level.height,
        grid.bounds,
    ) {
        Some(w) => w,
        None => {
            return Ok(SceneStats {
                item_id: scene.id.clone(),
                usable_fraction: 0.0,
                mean_clear: f32::NAN,
            });
        }
    };
    let tiles = level.tiles_for_window(win);
    let tile_bytes_expected =
        (level.tile_width as usize) * (level.tile_height as usize) * cog.sample_format.bytes();
    let url_arc = Arc::new(url);
    let decoded = read_tiles(
        http.clone(),
        url_arc.clone(),
        tiles,
        level,
        tile_bytes_expected,
        semaphore.clone(),
    )
    .await?;
    // Stitch into an array sized to win (in COG pixels, at coarse level).
    let mut buf = vec![0u8; (win.width * win.height) as usize];
    stitch_tiles(&mut buf, win, level, &decoded, 1)?;
    // Resample to grid (output resolution may be different from level pixel size).
    let dst_w = grid.width;
    let dst_h = grid.height;
    let level_origin = [
        origin[0] + win.col_off as f64 * pixel_size,
        origin[1] - win.row_off as f64 * pixel_size,
    ];
    let dst_origin = [grid.bounds[0], grid.bounds[3]];
    let scl = resample_u8_to_u8(
        &buf,
        (win.width, win.height),
        level_origin,
        pixel_size,
        (dst_w, dst_h),
        dst_origin,
        grid.resolution,
    )?;
    let (usable, mean_clear) = scl_to_stats(&scl);
    Ok(SceneStats {
        item_id: scene.id.clone(),
        usable_fraction: usable,
        mean_clear,
    })
}

fn scl_to_stats(scl: &[u8]) -> (f32, f32) {
    let mut valid = 0u32;
    let mut clear = 0u32;
    for &v in scl {
        if v == SCL_NODATA {
            continue;
        }
        valid += 1;
        if SCL_CLEAR.contains(&v) {
            clear += 1;
        }
    }
    let total = scl.len() as f32;
    let usable = if total > 0.0 { valid as f32 / total } else { 0.0 };
    let mean_clear = if valid > 0 {
        clear as f32 / valid as f32
    } else {
        f32::NAN
    };
    (usable, mean_clear)
}

#[derive(Debug, Clone)]
pub struct ScenePick<'a> {
    pub scene: &'a StacItem,
    pub clear: f32,
}

impl<'a> Copy for ScenePick<'a> {}

/// Chunk-level tile-aware selection.
///
/// Mirrors Python's `select_tile_aware`: for each chunk window, pick
/// top-K scenes per *required* MGRS tile (where "required" means the
/// tile has non-zero exclusive coverage of the chunk), then union the
/// picks across chunks. This is the production selector — it both
/// avoids over-picking (chunks that need only one tile only contribute
/// top-K scenes, not top-K-per-tile-in-AOI) and guarantees seam
/// coverage.
pub fn select_chunk_tile_aware<'a>(
    scenes: &'a [StacItem],
    stats: &HashMap<String, SceneStats>,
    partition: &crate::tile_classification::TilePartition,
    top_k: usize,
) -> Vec<ScenePick<'a>> {
    // Filter scored scenes by tile.
    let mut by_tile: std::collections::HashMap<String, Vec<ScenePick<'a>>> = std::collections::HashMap::new();
    for scene in scenes {
        let Some(st) = stats.get(&scene.id) else { continue };
        if st.usable_fraction <= 0.0 || !st.mean_clear.is_finite() {
            continue;
        }
        if scene.mgrs_tile.is_empty() {
            continue;
        }
        by_tile
            .entry(scene.mgrs_tile.clone())
            .or_default()
            .push(ScenePick { scene, clear: st.mean_clear });
    }
    for v in by_tile.values_mut() {
        v.sort_by(|a, b| b.clear.partial_cmp(&a.clear).unwrap_or(std::cmp::Ordering::Equal));
    }

    let mut selected: Vec<ScenePick> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    // For each chunk, union top-k per required tile.
    for req in partition.requirements.values() {
        for tile in &req.required_tiles {
            let Some(tile_picks) = by_tile.get(tile) else { continue };
            for pick in tile_picks.iter().take(top_k) {
                if seen.insert(pick.scene.id.clone()) {
                    selected.push(*pick);
                }
            }
        }
    }
    selected
}

pub fn select_top_k<'a>(
    scenes: &'a [StacItem],
    stats: &HashMap<String, SceneStats>,
    top_k: usize,
) -> Vec<ScenePick<'a>> {
    // Tile-aware MVP: group scenes by MGRS tile present in the search,
    // pick top_k per tile by mean_clear, then union round-robin so
    // every tile gets its #1 before any tile gets its #2. Without a
    // chunk-vs-tile geometry classifier this over-picks single-tile
    // chunks slightly but eliminates the seam-stripe coverage gap.
    let mut by_tile: std::collections::BTreeMap<String, Vec<ScenePick>> =
        std::collections::BTreeMap::new();
    let mut untiled: Vec<ScenePick> = Vec::new();
    for scene in scenes {
        let Some(st) = stats.get(&scene.id) else { continue };
        if st.usable_fraction <= 0.0 || !st.mean_clear.is_finite() {
            continue;
        }
        let pick = ScenePick {
            scene,
            clear: st.mean_clear,
        };
        if scene.mgrs_tile.is_empty() {
            untiled.push(pick);
        } else {
            by_tile.entry(scene.mgrs_tile.clone()).or_default().push(pick);
        }
    }
    for picks in by_tile.values_mut() {
        picks.sort_by(|a, b| b.clear.partial_cmp(&a.clear).unwrap_or(std::cmp::Ordering::Equal));
        picks.truncate(top_k.max(1));
    }
    untiled.sort_by(|a, b| b.clear.partial_cmp(&a.clear).unwrap_or(std::cmp::Ordering::Equal));
    untiled.truncate(top_k.max(1));
    // Round-robin union across tiles.
    let mut out: Vec<ScenePick> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let max_rank = by_tile
        .values()
        .map(|v| v.len())
        .max()
        .unwrap_or(0)
        .max(untiled.len());
    let mut iters: Vec<std::vec::IntoIter<ScenePick>> = by_tile
        .into_values()
        .map(|v| v.into_iter())
        .collect();
    iters.push(untiled.into_iter());
    let mut buffered: Vec<Vec<ScenePick>> = iters.into_iter().map(|it| it.collect()).collect();
    for rank in 0..max_rank {
        for col in &mut buffered {
            if rank < col.len() {
                let pick = col[rank].clone();
                if seen.insert(pick.scene.id.clone()) {
                    out.push(pick);
                }
            }
        }
    }
    out
}

/// Fetch one band's array (u16) for one scene, with detailed timing.
/// Reports HTTP / decode / stitch / resample sub-phases via tracing
/// when RUST_LOG=debug; otherwise silent so production runs aren't
/// chatty.
pub async fn fetch_band(
    http: Arc<reqwest::Client>,
    scene: &StacItem,
    band_asset: &str,
    grid: &GridSpec,
    semaphore: Arc<tokio::sync::Semaphore>,
) -> Result<Option<Vec<u16>>> {
    let url = match scene.assets.get(band_asset) {
        Some(u) => u.clone(),
        None => return Ok(None),
    };
    let t_open = std::time::Instant::now();
    let cog = open_cog(&http, &url).await.ok();
    let dt_open = t_open.elapsed().as_secs_f64();
    let Some(cog) = cog else { return Ok(None) };
    if cog.sample_format != SampleFormat::UInt16 {
        return Ok(None);
    }
    let native_res = cog
        .pixel_scale
        .map(|s| s[0])
        .unwrap_or(10.0);
    let decimation = (grid.resolution / native_res).max(1.0);
    let level = cog.level_for_decimation(decimation);
    let pixel_size = native_res * (cog.width as f64 / level.width as f64);
    let origin = cog
        .tie_point
        .map(|t| [t[3], t[4]])
        .unwrap_or([0.0, 0.0]);
    let win = match cog_window_for_utm(
        origin,
        pixel_size,
        level.width,
        level.height,
        grid.bounds,
    ) {
        Some(w) => w,
        None => return Ok(None),
    };
    let tiles = level.tiles_for_window(win);
    let n_tiles = tiles.len();
    let tile_bytes_expected =
        (level.tile_width as usize) * (level.tile_height as usize) * cog.sample_format.bytes();
    let url_arc = Arc::new(url);
    let t_read = std::time::Instant::now();
    let decoded = read_tiles(
        http.clone(),
        url_arc.clone(),
        tiles,
        level,
        tile_bytes_expected,
        semaphore.clone(),
    )
    .await?;
    let dt_read = t_read.elapsed().as_secs_f64();
    // Stitch tiles into a contiguous u16 buffer, then resample via the
    // AVX2 + FMA bilinear kernel. We tried a tile-aware resample that
    // skipped the stitch alloc but the per-pixel HashMap+modulo lookup
    // cost more than the ~1.5 s/year stitch saves; the contiguous
    // buffer also keeps the AVX2 kernel's loads cache-friendly.
    let t_stitch = std::time::Instant::now();
    let n_pixels = (win.width * win.height) as usize;
    let mut src_u16 = vec![0u16; n_pixels];
    {
        let buf_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut src_u16);
        stitch_tiles(buf_bytes, win, level, &decoded, 2)?;
    }
    let dt_stitch = t_stitch.elapsed().as_secs_f64();
    let level_origin = [
        origin[0] + win.col_off as f64 * pixel_size,
        origin[1] - win.row_off as f64 * pixel_size,
    ];
    let dst_origin = [grid.bounds[0], grid.bounds[3]];
    let t_resample = std::time::Instant::now();
    let out = resample_u16_to_u16(
        &src_u16,
        (win.width, win.height),
        level_origin,
        pixel_size,
        (grid.width, grid.height),
        dst_origin,
        grid.resolution,
    )?;
    let dt_resample = t_resample.elapsed().as_secs_f64();
    tracing::debug!(
        band = %band_asset, n_tiles, dt_open, dt_read, dt_stitch, dt_resample,
        "fetch_band timing"
    );
    Ok(Some(out))
}

/// SCL quality at full resolution for fetch-time best-pixel scoring.
pub async fn fetch_quality(
    http: Arc<reqwest::Client>,
    scene: &StacItem,
    grid: &GridSpec,
    semaphore: Arc<tokio::sync::Semaphore>,
    scl_asset: &str,
) -> Result<Option<Vec<u8>>> {
    let Some(url) = scene.assets.get(scl_asset).cloned() else {
        return Ok(None);
    };
    let cog = open_cog(&http, &url).await?;
    let native_res = cog
        .pixel_scale
        .map(|s| s[0])
        .unwrap_or(20.0);
    let decimation = (grid.resolution / native_res).max(1.0);
    let level = cog.level_for_decimation(decimation);
    let pixel_size = native_res * (cog.width as f64 / level.width as f64);
    let origin = cog
        .tie_point
        .map(|t| [t[3], t[4]])
        .unwrap_or([0.0, 0.0]);
    let win = match cog_window_for_utm(
        origin,
        pixel_size,
        level.width,
        level.height,
        grid.bounds,
    ) {
        Some(w) => w,
        None => return Ok(None),
    };
    let tiles = level.tiles_for_window(win);
    let url_arc = Arc::new(url);
    let tile_bytes_expected =
        (level.tile_width as usize) * (level.tile_height as usize) * cog.sample_format.bytes();
    let decoded = read_tiles(
        http.clone(),
        url_arc.clone(),
        tiles,
        level,
        tile_bytes_expected,
        semaphore.clone(),
    )
    .await?;
    let mut buf = vec![0u8; (win.width * win.height) as usize];
    stitch_tiles(&mut buf, win, level, &decoded, 1)?;
    let level_origin = [
        origin[0] + win.col_off as f64 * pixel_size,
        origin[1] - win.row_off as f64 * pixel_size,
    ];
    let dst_origin = [grid.bounds[0], grid.bounds[3]];
    let scl = resample_u8_to_u8(
        &buf,
        (win.width, win.height),
        level_origin,
        pixel_size,
        (grid.width, grid.height),
        dst_origin,
        grid.resolution,
    )?;
    Ok(Some(scl))
}

/// SCL → quality score: lower is better. Clear=0, marginal=1, dark=2,
/// everything else is nodata (max u16).
pub fn scl_to_quality(scl: &[u8]) -> Vec<u16> {
    const NODATA: u16 = 65535;
    let mut out = vec![NODATA; scl.len()];
    for (i, &v) in scl.iter().enumerate() {
        out[i] = match v {
            4 | 5 | 6 | 11 => 0,
            7 => 1,
            2 | 3 => 2,
            _ => NODATA,
        };
    }
    out
}

#[derive(Debug)]
pub struct Composite {
    pub grid: GridSpec,
    /// Per-band best-pixel surface reflectance; length = N bands, each `width*height`.
    pub bands: Vec<Vec<u16>>,
    /// Per-pixel quality of the winning observation (lower = better).
    pub quality: Vec<u16>,
    /// Per-pixel count of valid observations.
    pub observation_count: Vec<u16>,
    /// Per-pixel index into `source_ids` of the winning observation (-1 = none).
    pub selected_observation: Vec<i16>,
    /// In order they were composed; index matches `selected_observation`.
    pub source_ids: Vec<String>,
}

/// Best-pixel compose across all selected observations.
/// The compose step is ~0.3 s on the verification AOI; keep it
/// sequential for clarity. Quality mapping (the only branch-heavy
/// part) is precomputed in parallel.
pub fn compose_best_pixel(
    grid: GridSpec,
    n_bands: usize,
    observations: Vec<(String, Vec<Vec<u16>>, Vec<u8>)>,
) -> Composite {
    let n_pixels = (grid.width * grid.height) as usize;
    const NODATA: u16 = 65535;
    let mut best_data: Vec<Vec<u16>> = (0..n_bands).map(|_| vec![NODATA; n_pixels]).collect();
    let mut best_quality = vec![NODATA; n_pixels];
    let mut observation_count = vec![0u16; n_pixels];
    let mut selected_observation = vec![-1_i16; n_pixels];
    let source_ids: Vec<String> = observations.iter().map(|(id, _, _)| id.clone()).collect();
    let qualities: Vec<Vec<u16>> = observations
        .par_iter()
        .map(|(_, _, scl)| scl_to_quality(scl))
        .collect();

    for (obs_idx, (_, bands_data, _)) in observations.iter().enumerate() {
        let quality = &qualities[obs_idx];
        let obs_i16 = obs_idx as i16;
        for p in 0..n_pixels {
            let q = quality[p];
            if q == NODATA {
                continue;
            }
            observation_count[p] = observation_count[p].saturating_add(1);
            if q < best_quality[p] {
                best_quality[p] = q;
                selected_observation[p] = obs_i16;
                for (b, band) in bands_data.iter().enumerate() {
                    best_data[b][p] = band[p];
                }
            }
        }
    }
    Composite {
        grid,
        bands: best_data,
        quality: best_quality,
        observation_count,
        selected_observation,
        source_ids,
    }
}

/// Convenience helper to track per-phase wall times.
#[derive(Debug, Default, Clone)]
pub struct Timing {
    pub list_scenes: f64,
    pub scout: f64,
    pub fetch: f64,
    pub compose: f64,
    pub write: f64,
    pub total: f64,
}

impl Timing {
    pub fn checkpoint(start: Instant) -> (f64, Instant) {
        let now = Instant::now();
        (now.duration_since(start).as_secs_f64(), now)
    }
}
