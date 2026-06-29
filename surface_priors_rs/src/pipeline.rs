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
use crate::endpoint::QualityKind;
use crate::grid::{
    cog_window_for_utm, reproject_u16_to_u16, reproject_u8_to_u8, resample_u16_to_u16,
    resample_u8_to_u8, GridSpec,
};
use crate::stac::StacItem;

/// Per-(scene, chunk) statistics; cached on disk via [`crate::disk_cache`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneChunkStat {
    pub chunk_id: u32,
    pub usable_fraction: f32,
    pub mean_clear: f32,
    /// Coarse clear-cell bitmap for this chunk (empty if not observed).
    #[serde(default)]
    pub clear_mask: Vec<u8>,
    /// Coarse observed-cell bitmap for this chunk (empty if not observed).
    #[serde(default)]
    pub valid_mask: Vec<u8>,
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
    /// Per-chunk clear fraction (clear pixels / total pixels in chunk),
    /// indexed by `chunk_id` (row-major 512-px blocks, matching
    /// `chunks_from_grid`). Empty when the scene misses the AOI. Used for
    /// `usable_fraction` and the legacy top-k selectors.
    pub chunk_clear: Vec<f32>,
    /// Per-chunk coarse clear/observed mask, indexed by `chunk_id` like
    /// `chunk_clear`. Drives mask-based adaptive depth: selection stacks
    /// scenes until the union of clear cells covers the chunk.
    pub chunk_masks: Vec<ChunkMask>,
}

/// Reconstruct per-scene stats (incl. per-chunk clear fractions) from the
/// disk cache's per-chunk records. Inverse of [`scenestats_to_cache`].
pub fn scenestats_from_cache(item_id: &str, cached: &[SceneChunkStat]) -> SceneStats {
    let n = cached.iter().map(|s| s.chunk_id as usize + 1).max().unwrap_or(0);
    let mut chunk_clear = vec![0.0f32; n];
    let mut chunk_masks = vec![ChunkMask::default(); n];
    let mut mean_clear = f32::NAN;
    for s in cached {
        let c = s.chunk_id as usize;
        chunk_clear[c] = s.usable_fraction;
        chunk_masks[c] = ChunkMask { clear: s.clear_mask.clone(), valid: s.valid_mask.clone() };
        if s.mean_clear.is_finite() && !mean_clear.is_finite() {
            mean_clear = s.mean_clear;
        }
    }
    let usable = chunk_clear.iter().copied().fold(0.0f32, f32::max);
    SceneStats {
        item_id: item_id.to_string(),
        usable_fraction: usable,
        mean_clear,
        chunk_clear,
        chunk_masks,
    }
}

/// Per-chunk records to persist for one scene's scout result. Inverse of
/// [`scenestats_from_cache`].
pub fn scenestats_to_cache(s: &SceneStats) -> Vec<SceneChunkStat> {
    s.chunk_clear
        .iter()
        .enumerate()
        .map(|(c, &f)| {
            let m = s.chunk_masks.get(c).cloned().unwrap_or_default();
            SceneChunkStat {
                chunk_id: c as u32,
                usable_fraction: f,
                mean_clear: s.mean_clear,
                clear_mask: m.clear,
                valid_mask: m.valid,
            }
        })
        .collect()
}

/// Chunk edge length in pixels. Must match the value passed to
/// `chunks_from_grid` / `build_partition` in the build pipeline so that
/// scout's per-chunk binning lines up with the partition's chunk ids.
pub const SELECT_CHUNK_SIZE: u32 = 512;

/// Scout a single scene: open its quality COG, read at coarse
/// resolution, compute per-AOI usable_fraction + mean_clear. The
/// `quality_kind` selects how each pixel is interpreted (SCL classes
/// for S2 L2A, Fmask bit-flags for HLS).
pub async fn scout_scene(
    http: Arc<reqwest::Client>,
    scene: &StacItem,
    grid: &GridSpec,
    coarse_resolution: f64,
    semaphore: Arc<tokio::sync::Semaphore>,
    quality_asset: &str,
    quality_kind: QualityKind,
    source_proj: Option<&str>,
) -> Result<SceneStats> {
    let url = scene
        .assets
        .get(quality_asset)
        .ok_or_else(|| anyhow::anyhow!("scene {} missing {} asset", scene.id, quality_asset))?
        .clone();
    let cog = open_cog(&http, &url)
        .await
        .with_context(|| format!("open quality {url}"))?;
    // Native quality pixel size: SCL is 20 m, Fmask is 30 m; pulled
    // from the COG tags so the math is endpoint-agnostic.
    let native_res = cog
        .pixel_scale
        .map(|s| s[0])
        .unwrap_or(20.0);
    let decimation = (coarse_resolution / native_res).max(1.0);
    let level = cog.level_for_decimation(decimation);
    let pixel_size = native_res * (cog.width as f64 / level.width as f64);
    let origin = cog
        .tie_point
        .map(|t| [t[3], t[4]])
        .unwrap_or([0.0, 0.0]);
    let grid_proj = grid.proj_def();
    let cross_crs = source_proj.is_some() && source_proj.unwrap() != grid_proj;
    let source_bounds = if cross_crs {
        crate::projx::transform_bounds(&grid_proj, source_proj.unwrap(), grid.bounds, 21)
            .map_err(|e| anyhow::anyhow!("grid->source bounds transform: {e}"))?
    } else {
        grid.bounds
    };
    let win = match cog_window_for_utm(
        origin,
        pixel_size,
        level.width,
        level.height,
        source_bounds,
    ) {
        Some(w) => w,
        None => {
            return Ok(SceneStats {
                item_id: scene.id.clone(),
                usable_fraction: 0.0,
                mean_clear: f32::NAN,
                chunk_clear: Vec::new(),
                chunk_masks: Vec::new(),
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
    let quality_buf = if cross_crs {
        reproject_u8_to_u8(
            &buf,
            (win.width, win.height),
            level_origin,
            pixel_size,
            source_proj.unwrap(),
            (dst_w, dst_h),
            dst_origin,
            grid.resolution,
            &grid_proj,
            quality_kind.nodata_fill(),
        )?
    } else {
        resample_u8_to_u8(
            &buf,
            (win.width, win.height),
            level_origin,
            pixel_size,
            (dst_w, dst_h),
            dst_origin,
            grid.resolution,
            quality_kind.nodata_fill(),
        )?
    };
    let (usable, mean_clear) = quality_to_stats(&quality_buf, quality_kind);
    let chunk_clear = per_chunk_clear(&quality_buf, dst_w, dst_h, SELECT_CHUNK_SIZE, quality_kind);
    let chunk_masks = per_chunk_masks(&quality_buf, dst_w, dst_h, SELECT_CHUNK_SIZE, quality_kind);
    Ok(SceneStats {
        item_id: scene.id.clone(),
        usable_fraction: usable,
        mean_clear,
        chunk_clear,
        chunk_masks,
    })
}

/// Clear fraction (clear pixels / total pixels) per chunk, binning the
/// full-grid quality buffer into row-major `chunk`-px blocks so the
/// index lines up with `chunks_from_grid` chunk ids.
pub fn per_chunk_clear(buf: &[u8], width: u32, height: u32, chunk: u32, kind: QualityKind) -> Vec<f32> {
    let (w, h, cs) = (width as usize, height as usize, chunk as usize);
    let n_cols = w.div_ceil(cs);
    let n_rows = h.div_ceil(cs);
    let mut out = vec![0.0f32; n_cols * n_rows];
    for cr in 0..n_rows {
        let r0 = cr * cs;
        let r1 = (r0 + cs).min(h);
        for cc in 0..n_cols {
            let c0 = cc * cs;
            let c1 = (c0 + cs).min(w);
            let (mut total, mut clear) = (0u32, 0u32);
            for r in r0..r1 {
                let row = &buf[r * w + c0..r * w + c1];
                for &v in row {
                    total += 1;
                    if !kind.is_nodata(v) && kind.is_clear(v) {
                        clear += 1;
                    }
                }
            }
            out[cr * n_cols + cc] = if total > 0 { clear as f32 / total as f32 } else { 0.0 };
        }
    }
    out
}

/// Side length, in cells, of the coarse per-chunk clear mask. A 512-px
/// chunk bins to a `CHUNK_MASK_DIM × CHUNK_MASK_DIM` grid; selection
/// stacks scenes until the *union* of their clear cells covers the
/// chunk's reachable area, so depth adapts to the actual cloud pattern
/// (overlapping clouds → more scenes; complementary gaps → fewer).
pub const CHUNK_MASK_DIM: usize = 32;
const CHUNK_MASK_CELLS: usize = CHUNK_MASK_DIM * CHUNK_MASK_DIM;
const CHUNK_MASK_BYTES: usize = CHUNK_MASK_CELLS / 8;

/// Coarse clear/observed bitmap for one (scene, chunk). `clear` and
/// `valid` are `CHUNK_MASK_CELLS`-bit sets (row-major cells), bit-packed.
/// Both empty when the scene doesn't observe the chunk — that keeps the
/// scout cache lean for scenes that only clip a corner of the AOI.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChunkMask {
    pub clear: Vec<u8>,
    pub valid: Vec<u8>,
}

impl ChunkMask {
    fn observes(&self) -> bool {
        !self.valid.is_empty()
    }
}

#[inline]
fn mask_set(bits: &mut [u8], cell: usize) {
    bits[cell / 8] |= 1 << (cell % 8);
}

#[inline]
fn mask_popcount(bits: &[u8]) -> u32 {
    bits.iter().map(|b| b.count_ones()).sum()
}

/// Per-chunk coarse clear/observed masks, parallel to [`per_chunk_clear`]
/// (same row-major chunk order). A cell is `valid` when it covers any
/// non-nodata pixel and `clear` when the majority of its observed pixels
/// are clear — so the mask is tile-relative: a cloudless scene reads
/// ~fully clear within its footprint regardless of how much of the chunk
/// its tile covers.
pub fn per_chunk_masks(buf: &[u8], width: u32, height: u32, chunk: u32, kind: QualityKind) -> Vec<ChunkMask> {
    let (w, h, cs) = (width as usize, height as usize, chunk as usize);
    let n_cols = w.div_ceil(cs);
    let n_rows = h.div_ceil(cs);
    let mut out = Vec::with_capacity(n_cols * n_rows);
    for cr in 0..n_rows {
        let (r0, r1) = (cr * cs, (cr * cs + cs).min(h));
        let ch = r1 - r0;
        for cc in 0..n_cols {
            let (c0, c1) = (cc * cs, (cc * cs + cs).min(w));
            let cw = c1 - c0;
            let mut clear = vec![0u8; CHUNK_MASK_BYTES];
            let mut valid = vec![0u8; CHUNK_MASK_BYTES];
            let mut any = false;
            for my in 0..CHUNK_MASK_DIM {
                let cr0 = r0 + my * ch / CHUNK_MASK_DIM;
                let cr1 = r0 + (my + 1) * ch / CHUNK_MASK_DIM;
                if cr0 >= cr1 {
                    continue;
                }
                for mx in 0..CHUNK_MASK_DIM {
                    let cc0 = c0 + mx * cw / CHUNK_MASK_DIM;
                    let cc1 = c0 + (mx + 1) * cw / CHUNK_MASK_DIM;
                    if cc0 >= cc1 {
                        continue;
                    }
                    let (mut tot, mut cl) = (0u32, 0u32);
                    for r in cr0..cr1 {
                        for &v in &buf[r * w + cc0..r * w + cc1] {
                            if !kind.is_nodata(v) {
                                tot += 1;
                                if kind.is_clear(v) {
                                    cl += 1;
                                }
                            }
                        }
                    }
                    if tot > 0 {
                        let cell = my * CHUNK_MASK_DIM + mx;
                        mask_set(&mut valid, cell);
                        any = true;
                        if cl * 2 >= tot {
                            mask_set(&mut clear, cell);
                        }
                    }
                }
            }
            out.push(if any { ChunkMask { clear, valid } } else { ChunkMask::default() });
        }
    }
    out
}

fn quality_to_stats(buf: &[u8], kind: QualityKind) -> (f32, f32) {
    let mut valid = 0u32;
    let mut clear = 0u32;
    for &v in buf {
        if kind.is_nodata(v) {
            continue;
        }
        valid += 1;
        if kind.is_clear(v) {
            clear += 1;
        }
    }
    let total = buf.len() as f32;
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

/// A scene selected by the adaptive policy, plus the chunk ids that
/// requested it. `chunk_ids` lets the fetch path read the scene over
/// only the windows that need it (Level 2 windowed fetch); Level 1
/// ignores it and reads the full grid.
#[derive(Debug, Clone)]
pub struct AdaptivePick<'a> {
    pub scene: &'a StacItem,
    pub clear: f32,
    pub chunk_ids: Vec<u32>,
}

/// Adaptive-depth selection driven by per-chunk scout coverage.
///
/// For each chunk, rank scenes by their clear fraction *in that chunk*
/// and add them greedily until the estimated clear coverage
/// `1 - Π(1-f_i)` reaches `coverage_target` (with at least `min_k`
/// scenes for best-pixel redundancy), capped at `max_k`. Chunks a few
/// clear scenes already cover stop early; chronically thin chunks (the
/// under-observed swath-edge corner) keep pulling scenes up to `max_k`.
/// This spends fetch depth only where coverage needs it.
///
/// Returns one entry per distinct selected scene, carrying the set of
/// chunks that requested it.
pub fn select_adaptive<'a>(
    scenes: &'a [StacItem],
    stats: &HashMap<String, SceneStats>,
    partition: Option<&crate::tile_classification::TilePartition>,
    n_chunks: usize,
    coverage_target: f32,
    min_k: usize,
    max_k: usize,
) -> Vec<AdaptivePick<'a>> {
    use std::collections::BTreeMap;
    // scene index -> chunks that selected it (BTreeMap keeps output stable).
    let mut chosen: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
    let target = coverage_target.clamp(0.0, 0.999);
    let cap = max_k.max(min_k).max(1);

    for chunk in 0..n_chunks as u32 {
        // Select PER REQUIRED TILE, not over all scenes at once. A scene's
        // clear pixels sit only in its tile's footprint, so a chunk that
        // straddles a tile seam needs scenes from *each* covering tile —
        // pooling them and stopping on aggregate clear-fraction would keep
        // picking one tile (its half looks "covered") and leave the other
        // half empty. Per-tile greedy guarantees both halves get scenes.
        let req_tiles = partition
            .and_then(|p| p.requirements.get(&chunk))
            .map(|r| r.required_tiles.as_slice())
            .filter(|t| !t.is_empty());
        match req_tiles {
            Some(tiles) => {
                for tile in tiles {
                    let cands = chunk_candidates(scenes, stats, chunk as usize, Some(tile));
                    greedy_take(chunk, &cands, target, min_k, cap, &mut chosen);
                }
            }
            // No partition / no tile geometry: treat the whole chunk as one
            // footprint (correct for single-tile AOIs).
            None => {
                let cands = chunk_candidates(scenes, stats, chunk as usize, None);
                greedy_take(chunk, &cands, target, min_k, cap, &mut chosen);
            }
        }
    }

    chosen
        .into_iter()
        .map(|(idx, chunk_ids)| AdaptivePick {
            scene: &scenes[idx],
            clear: stats
                .get(&scenes[idx].id)
                .map(|s| s.mean_clear)
                .unwrap_or(f32::NAN),
            chunk_ids,
        })
        .collect()
}

/// Candidate `(scene_idx, &chunk_mask)` for a chunk, optionally restricted
/// to one MGRS tile. Only scenes that actually observe the chunk (non-empty
/// mask) with finite stats qualify. Order follows scene index so the
/// marginal-gain greedy breaks ties deterministically.
fn chunk_candidates<'s>(
    scenes: &[StacItem],
    stats: &'s HashMap<String, SceneStats>,
    chunk: usize,
    tile: Option<&str>,
) -> Vec<(usize, &'s ChunkMask)> {
    scenes
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            if let Some(t) = tile {
                if s.mgrs_tile != t {
                    return None;
                }
            }
            let st = stats.get(&s.id)?;
            if !st.mean_clear.is_finite() {
                return None;
            }
            let m = st.chunk_masks.get(chunk)?;
            m.observes().then_some((i, m))
        })
        .collect()
}

/// Mask-based adaptive depth for one (chunk, tile) footprint. Stacks scenes
/// by greatest marginal clear-cell gain until the union of clear cells
/// covers `target` of the reachable (observed) area, holding a `min_k`
/// floor for best-pixel redundancy and a `cap` ceiling. This is where k is
/// genuinely decided from the SCL pattern: scenes whose clear regions
/// overlap add little and the chunk keeps pulling depth; scenes that fill
/// each other's gaps reach the target fast.
fn greedy_take(
    chunk: u32,
    cands: &[(usize, &ChunkMask)],
    target: f32,
    min_k: usize,
    cap: usize,
    chosen: &mut std::collections::BTreeMap<usize, Vec<u32>>,
) {
    if cands.is_empty() {
        return;
    }
    let nbytes = cands.iter().map(|(_, m)| m.valid.len()).max().unwrap_or(0);
    if nbytes == 0 {
        return;
    }
    // reachable = union of observed cells across this footprint's scenes.
    let mut reachable = vec![0u8; nbytes];
    for (_, m) in cands {
        for (b, &v) in m.valid.iter().enumerate() {
            reachable[b] |= v;
        }
    }
    let total = mask_popcount(&reachable);
    if total == 0 {
        return;
    }
    // Observation completeness per candidate: a "full observer" sees
    // (≥99% of) the chunk's reachable cells; a "partial" only clips part
    // of it (a swath/AOI edge). We prefer full observers — a chunk should
    // be built from whole, single-acquisition observations before falling
    // back to edge slivers, even when a sliver is clearer in its part.
    // SPX_FULL_PREF=0 disables the preference (all candidates treated as a
    // single tier ⇒ pure marginal-gain greedy), for A/B comparison.
    let prefer_full = std::env::var("SPX_FULL_PREF").map(|v| v != "0").unwrap_or(true);
    let is_full: Vec<bool> = cands
        .iter()
        .map(|(_, m)| {
            if !prefer_full {
                return true;
            }
            let unobserved: u32 = (0..nbytes)
                .map(|b| (reachable[b] & !m.valid.get(b).copied().unwrap_or(0)).count_ones())
                .sum();
            unobserved.saturating_mul(100) <= total
        })
        .collect();
    let mut covered = vec![0u8; nbytes];
    let mut used = vec![false; cands.len()];
    let mut taken = 0usize;
    let gain = |m: &ChunkMask, covered: &[u8]| -> u32 {
        (0..nbytes)
            .map(|b| {
                let c = m.clear.get(b).copied().unwrap_or(0);
                (c & reachable[b] & !covered[b]).count_ones()
            })
            .sum()
    };
    while taken < cap {
        if taken >= min_k && mask_popcount(&covered) as f32 / total as f32 >= target {
            break;
        }
        // Best unused full observer and best unused partial, by marginal
        // clear-cell gain.
        let mut best_full: Option<(usize, u32)> = None;
        let mut best_part: Option<(usize, u32)> = None;
        for (ci, (_, m)) in cands.iter().enumerate() {
            if used[ci] {
                continue;
            }
            let g = gain(m, &covered);
            let slot = if is_full[ci] { &mut best_full } else { &mut best_part };
            if slot.map_or(true, |(_, bg)| g > bg) {
                *slot = Some((ci, g));
            }
        }
        // Prefer a full observer that still advances coverage; only when no
        // full observer adds anything do partials fill the remaining gaps.
        // The min_k redundancy floor likewise prefers full observers.
        let pick = match (best_full, best_part) {
            (Some((cf, gf)), _) if gf > 0 => Some(cf),
            (_, Some((cp, gp))) if gp > 0 => Some(cp),
            _ if taken < min_k => best_full.or(best_part).map(|(ci, _)| ci),
            _ => None,
        };
        let Some(ci) = pick else { break };
        used[ci] = true;
        taken += 1;
        let (idx, m) = cands[ci];
        for b in 0..nbytes {
            covered[b] |= m.clear.get(b).copied().unwrap_or(0) & reachable[b];
        }
        chosen.entry(idx).or_default().push(chunk);
    }
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
    source_proj: Option<&str>,
    apply_s2_boa_offset: bool,
) -> Result<Option<Vec<u16>>> {
    let url = match scene.assets.get(band_asset) {
        Some(u) => u.clone(),
        None => return Ok(None),
    };
    let t_open = std::time::Instant::now();
    let cog = open_cog(&http, &url).await.ok();
    let dt_open = t_open.elapsed().as_secs_f64();
    let Some(cog) = cog else { return Ok(None) };
    // Accept both unsigned (S2) and signed (HLS) 16-bit reflectance.
    if !matches!(cog.sample_format, SampleFormat::UInt16 | SampleFormat::Int16) {
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
    // Cross-CRS path: source CRS (from endpoint config) differs from
    // the target grid CRS, so we need to transform the grid bounds
    // into source coords before sizing the COG window.
    let grid_proj = grid.proj_def();
    let cross_crs = source_proj.is_some() && source_proj.unwrap() != grid_proj;
    let source_bounds = if cross_crs {
        crate::projx::transform_bounds(&grid_proj, source_proj.unwrap(), grid.bounds, 21)
            .map_err(|e| anyhow::anyhow!("grid->source bounds transform: {e}"))?
    } else {
        grid.bounds
    };
    let win = match cog_window_for_utm(
        origin,
        pixel_size,
        level.width,
        level.height,
        source_bounds,
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
    // Signed sources (HLS Int16) share the same DN/10000 reflectance scale
    // as S2, so the only fix-up is reinterpreting the bits as `i16` and
    // clamping the -9999 fill / negative reflectance to 0. Pixels clamped
    // here sit under the scene's own Fmask nodata, so compose skips them.
    if cog.sample_format == SampleFormat::Int16 {
        for v in src_u16.iter_mut() {
            *v = (*v as i16).max(0) as u16;
        }
    }
    let dt_stitch = t_stitch.elapsed().as_secs_f64();
    let level_origin = [
        origin[0] + win.col_off as f64 * pixel_size,
        origin[1] - win.row_off as f64 * pixel_size,
    ];
    let dst_origin = [grid.bounds[0], grid.bounds[3]];
    let t_resample = std::time::Instant::now();
    let out = if cross_crs {
        reproject_u16_to_u16(
            &src_u16,
            (win.width, win.height),
            level_origin,
            pixel_size,
            source_proj.unwrap(),
            (grid.width, grid.height),
            dst_origin,
            grid.resolution,
            &grid_proj,
        )?
    } else {
        resample_u16_to_u16(
            &src_u16,
            (win.width, win.height),
            level_origin,
            pixel_size,
            (grid.width, grid.height),
            dst_origin,
            grid.resolution,
        )?
    };
    let dt_resample = t_resample.elapsed().as_secs_f64();
    tracing::debug!(
        band = %band_asset, n_tiles, dt_open, dt_read, dt_stitch, dt_resample,
        "fetch_band timing"
    );
    // Harmonize the Sentinel-2 N0400 BOA_ADD_OFFSET so every processing
    // baseline shares the reflectance = DN / 10000 convention. Skipped for
    // providers that already harmonize (see applies_s2_boa_offset); applying it
    // to harmonized data clamps the dark visible bands to zero.
    let offset = if apply_s2_boa_offset { s2_boa_offset(scene) } else { 0 };
    let out = if offset > 0 {
        out.into_iter().map(|v| v.saturating_sub(offset)).collect()
    } else {
        out
    };
    Ok(Some(out))
}

/// Sentinel-2 L2A processing baseline N0400 (≥ 04.00, ~2022-01-25) bakes a
/// `+1000` DN `BOA_ADD_OFFSET` into the raster: true reflectance is
/// `(DN - 1000) / 10000`, while earlier baselines use `DN / 10000`. We
/// subtract it at fetch (saturating at 0) so all baselines land on the same
/// `DN / 10000` scale and multi-year composites are comparable. Other
/// products (HLS, MCD43A4) carry no such offset → 0.
pub fn s2_boa_offset(scene: &StacItem) -> u16 {
    if scene.collection != "sentinel-2-l2a" {
        return 0;
    }
    // Prefer the explicit STAC property; fall back to the `N####` token in
    // the Sen2Cor-style id (e.g. `..._N0400_...`) for providers that omit it.
    let from_prop = scene
        .properties
        .get("s2:processing_baseline")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f32>().ok());
    let from_id = scene.id.split('_').find_map(|t| {
        t.strip_prefix('N')
            .filter(|d| d.len() == 4 && d.bytes().all(|b| b.is_ascii_digit()))
            .and_then(|d| d.parse::<f32>().ok())
            .map(|n| n / 100.0)
    });
    match from_prop.or(from_id) {
        Some(baseline) if baseline >= 4.0 => 1000,
        _ => 0,
    }
}

/// Quality raster at full resolution for fetch-time best-pixel
/// scoring. Returns the resampled u8 buffer; caller scores it via
/// `quality_to_score` using the endpoint's QualityKind.
pub async fn fetch_quality(
    http: Arc<reqwest::Client>,
    scene: &StacItem,
    grid: &GridSpec,
    semaphore: Arc<tokio::sync::Semaphore>,
    quality_asset: &str,
    quality_kind: QualityKind,
    source_proj: Option<&str>,
) -> Result<Option<Vec<u8>>> {
    let Some(url) = scene.assets.get(quality_asset).cloned() else {
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
    let grid_proj = grid.proj_def();
    let cross_crs = source_proj.is_some() && source_proj.unwrap() != grid_proj;
    let source_bounds = if cross_crs {
        crate::projx::transform_bounds(&grid_proj, source_proj.unwrap(), grid.bounds, 21)
            .map_err(|e| anyhow::anyhow!("grid->source bounds transform: {e}"))?
    } else {
        grid.bounds
    };
    let win = match cog_window_for_utm(
        origin,
        pixel_size,
        level.width,
        level.height,
        source_bounds,
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
    let scl = if cross_crs {
        reproject_u8_to_u8(
            &buf,
            (win.width, win.height),
            level_origin,
            pixel_size,
            source_proj.unwrap(),
            (grid.width, grid.height),
            dst_origin,
            grid.resolution,
            &grid_proj,
            quality_kind.nodata_fill(),
        )?
    } else {
        resample_u8_to_u8(
            &buf,
            (win.width, win.height),
            level_origin,
            pixel_size,
            (grid.width, grid.height),
            dst_origin,
            grid.resolution,
            quality_kind.nodata_fill(),
        )?
    };
    Ok(Some(scl))
}

/// Quality raster → per-pixel score: lower is better. The mapping
/// is selected by `QualityKind` so this works for both SCL (S2 L2A)
/// and Fmask (HLS).
pub fn quality_to_score(buf: &[u8], kind: QualityKind) -> Vec<u16> {
    let mut out = vec![0u16; buf.len()];
    for (i, &v) in buf.iter().enumerate() {
        out[i] = kind.score(v);
    }
    out
}

/// Back-compat alias: existing callers that fetched SCL u8 still call
/// `scl_to_quality` from outside the crate. Kept as a thin shim over
/// `quality_to_score(SCL)`.
pub fn scl_to_quality(scl: &[u8]) -> Vec<u16> {
    quality_to_score(scl, QualityKind::Scl)
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
    /// Optional per-band temporal-spread uncertainty (DN units, NaN where the
    /// winning pixel is nodata); `Some` only when `emit_uncertainty` was set.
    /// Length = N bands, each `width*height`.
    pub reflectance_std: Option<Vec<Vec<f32>>>,
}

/// Best-pixel compose across all selected observations. Quality
/// mapping is precomputed in parallel using `kind` so the same
/// composer handles both SCL and Fmask scenes.
pub fn compose_best_pixel(
    grid: &GridSpec,
    n_bands: usize,
    observations: Vec<(String, Vec<Vec<u16>>, Vec<u8>)>,
    kind: QualityKind,
    emit_uncertainty: bool,
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
        .map(|(_, _, q)| quality_to_score(q, kind))
        .collect();

    // Per-band Welford accumulators for the temporal-spread uncertainty,
    // allocated only when requested (zero overhead on the default path).
    let mut acc_count: Vec<Vec<u32>> = Vec::new();
    let mut acc_mean: Vec<Vec<f32>> = Vec::new();
    let mut acc_m2: Vec<Vec<f32>> = Vec::new();
    if emit_uncertainty {
        acc_count = (0..n_bands).map(|_| vec![0u32; n_pixels]).collect();
        acc_mean = (0..n_bands).map(|_| vec![0.0f32; n_pixels]).collect();
        acc_m2 = (0..n_bands).map(|_| vec![0.0f32; n_pixels]).collect();
    }

    for (obs_idx, (_, bands_data, _)) in observations.iter().enumerate() {
        let quality = &qualities[obs_idx];
        let obs_i16 = obs_idx as i16;
        for p in 0..n_pixels {
            let q = quality[p];
            if q == NODATA {
                continue;
            }
            observation_count[p] = observation_count[p].saturating_add(1);
            if emit_uncertainty {
                // Running per-band mean/M2 over every valid observation (not
                // just the winner) — the spread across candidate days.
                for (b, band) in bands_data.iter().enumerate() {
                    let v = band[p];
                    if v == NODATA {
                        continue;
                    }
                    let x = v as f32;
                    let c = acc_count[b][p] + 1;
                    acc_count[b][p] = c;
                    let delta = x - acc_mean[b][p];
                    acc_mean[b][p] += delta / c as f32;
                    acc_m2[b][p] += delta * (x - acc_mean[b][p]);
                }
            }
            if q < best_quality[p] {
                best_quality[p] = q;
                selected_observation[p] = obs_i16;
                for (b, band) in bands_data.iter().enumerate() {
                    best_data[b][p] = band[p];
                }
            }
        }
    }

    // Finalize per-band temporal-spread uncertainty (DN units; NaN where the
    // winning pixel is nodata). unc = sqrt(sample_var + (REL_FLOOR*boa)^2),
    // floored at ABS_FLOOR_DN so it is always finite & positive; the solver
    // re-floors in reflectance space. n < 2 falls back to the relative floor.
    let reflectance_std: Option<Vec<Vec<f32>>> = if emit_uncertainty {
        const REL_FLOOR: f32 = 0.02; // 2% of reflectance
        const ABS_FLOOR_DN: f32 = 10.0; // 0.001 reflectance (DN = refl * 10000)
        let mut std_bands: Vec<Vec<f32>> =
            (0..n_bands).map(|_| vec![f32::NAN; n_pixels]).collect();
        for b in 0..n_bands {
            for p in 0..n_pixels {
                let boa = best_data[b][p];
                if boa == NODATA {
                    continue;
                }
                let c = acc_count[b][p];
                let var = if c >= 2 {
                    acc_m2[b][p] / (c as f32 - 1.0)
                } else {
                    0.0
                };
                let rel = REL_FLOOR * boa as f32;
                std_bands[b][p] = (var + rel * rel).sqrt().max(ABS_FLOOR_DN);
            }
        }
        Some(std_bands)
    } else {
        None
    };

    Composite {
        grid: grid.clone(),
        bands: best_data,
        quality: best_quality,
        observation_count,
        selected_observation,
        source_ids,
        reflectance_std,
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

/// Optional external-aerosol (e.g. MAIAC) day-quality gate for the best-pixel
/// composite. Returns `true` if a scene acquired on `datetime` should be KEPT.
///
/// When the gate is active, a scene is KEPT iff its `"YYYY-MM-DD"` acquisition
/// day is present in `aod_by_day` with a value at or below `aod_max`. This lets
/// a caller drop atmospherically dirty days *before* fetch+compose, so the
/// composite is built only from low-AOD acquisitions — mirroring the L1C
/// low-AOD day selection. When either `aod_max` or `aod_by_day` is `None` the
/// gate is a no-op (keep everything).
///
/// Unknown days — absent from the map, or a malformed `datetime` — are governed
/// by `reject_unknown`: `false` (the default) KEEPS them (missing AOD is not
/// treated as dirty); `true` DROPS them (keep only days we can vouch for).
pub fn day_aod_passes(
    datetime: &str,
    aod_by_day: Option<&HashMap<String, f64>>,
    aod_max: Option<f64>,
    reject_unknown: bool,
) -> bool {
    match (aod_max, aod_by_day) {
        (Some(thr), Some(map)) => datetime
            .get(..10)
            .and_then(|day| map.get(day))
            .map(|&aod| aod <= thr)
            .unwrap_or(!reject_unknown),
        _ => true,
    }
}

/// Calendar-correct last day (28/29/30/31) of `month` in `year`, accounting
/// for leap years. Used to build valid STAC datetime ranges per period — a
/// hardcoded "-31" yields invalid dates for 30-day months and February, which
/// STAC endpoints reject with HTTP 400.
pub fn last_day_of_month(year: u32, month: u32) -> u32 {
    match month {
        4 | 6 | 9 | 11 => 30,
        2 => {
            let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
            if leap {
                29
            } else {
                28
            }
        }
        _ => 31, // Jan/Mar/May/Jul/Aug/Oct/Dec (and any out-of-range guard)
    }
}

#[cfg(test)]
mod last_day_tests {
    use super::last_day_of_month;

    #[test]
    fn known_month_lengths() {
        assert_eq!(last_day_of_month(2020, 1), 31);
        assert_eq!(last_day_of_month(2020, 6), 30); // June — the bug case
        assert_eq!(last_day_of_month(2020, 9), 30);
        assert_eq!(last_day_of_month(2020, 2), 29); // leap
        assert_eq!(last_day_of_month(2021, 2), 28); // non-leap
        assert_eq!(last_day_of_month(2000, 2), 29); // /400 leap
        assert_eq!(last_day_of_month(1900, 2), 28); // /100 non-leap
    }
}

#[cfg(test)]
mod compose_unc_tests {
    use super::compose_best_pixel;
    use crate::endpoint::QualityKind;
    use crate::grid::GridSpec;

    fn grid_2px() -> GridSpec {
        GridSpec {
            bounds: [0.0, 0.0, 120.0, 60.0],
            epsg: 32631,
            proj4: None,
            resolution: 60.0,
            width: 2,
            height: 1,
        }
    }

    #[test]
    fn emit_uncertainty_off_yields_no_std() {
        // SCL 4 = clear (score 0); single band, two pixels.
        let obs = vec![(
            "a".to_string(),
            vec![vec![1000u16, 500u16]],
            vec![4u8, 4u8],
        )];
        let c = compose_best_pixel(&grid_2px(), 1, obs, QualityKind::Scl, false);
        assert!(c.reflectance_std.is_none());
    }

    #[test]
    fn per_band_temporal_std_with_floor_and_single_sample() {
        // pixel0: 3 clear obs (1000,1100,1200) -> sample std 100, +2% rel floor of
        //   the winner (lowest-quality tie -> first = 1000, rel = 20) => ~101.98.
        // pixel1: only obs "a" is clear (others SCL 0 = nodata) -> n=1 -> var 0,
        //   unc = max(rel_floor*500=10, abs_floor 10) = 10.
        let obs = vec![
            ("a".to_string(), vec![vec![1000u16, 500u16]], vec![4u8, 4u8]),
            ("b".to_string(), vec![vec![1100u16, 65535u16]], vec![4u8, 0u8]),
            ("c".to_string(), vec![vec![1200u16, 65535u16]], vec![4u8, 0u8]),
        ];
        let c = compose_best_pixel(&grid_2px(), 1, obs, QualityKind::Scl, true);
        let std = c.reflectance_std.expect("std present");
        assert_eq!(c.observation_count[0], 3);
        assert_eq!(c.observation_count[1], 1);
        assert!((std[0][0] - 101.98).abs() < 1.0, "pixel0 std = {}", std[0][0]);
        assert!((std[0][1] - 10.0).abs() < 0.5, "pixel1 std = {}", std[0][1]);
    }

    #[test]
    fn nodata_pixel_std_is_nan() {
        // both observations nodata at pixel1 -> winner nodata -> std NaN there.
        let obs = vec![
            ("a".to_string(), vec![vec![1000u16, 65535u16]], vec![4u8, 0u8]),
            ("b".to_string(), vec![vec![1100u16, 65535u16]], vec![4u8, 0u8]),
        ];
        let c = compose_best_pixel(&grid_2px(), 1, obs, QualityKind::Scl, true);
        let std = c.reflectance_std.unwrap();
        assert!(std[0][0].is_finite());
        assert!(std[0][1].is_nan());
    }
}

#[cfg(test)]
mod aod_gate_tests {
    use super::day_aod_passes;
    use std::collections::HashMap;

    fn map() -> HashMap<String, f64> {
        HashMap::from([
            ("2020-06-12".to_string(), 0.08), // clean
            ("2020-06-19".to_string(), 0.55), // dirty
        ])
    }

    #[test]
    fn keeps_clean_day_rejects_dirty_day() {
        let m = map();
        assert!(day_aod_passes("2020-06-12T10:30:00Z", Some(&m), Some(0.3), false));
        assert!(!day_aod_passes("2020-06-19T10:30:00Z", Some(&m), Some(0.3), false));
    }

    #[test]
    fn threshold_is_inclusive() {
        let m = HashMap::from([("2020-06-12".to_string(), 0.30)]);
        // aod == threshold is kept (<=), strictly-above is rejected.
        assert!(day_aod_passes("2020-06-12T00:00:00Z", Some(&m), Some(0.30), false));
        assert!(!day_aod_passes("2020-06-12T00:00:00Z", Some(&m), Some(0.29), false));
    }

    #[test]
    fn unknown_day_kept_by_default_rejected_when_opted_in() {
        let m = map();
        // default: unknown day kept
        assert!(day_aod_passes("2021-01-01T10:30:00Z", Some(&m), Some(0.1), false));
        // reject_unknown: unknown day dropped
        assert!(!day_aod_passes("2021-01-01T10:30:00Z", Some(&m), Some(0.1), true));
        // a KNOWN clean day still passes even with reject_unknown
        assert!(day_aod_passes("2020-06-12T10:30:00Z", Some(&m), Some(0.3), true));
    }

    #[test]
    fn no_op_when_either_arg_missing() {
        let m = map();
        // gate inactive -> reject_unknown is irrelevant, keep everything
        assert!(day_aod_passes("2020-06-19T10:30:00Z", Some(&m), None, true));
        assert!(day_aod_passes("2020-06-19T10:30:00Z", None, Some(0.1), true));
    }

    #[test]
    fn malformed_datetime_follows_reject_unknown() {
        let m = map();
        assert!(day_aod_passes("bad", Some(&m), Some(0.1), false));
        assert!(!day_aod_passes("bad", Some(&m), Some(0.1), true));
    }
}
