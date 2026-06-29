//! PyO3 bindings exposing the composite pipeline to Python.
//!
//! Python callers get a sync function `build_composite(...)` that
//! returns a dict with numpy arrays for each band + auxiliary buffers
//! and a `grid` description. No file I/O — arrays are handed straight
//! to downstream Python processes (numpy, xarray, rasterio.MemoryFile,
//! etc.) without ever touching disk.
//!
//! Internally the function spins up a tokio runtime, runs the full
//! Rust pipeline (STAC search → scout → select → fetch → compose),
//! and converts result buffers to `numpy.ndarray` via the `numpy`
//! crate's zero-copy `IntoPyArray`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tokio::runtime::Runtime;

/// Process-global tokio runtime. Rebuilding the runtime per call costs
/// ~3 s of drop overhead because reqwest's idle connection pool has
/// to close 256+ TLS sessions; reusing one runtime across all
/// build_composite() calls avoids that entirely. Block_on is safe to
/// call from multiple Python threads on the same runtime — the worker
/// pool services all parallel futures.
fn shared_runtime() -> &'static Runtime {
    static RT: OnceLock<Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);
        if cpus <= 1 {
            // current_thread runtime still works with block_on from
            // the same OS thread, but multiple Python threads calling
            // build_composite would serialize. Single-CPU runs are
            // unusual so we accept that.
            Runtime::new().expect("tokio runtime")
        } else {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(cpus.max(2))
                .enable_all()
                .build()
                .expect("tokio runtime")
        }
    })
}

/// Process-global reqwest::Client. Each call previously built a fresh
/// Client with `pool_max_idle_per_host(256)`; reusing one client keeps
/// HTTP/2 + TLS connections warm across calls and across STAC + tile
/// + SAS endpoints alike.
fn shared_http() -> Arc<reqwest::Client> {
    static HTTP: OnceLock<Arc<reqwest::Client>> = OnceLock::new();
    HTTP.get_or_init(|| {
        Arc::new(
            reqwest::Client::builder()
                .gzip(true)
                .http2_adaptive_window(true)
                .pool_max_idle_per_host(256)
                .tcp_keepalive(std::time::Duration::from_secs(60))
                .tcp_nodelay(true)
                .build()
                .expect("reqwest client"),
        )
    })
    .clone()
}

/// Process-global EndpointConfig per kind. Caches PC SAS tokens across
/// calls so the second call doesn't re-fetch them. EndpointConfig is
/// internally Send + Sync; its sas_tokens_per_container is a RwLock so
/// concurrent calls from multiple Python threads share the cache
/// without contention.
fn shared_endpoint(kind: EndpointKind) -> Arc<EndpointConfig> {
    use std::sync::Mutex;
    static CACHE: OnceLock<Mutex<HashMap<EndpointKind, Arc<EndpointConfig>>>> =
        OnceLock::new();
    let map = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = map.lock().expect("endpoint cache poisoned");
    guard
        .entry(kind)
        .or_insert_with(|| Arc::new(EndpointConfig::build(kind)))
        .clone()
}

use crate::disk_cache::{grid_signature, DiskCache};
use crate::endpoint::{auto_pick, EndpointConfig, EndpointKind};

/// Build the output grid based on the caller's `output_crs` request
/// and the endpoint's native source CRS.
///   - "native": match the endpoint's source CRS (UTM for S2/HLS,
///     MODIS Sinusoidal for MCD43A4).
///   - "utm": always derive a UTM zone from the AOI centroid, even
///     when the endpoint is MCD43A4 (requires cross-CRS reprojection
///     in the fetch path).
fn choose_grid(
    endpoint: &EndpointConfig,
    bbox: [f64; 4],
    resolution: f64,
    output_crs: &str,
) -> anyhow::Result<GridSpec> {
    // HLS georeferences every tile in the northern UTM zone (false_northing=0);
    // the grid must follow suit or southern AOIs read empty. See
    // EndpointConfig::uses_north_utm_convention.
    let utm_grid = |bbox: [f64; 4], resolution: f64| {
        if endpoint.uses_north_utm_convention() {
            GridSpec::from_wgs84_bounds_force_north(bbox, resolution)
        } else {
            GridSpec::from_wgs84_bounds(bbox, resolution)
        }
    };
    match output_crs {
        "native" => {
            if endpoint.uses_modis_sinusoidal() {
                Ok(GridSpec::from_wgs84_bounds_modis_sinu(bbox, resolution))
            } else {
                Ok(utm_grid(bbox, resolution))
            }
        }
        "utm" => Ok(utm_grid(bbox, resolution)),
        other => anyhow::bail!(
            "unsupported output_crs {other:?}; valid: native, utm"
        ),
    }
}

/// The native source CRS of one scene's COGs, used by scout/fetch to decide
/// whether (and from where) to reproject onto the AOI grid.
///
/// MODIS is a single sinusoidal grid shared by every scene. S2/HLS COGs each
/// sit in their own UTM zone, so the CRS is derived *per scene* from its MGRS
/// tile — the fix for AOIs on a 6° UTM seam: the grid is one zone but scenes
/// arrive from both, and a single shared source CRS made the neighbouring
/// zone's tiles read in the wrong zone (empty window → 0 usable → dropped,
/// so `n_src=0`). scout_scene/fetch_band still no-op the reprojection when a
/// scene's zone equals the grid zone (`source_proj == grid_proj`), so
/// single-zone AOIs are byte-for-byte unchanged. Returns None only when the
/// tile code is absent/unparseable, restoring the legacy same-CRS path.
fn scene_source_proj(endpoint: &EndpointConfig, scene: &crate::stac::StacItem) -> Option<String> {
    if let Some(sp) = endpoint.source_proj() {
        return Some(sp.to_string());
    }
    crate::projx::epsg_from_mgrs_tile(&scene.mgrs_tile, endpoint.uses_north_utm_convention())
        .map(|e| format!("EPSG:{e}"))
}
use crate::grid::GridSpec;
use crate::pipeline::{
    compose_best_pixel, fetch_band, fetch_quality, scenestats_from_cache, scenestats_to_cache,
    scout_scene, select_adaptive, select_chunk_tile_aware, select_top_k, SELECT_CHUNK_SIZE,
};
use crate::stac::StacClient;
use crate::tile_classification::{build_partition, chunks_from_grid, scenes_signature};

#[pyfunction]
#[pyo3(signature = (
    bbox,
    datetime,
    resolution = 60.0,
    top_k = 3,
    max_cloud_cover = 90.0,
    concurrency = 192,
    endpoint = "auto".to_string(),
    disk_cache = None,
    scout_factor = 8,
    bands = None,
    output_crs = "native".to_string(),
    coverage_target = 0.0,
    min_k = 2,
    max_k = 8,
    windowed_fetch = false,
))]
#[allow(clippy::too_many_arguments)]
fn build_composite(
    py: Python<'_>,
    bbox: [f64; 4],
    datetime: String,
    resolution: f64,
    top_k: usize,
    max_cloud_cover: f64,
    concurrency: usize,
    endpoint: String,
    disk_cache: Option<String>,
    scout_factor: u32,
    bands: Option<Vec<String>>,
    output_crs: String,
    coverage_target: f32,
    min_k: usize,
    max_k: usize,
    windowed_fetch: bool,
) -> PyResult<Bound<'_, PyDict>> {
    let sel = SelectParams { top_k, coverage_target, min_k, max_k, windowed: windowed_fetch };
    // Release the GIL while the heavy async work runs — lets concurrent
    // Python threads do other things even though we block on tokio.
    let result = py
        .allow_threads(|| {
            run_build(
                bbox, datetime, resolution, sel, max_cloud_cover, concurrency, endpoint,
                disk_cache, scout_factor, bands, output_crs,
            )
        })
        .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
    encode_result(py, result)
}

#[pyfunction]
#[pyo3(signature = (
    bbox,
    years,
    months,
    resolution = 60.0,
    top_k = 3,
    max_cloud_cover = 90.0,
    concurrency = 192,
    endpoint = "auto".to_string(),
    disk_cache = None,
    scout_factor = 8,
    bands = None,
    output_crs = "native".to_string(),
    coverage_target = 0.0,
    min_k = 2,
    max_k = 8,
    windowed_fetch = false,
    aod_by_day = None,
    aod_max = None,
    reject_unknown = false,
))]
#[allow(clippy::too_many_arguments)]
fn build_monthly_composites(
    py: Python<'_>,
    bbox: [f64; 4],
    years: Vec<u32>,
    months: Vec<u32>,
    resolution: f64,
    top_k: usize,
    max_cloud_cover: f64,
    concurrency: usize,
    endpoint: String,
    disk_cache: Option<String>,
    scout_factor: u32,
    bands: Option<Vec<String>>,
    output_crs: String,
    coverage_target: f32,
    min_k: usize,
    max_k: usize,
    windowed_fetch: bool,
    // Optional MAIAC-style aerosol pre-filter: keyed by "YYYY-MM-DD" acquisition
    // day -> AOD. Scenes whose day exceeds `aod_max` are dropped before
    // fetch+compose, so the composite is built only from low-AOD days.
    aod_by_day: Option<HashMap<String, f64>>,
    aod_max: Option<f64>,
    // When the gate is active, also drop scenes whose day is ABSENT from
    // `aod_by_day` (default false = keep unknown days).
    reject_unknown: bool,
) -> PyResult<Bound<'_, PyList>> {
    if years.is_empty() {
        return Err(PyRuntimeError::new_err("years must be non-empty"));
    }
    if months.is_empty() {
        return Err(PyRuntimeError::new_err("months must be non-empty"));
    }
    for &m in &months {
        if !(1..=12).contains(&m) {
            return Err(PyRuntimeError::new_err(format!(
                "month {m} out of range 1..=12"
            )));
        }
    }
    let sel = SelectParams { top_k, coverage_target, min_k, max_k, windowed: windowed_fetch };
    let results = py
        .allow_threads(|| {
            run_build_periods(
                bbox, years, months, resolution, sel, max_cloud_cover, concurrency,
                endpoint, disk_cache, scout_factor, bands, output_crs, aod_by_day, aod_max,
                reject_unknown,
            )
        })
        .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
    encode_period_results(py, results)
}

struct PeriodResult {
    year: u32,
    month: u32,
    build: BuildResult,
}

struct BuildResult {
    grid: GridSpec,
    bands: Vec<Vec<u16>>,
    /// Stable band names matching `bands` element-for-element.
    band_names: Vec<String>,
    quality: Vec<u16>,
    observation_count: Vec<u16>,
    selected_observation: Vec<i16>,
    source_ids: Vec<String>,
    timings: HashMap<String, f64>,
    endpoint_url: String,
    collection: String,
    partition_tiles: Vec<String>,
    multi_tile_chunks: usize,
}

/// Scene-selection knobs threaded from the Python signature down to the
/// per-period compositor.
#[derive(Debug, Clone, Copy)]
struct SelectParams {
    /// Fixed scenes-per-chunk used when `coverage_target <= 0`.
    top_k: usize,
    /// Adaptive coverage target in (0,1). `<= 0` keeps the legacy fixed
    /// `top_k` selector.
    coverage_target: f32,
    /// Floor on scenes per chunk under adaptive selection (best-pixel
    /// redundancy).
    min_k: usize,
    /// Cap on scenes per chunk under adaptive selection.
    max_k: usize,
    /// Level 2: read each scene only over the window of the chunks that
    /// requested it, instead of the full grid.
    windowed: bool,
}

impl SelectParams {
    fn adaptive(&self) -> bool {
        self.coverage_target > 0.0
    }
}

/// Pixel-space window (Level 2 windowed fetch): the bounding box, in
/// full-grid pixels, of the chunks that selected a scene.
#[derive(Debug, Clone, Copy)]
struct FetchWin {
    col0: u32,
    row0: u32,
    w: u32,
    h: u32,
}

/// Bounding pixel window of a set of chunk ids (row-major
/// `SELECT_CHUNK_SIZE` blocks over the grid).
fn chunks_to_window(chunk_ids: &[u32], grid: &GridSpec) -> Option<FetchWin> {
    if chunk_ids.is_empty() {
        return None;
    }
    let cs = SELECT_CHUNK_SIZE;
    let n_cols = grid.width.div_ceil(cs);
    let (mut c0, mut r0, mut c1, mut r1) = (u32::MAX, u32::MAX, 0u32, 0u32);
    for &cid in chunk_ids {
        let cc = cid % n_cols;
        let cr = cid / n_cols;
        c0 = c0.min(cc * cs);
        r0 = r0.min(cr * cs);
        c1 = c1.max(((cc + 1) * cs).min(grid.width));
        r1 = r1.max(((cr + 1) * cs).min(grid.height));
    }
    Some(FetchWin { col0: c0, row0: r0, w: c1 - c0, h: r1 - r0 })
}

/// A sub-grid covering `win`, pixel-aligned to `grid` (same CRS /
/// resolution / origin offset by an integer number of pixels), so a
/// windowed read scatters back exactly.
fn subgrid(grid: &GridSpec, win: &FetchWin) -> GridSpec {
    let xmin = grid.bounds[0] + win.col0 as f64 * grid.resolution;
    let ymax = grid.bounds[3] - win.row0 as f64 * grid.resolution;
    GridSpec {
        bounds: [xmin, ymax - win.h as f64 * grid.resolution, xmin + win.w as f64 * grid.resolution, ymax],
        epsg: grid.epsg,
        proj4: grid.proj4.clone(),
        resolution: grid.resolution,
        width: win.w,
        height: win.h,
    }
}

/// Scatter a window-sized buffer into a full-grid buffer (the rest left
/// as `fill`).
fn scatter<T: Copy>(sub: &[T], win: &FetchWin, grid_w: usize, grid_h: usize, fill: T) -> Vec<T> {
    let mut full = vec![fill; grid_w * grid_h];
    let ww = win.w as usize;
    for r in 0..win.h as usize {
        let d0 = (win.row0 as usize + r) * grid_w + win.col0 as usize;
        full[d0..d0 + ww].copy_from_slice(&sub[r * ww..r * ww + ww]);
    }
    full
}

fn run_build(
    bbox: [f64; 4],
    datetime: String,
    resolution: f64,
    sel: SelectParams,
    max_cloud_cover: f64,
    concurrency: usize,
    endpoint: String,
    disk_cache: Option<String>,
    scout_factor: u32,
    bands_subset: Option<Vec<String>>,
    output_crs: String,
) -> anyhow::Result<BuildResult> {
    use anyhow::Context;
    use futures::stream::{FuturesUnordered, StreamExt};
    use std::time::Instant;

    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);
    // Cap by CPUs as a floor of safety; the request concurrency default
    // (192) was tuned against Planetary Computer: a 5-year Egypt batch ran
    // ~13s at 192 vs ~23s at 600 — oversubscribing the blob endpoint gives
    // no extra throughput and trips its 429 rate-limit (→ backoff).
    let safe_concurrency = (cpus.saturating_mul(50)).max(50);
    let effective_concurrency = concurrency.min(safe_concurrency);

    // Reuse the shared runtime — see shared_runtime() for why.
    let rt = shared_runtime();
    rt.block_on(async move {
        let endpoint_kind = if endpoint == "auto" {
            auto_pick()
        } else {
            EndpointKind::parse(&endpoint)?
        };
        // Both endpoint config (SAS token cache) and reqwest client
        // (connection pool) are shared across all calls in this
        // process — see shared_endpoint() / shared_http().
        let endpoint = shared_endpoint(endpoint_kind);
        let grid = choose_grid(&endpoint, bbox, resolution, &output_crs)?;
        let apply_s2_offset = endpoint.applies_s2_boa_offset();

        let http = shared_http();
        let request_semaphore =
            Arc::new(tokio::sync::Semaphore::new(effective_concurrency.max(1)));
        let cache = disk_cache.as_ref().map(|p| DiskCache::new(PathBuf::from(p)));
        // Hook the same disk cache into the COG header reader so the
        // 64 KiB header GET is skipped on warm runs.
        crate::cog::set_cog_disk_cache(cache.clone());

        let mut timings: HashMap<String, f64> = HashMap::new();
        let t_total = Instant::now();

        // 1. list_scenes
        let t = Instant::now();
        let collections_key = endpoint.collections_key();
        // Cloud-cover filter is only meaningful for S2-style scenes.
        // MCD43A4 has already filtered clouds in the BRDF inversion,
        // so its STAC items don't carry `eo:cloud_cover`.
        let cloud_cover_filter = if endpoint.supports_cloud_cover_filter() {
            Some(max_cloud_cover)
        } else {
            None
        };
        let stac = StacClient::new(
            &endpoint.stac_url,
            endpoint.collections.clone(),
            bbox,
            datetime.clone(),
            cloud_cover_filter,
        )?;
        let raw_items = if let Some(c) = &cache {
            let key = c.search_key(
                &endpoint.stac_url,
                &collections_key,
                bbox,
                &datetime,
                cloud_cover_filter,
            );
            match c.load_search(&key)? {
                Some(items) => items,
                None => {
                    let fresh = stac.search_raw().await.context("STAC search")?;
                    c.store_search(&key, &fresh)?;
                    fresh
                }
            }
        } else {
            stac.search_raw().await.context("STAC search")?
        };
        let t_list = t.elapsed().as_secs_f64();
        // Sign for PC / HLS if needed (anonymous endpoints pass through
        // sign_item unchanged).
        let t_sign = Instant::now();
        let signed_items: Vec<serde_json::Value> = {
            let mut signed = Vec::with_capacity(raw_items.len());
            for raw in raw_items {
                signed.push(endpoint.sign_item(&http, raw).await?);
            }
            signed
        };
        let items = StacClient::items_from_raw(signed_items);
        timings.insert("list_scenes".to_string(), t_list);
        timings.insert("sign".to_string(), t_sign.elapsed().as_secs_f64());

        // 2. scout
        let t = Instant::now();
        let coarse_resolution = resolution * scout_factor as f64;
        let mut stats_map: HashMap<String, crate::pipeline::SceneStats> = HashMap::new();
        let mut to_scout: Vec<crate::stac::StacItem> = Vec::new();
        let gsig =
            grid_signature(grid.bounds, grid.epsg, grid.resolution, grid.width, grid.height);
        if let Some(c) = &cache {
            for item in &items {
                let key = c.scout_key(
                    &endpoint.stac_url,
                    &collections_key,
                    &item.id,
                    &gsig,
                    512,
                    scout_factor,
                );
                if let Some(cached) = c.load_scout(&key)? {
                    stats_map.insert(item.id.clone(), scenestats_from_cache(&item.id, &cached));
                    continue;
                }
                to_scout.push(item.clone());
            }
        } else {
            to_scout.extend(items.iter().cloned());
        }
        let mut tasks = FuturesUnordered::new();
        let quality_kind = endpoint.quality_kind;
        for item in to_scout {
            let http = http.clone();
            let grid = grid.clone();
            let sem = request_semaphore.clone();
            // Each scene resolves to its own per-collection quality asset
            // (SCL for S2 L2A scenes, Fmask for HLS L30/S30 scenes).
            let quality_asset = match endpoint.quality_asset_for(&item.collection) {
                Some(a) => a.to_string(),
                None => {
                    tracing::warn!(
                        scene = %item.id,
                        collection = %item.collection,
                        "no quality asset for this collection; skipping scout"
                    );
                    continue;
                }
            };
            let scene_src = scene_source_proj(&endpoint, &item);
            tasks.push(tokio::spawn(async move {
                scout_scene(
                    http,
                    &item,
                    &grid,
                    coarse_resolution,
                    sem,
                    &quality_asset,
                    quality_kind,
                    scene_src.as_deref(),
                )
                .await
            }));
        }
        let scout_cache = cache.clone();
        while let Some(res) = tasks.next().await {
            match res {
                Ok(Ok(s)) => {
                    if let Some(c) = &scout_cache {
                        let key = c.scout_key(
                            &endpoint.stac_url,
                            &collections_key,
                            &s.item_id,
                            &gsig,
                            512,
                            scout_factor,
                        );
                        let _ = c.store_scout(&key, &scenestats_to_cache(&s));
                    }
                    stats_map.insert(s.item_id.clone(), s);
                }
                Ok(Err(e)) => eprintln!("scout error: {e}"),
                Err(e) => eprintln!("scout task panic: {e}"),
            }
        }
        timings.insert("scout".to_string(), t.elapsed().as_secs_f64());

        // 3. tile partition + select
        let t_part = Instant::now();
        let chunks = chunks_from_grid(
            grid.bounds,
            grid.resolution,
            (grid.width, grid.height),
            SELECT_CHUNK_SIZE,
        );
        let scene_geoms: Vec<(usize, String, serde_json::Value)> = items
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.mgrs_tile.is_empty() && !s.geometry.is_null())
            .map(|(idx, s)| (idx, s.mgrs_tile.clone(), s.geometry.clone()))
            .collect();
        let partition = build_partition(
            &chunks,
            &grid.proj_def(),
            &scene_geoms,
            1,
            (grid.resolution * grid.resolution) as f64,
        )?;
        let (scenes_for_fetch, windows): (Vec<crate::stac::StacItem>, Vec<Option<FetchWin>>) =
            if sel.adaptive() {
                select_adaptive(
                    &items,
                    &stats_map,
                    partition.as_ref(),
                    chunks.len(),
                    sel.coverage_target,
                    sel.min_k,
                    sel.max_k,
                )
                .into_iter()
                .map(|p| {
                    // Level 2: read each scene only over the bounding window of
                    // the chunks that selected it. Level 1 (windowed=false) reads
                    // the full grid for every pick.
                    let win = if sel.windowed { chunks_to_window(&p.chunk_ids, &grid) } else { None };
                    (p.scene.clone(), win)
                })
                .unzip()
            } else {
                let picks = if let Some(p) = &partition {
                    select_chunk_tile_aware(&items, &stats_map, p, sel.top_k)
                } else {
                    select_top_k(&items, &stats_map, sel.top_k)
                };
                picks.iter().map(|p| (p.scene.clone(), None)).unzip()
            };
        // Per-scene read grid: the sub-window when windowed, else the full grid.
        let scene_grids: Vec<GridSpec> = windows
            .iter()
            .map(|w| w.map(|win| subgrid(&grid, &win)).unwrap_or_else(|| grid.clone()))
            .collect();
        // Deterministic proxy for fetch volume: total source pixels read across
        // scenes (bytes ∝ this × bands). Windowed reads shrink it for the
        // marginal scenes that only a few chunks requested.
        let read_megapixels: f64 = scene_grids
            .iter()
            .map(|g| g.width as f64 * g.height as f64)
            .sum::<f64>()
            / 1e6;
        timings.insert("partition".to_string(), t_part.elapsed().as_secs_f64());

        // 4. fetch
        let t = Instant::now();
        // Resolve the band subset (if any) against the endpoint's
        // supported band set, preserving caller-requested order. If no
        // subset is given we default to whatever the endpoint exposes
        // (12 bands for S2 L2A, 7 harmonized bands for HLS).
        let band_names_out: Vec<String> = if let Some(subset) = bands_subset.as_ref() {
            let supported: HashMap<&str, ()> = endpoint
                .band_names_supported
                .iter()
                .copied()
                .map(|n| (n, ()))
                .collect();
            for n in subset {
                if !supported.contains_key(n.as_str()) {
                    anyhow::bail!(
                        "band {n:?} not supported by endpoint {:?}; valid: {:?}",
                        endpoint.kind,
                        endpoint.band_names_supported
                    );
                }
            }
            subset.clone()
        } else {
            endpoint
                .band_names_supported
                .iter()
                .map(|s| (*s).to_string())
                .collect()
        };
        let mut band_tasks = FuturesUnordered::new();
        for (scene_idx, scene) in scenes_for_fetch.iter().enumerate() {
            let read_grid = scene_grids[scene_idx].clone();
            let scene_src = scene_source_proj(&endpoint, scene);
            for (band_idx, band_name) in band_names_out.iter().enumerate() {
                // Per-scene asset resolution: HLS L30 and S30 use the
                // same stable band name but different asset keys.
                let Some(asset) = endpoint.asset_for(&scene.collection, band_name) else {
                    tracing::warn!(
                        scene = %scene.id,
                        collection = %scene.collection,
                        band = %band_name,
                        "endpoint doesn't expose this band on this collection; skipping"
                    );
                    continue;
                };
                let asset = asset.to_string();
                let scene = scene.clone();
                let http = http.clone();
                let grid = read_grid.clone();
                let sem = request_semaphore.clone();
                let scene_src = scene_src.clone();
                band_tasks.push(tokio::spawn(async move {
                    let res = fetch_band(
                        http,
                        &scene,
                        &asset,
                        &grid,
                        sem,
                        scene_src.as_deref(),
                        apply_s2_offset,
                    )
                    .await?;
                    Ok::<(usize, usize, Option<Vec<u16>>), anyhow::Error>((
                        scene_idx, band_idx, res,
                    ))
                }));
            }
            let Some(quality_asset) = endpoint.quality_asset_for(&scene.collection) else {
                tracing::warn!(
                    scene = %scene.id,
                    collection = %scene.collection,
                    "no quality asset for this collection; scene cannot contribute"
                );
                continue;
            };
            let quality_asset = quality_asset.to_string();
            let scene = scene.clone();
            let http = http.clone();
            let grid = read_grid.clone();
            let sem = request_semaphore.clone();
            let scene_src = scene_src.clone();
            band_tasks.push(tokio::spawn(async move {
                let res = fetch_quality(
                    http,
                    &scene,
                    &grid,
                    sem,
                    &quality_asset,
                    quality_kind,
                    scene_src.as_deref(),
                )
                .await?;
                Ok::<(usize, usize, Option<Vec<u16>>), anyhow::Error>((
                    scene_idx,
                    usize::MAX,
                    res.map(|q| q.iter().map(|&b| b as u16).collect()),
                ))
            }));
        }
        let mut bands_by_scene: Vec<Vec<Option<Vec<u16>>>> =
            vec![vec![None; band_names_out.len()]; scenes_for_fetch.len()];
        let mut quality_by_scene: Vec<Option<Vec<u8>>> = vec![None; scenes_for_fetch.len()];
        while let Some(res) = band_tasks.next().await {
            match res {
                Ok(Ok((scene_idx, band_idx, data))) => {
                    if band_idx == usize::MAX {
                        quality_by_scene[scene_idx] =
                            data.map(|v| v.iter().map(|&x| x as u8).collect());
                    } else {
                        bands_by_scene[scene_idx][band_idx] = data;
                    }
                }
                Ok(Err(e)) => eprintln!("fetch error: {e}"),
                Err(e) => eprintln!("fetch task panic: {e}"),
            }
        }
        let (grid_w, grid_h) = (grid.width as usize, grid.height as usize);
        let nodata_fill = quality_kind.nodata_fill();
        let mut observations: Vec<(String, Vec<Vec<u16>>, Vec<u8>)> = Vec::new();
        for (idx, scene) in scenes_for_fetch.iter().enumerate() {
            let quality = match quality_by_scene[idx].take() {
                Some(v) => v,
                None => continue,
            };
            let mut bands_data = Vec::with_capacity(band_names_out.len());
            let mut ok = true;
            for b in 0..band_names_out.len() {
                match bands_by_scene[idx][b].take() {
                    Some(d) => bands_data.push(d),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            // Windowed reads come back sized to the sub-window — scatter them
            // into full-grid buffers (bands padded with 0, quality with a
            // nodata byte so compose ignores the padding) so compose_best_pixel
            // sees uniform full-grid observations.
            if let Some(win) = windows[idx] {
                bands_data = bands_data
                    .into_iter()
                    .map(|b| scatter(&b, &win, grid_w, grid_h, 0u16))
                    .collect();
                let quality = scatter(&quality, &win, grid_w, grid_h, nodata_fill);
                observations.push((scene.id.clone(), bands_data, quality));
            } else {
                observations.push((scene.id.clone(), bands_data, quality));
            }
        }
        timings.insert("fetch".to_string(), t.elapsed().as_secs_f64());
        timings.insert("read_megapixels".to_string(), read_megapixels);

        // 5. compose
        let t = Instant::now();
        let composite =
            compose_best_pixel(&grid, band_names_out.len(), observations, quality_kind);
        timings.insert("compose".to_string(), t.elapsed().as_secs_f64());
        timings.insert("total".to_string(), t_total.elapsed().as_secs_f64());

        let partition_tiles = partition
            .as_ref()
            .map(|p| p.tiles.clone())
            .unwrap_or_default();
        let multi_tile_chunks = partition
            .as_ref()
            .map(|p| {
                p.requirements
                    .values()
                    .filter(|r| r.required_tiles.len() > 1)
                    .count()
            })
            .unwrap_or(0);

        Ok::<BuildResult, anyhow::Error>(BuildResult {
            grid: grid.clone(),
            bands: composite.bands,
            band_names: band_names_out,
            quality: composite.quality,
            observation_count: composite.observation_count,
            selected_observation: composite.selected_observation,
            source_ids: composite.source_ids,
            timings,
            endpoint_url: endpoint.stac_url.clone(),
            collection: endpoint.collections_key(),
            partition_tiles,
            multi_tile_chunks,
        })
    })
}

/// Multi-period orchestrator: ONE STAC search + scout, then iterates
/// (year, month) combinations re-using the shared scene list.
///
/// For "monthly composite of June/July/August across 2018-2020" this
/// hits STAC + scout once across the full datetime range, then runs
/// 9 separate fetch+compose phases sharing the same connection pool.
#[allow(clippy::too_many_arguments)]
fn run_build_periods(
    bbox: [f64; 4],
    years: Vec<u32>,
    months: Vec<u32>,
    resolution: f64,
    sel: SelectParams,
    max_cloud_cover: f64,
    concurrency: usize,
    endpoint: String,
    disk_cache: Option<String>,
    scout_factor: u32,
    bands_subset: Option<Vec<String>>,
    output_crs: String,
    aod_by_day: Option<HashMap<String, f64>>,
    aod_max: Option<f64>,
    reject_unknown: bool,
) -> anyhow::Result<Vec<PeriodResult>> {
    use anyhow::Context;
    use futures::stream::{FuturesUnordered, StreamExt};
    use std::time::Instant;

    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);
    // Cap by CPUs as a floor of safety; the request concurrency default
    // (192) was tuned against Planetary Computer: a 5-year Egypt batch ran
    // ~13s at 192 vs ~23s at 600 — oversubscribing the blob endpoint gives
    // no extra throughput and trips its 429 rate-limit (→ backoff).
    let safe_concurrency = (cpus.saturating_mul(50)).max(50);
    let effective_concurrency = concurrency.min(safe_concurrency);

    // Sort + dedup years/months, build period list in (year, month) order.
    let mut years_sorted = years.clone();
    years_sorted.sort();
    years_sorted.dedup();
    let mut months_sorted = months.clone();
    months_sorted.sort();
    months_sorted.dedup();
    let periods: Vec<(u32, u32)> = years_sorted
        .iter()
        .flat_map(|y| months_sorted.iter().map(move |m| (*y, *m)))
        .collect();

    let rt = shared_runtime();
    rt.block_on(async move {
        let endpoint_kind = if endpoint == "auto" {
            auto_pick()
        } else {
            EndpointKind::parse(&endpoint)?
        };
        let endpoint = shared_endpoint(endpoint_kind);
        let grid = choose_grid(&endpoint, bbox, resolution, &output_crs)?;

        let http = shared_http();
        let request_semaphore =
            Arc::new(tokio::sync::Semaphore::new(effective_concurrency.max(1)));
        let cache = disk_cache.as_ref().map(|p| DiskCache::new(PathBuf::from(p)));
        crate::cog::set_cog_disk_cache(cache.clone());

        let t_total = Instant::now();
        let collections_key = endpoint.collections_key();
        let cloud_cover_filter = if endpoint.supports_cloud_cover_filter() {
            Some(max_cloud_cover)
        } else {
            None
        };

        // 1. One STAC search per requested (year, month), run concurrently.
        //
        // A single search over the contiguous min..max range pages through
        // every gap month between the requested ones — July-only across 5
        // years scans ~48 months of scenes (hundreds of items, many pages),
        // which measured ~11s and dominated the wall. Searching each period
        // separately pages only the months we actually compose; running the
        // searches concurrently keeps total latency to roughly one search.
        // The disk cache keys each search on its own per-period datetime, so
        // warm runs reuse them. (For contiguous month ranges this is a few
        // more searches than one union query, but each is small and they
        // overlap, so the wall is unchanged — and gap months are never
        // scanned.)
        let t = Instant::now();
        // Use the calendar-correct last day per month. A hardcoded "-31"
        // produces an invalid date for 30-day months (and February), which
        // both Element84 earth-search and Planetary Computer reject with HTTP
        // 400 — so that month's search fails outright rather than yielding no
        // items. `last_day_of_month` keeps the range valid for every month.
        let period_datetimes: Vec<String> = periods
            .iter()
            .map(|(y, m)| {
                let last = crate::pipeline::last_day_of_month(*y, *m);
                format!("{y:04}-{m:02}-01/{y:04}-{m:02}-{last:02}")
            })
            .collect();
        let search_futs = period_datetimes.iter().map(|dt| {
            let endpoint = &endpoint;
            let cache = &cache;
            let collections_key = &collections_key;
            async move {
                let stac = StacClient::new(
                    &endpoint.stac_url,
                    endpoint.collections.clone(),
                    bbox,
                    dt.clone(),
                    cloud_cover_filter,
                )?;
                if let Some(c) = cache {
                    let key = c.search_key(
                        &endpoint.stac_url,
                        collections_key,
                        bbox,
                        dt,
                        cloud_cover_filter,
                    );
                    if let Some(items) = c.load_search(&key)? {
                        return Ok::<Vec<serde_json::Value>, anyhow::Error>(items);
                    }
                    let fresh = stac.search_raw().await.context("STAC search")?;
                    c.store_search(&key, &fresh)?;
                    Ok(fresh)
                } else {
                    Ok(stac.search_raw().await.context("STAC search")?)
                }
            }
        });
        let per_period_raw = futures::future::try_join_all(search_futs).await?;
        // Flatten; dedup by item id (a scene belongs to exactly one month,
        // so collisions are unlikely, but be defensive).
        let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut raw_items: Vec<serde_json::Value> = Vec::new();
        for batch in per_period_raw {
            for it in batch {
                match it.get("id").and_then(|v| v.as_str()) {
                    Some(id) => {
                        if seen_ids.insert(id.to_string()) {
                            raw_items.push(it);
                        }
                    }
                    None => raw_items.push(it),
                }
            }
        }

        let signed_items: Vec<serde_json::Value> = {
            let mut signed = Vec::with_capacity(raw_items.len());
            for raw in raw_items {
                signed.push(endpoint.sign_item(&http, raw).await?);
            }
            signed
        };
        let items = StacClient::items_from_raw(signed_items);
        let dt_list = t.elapsed().as_secs_f64();

        // 2. Scout each scene once. The scout result is geometry-based
        // and doesn't care about the time window; one scout per scene
        // covers every period it could feed into.
        let t = Instant::now();
        let coarse_resolution = resolution * scout_factor as f64;
        let mut stats_map: HashMap<String, crate::pipeline::SceneStats> = HashMap::new();
        let mut to_scout: Vec<crate::stac::StacItem> = Vec::new();
        let gsig =
            grid_signature(grid.bounds, grid.epsg, grid.resolution, grid.width, grid.height);
        if let Some(c) = &cache {
            for item in &items {
                let key = c.scout_key(
                    &endpoint.stac_url,
                    &collections_key,
                    &item.id,
                    &gsig,
                    512,
                    scout_factor,
                );
                if let Some(cached) = c.load_scout(&key)? {
                    stats_map.insert(item.id.clone(), scenestats_from_cache(&item.id, &cached));
                    continue;
                }
                to_scout.push(item.clone());
            }
        } else {
            to_scout.extend(items.iter().cloned());
        }
        let quality_kind = endpoint.quality_kind;
        let mut scout_tasks = FuturesUnordered::new();
        for item in to_scout {
            let http = http.clone();
            let grid = grid.clone();
            let sem = request_semaphore.clone();
            let quality_asset = match endpoint.quality_asset_for(&item.collection) {
                Some(a) => a.to_string(),
                None => continue,
            };
            let scene_src = scene_source_proj(&endpoint, &item);
            scout_tasks.push(tokio::spawn(async move {
                scout_scene(
                    http,
                    &item,
                    &grid,
                    coarse_resolution,
                    sem,
                    &quality_asset,
                    quality_kind,
                    scene_src.as_deref(),
                )
                .await
            }));
        }
        let scout_cache = cache.clone();
        while let Some(res) = scout_tasks.next().await {
            match res {
                Ok(Ok(s)) => {
                    if let Some(c) = &scout_cache {
                        let key = c.scout_key(
                            &endpoint.stac_url,
                            &collections_key,
                            &s.item_id,
                            &gsig,
                            512,
                            scout_factor,
                        );
                        let _ = c.store_scout(&key, &scenestats_to_cache(&s));
                    }
                    stats_map.insert(s.item_id.clone(), s);
                }
                Ok(Err(e)) => eprintln!("scout error: {e}"),
                Err(e) => eprintln!("scout task panic: {e}"),
            }
        }
        let dt_scout = t.elapsed().as_secs_f64();

        // Band-name resolution (same for every period).
        let band_names_out: Vec<String> = if let Some(subset) = bands_subset.as_ref() {
            let supported: HashMap<&str, ()> = endpoint
                .band_names_supported
                .iter()
                .copied()
                .map(|n| (n, ()))
                .collect();
            for n in subset {
                if !supported.contains_key(n.as_str()) {
                    anyhow::bail!(
                        "band {n:?} not supported by endpoint {:?}; valid: {:?}",
                        endpoint.kind,
                        endpoint.band_names_supported
                    );
                }
            }
            subset.clone()
        } else {
            endpoint
                .band_names_supported
                .iter()
                .map(|s| (*s).to_string())
                .collect()
        };

        // 3. Compose per (year, month) SEQUENTIALLY.
        //
        // compose_one_period already fans every (scene × band) fetch out
        // across the shared semaphore and connection pool, which saturates
        // the link on its own. Running multiple periods concurrently only
        // oversubscribes it: in-flight requests climb past the idle-pool
        // size (forcing fresh TCP+TLS handshakes) and contend for bandwidth,
        // dropping aggregate throughput *below* the sequential case
        // (measured ~23.7s concurrent vs ~15.9s sequential for a 5-period
        // Nile Delta batch against PC, and per-period fetch inflated ~2.5s
        // → ~7.5s). The batch win is the shared (concurrent) searches +
        // single scout pass above — not period-level fetch parallelism. Each
        // period also reuses the now-warm connection pool left by the
        // previous one.
        let band_names_arc = Arc::new(band_names_out);
        let items_arc = Arc::new(items);
        let stats_arc = Arc::new(stats_map);

        let mut results: Vec<PeriodResult> = Vec::with_capacity(periods.len());
        let mut errs: Vec<String> = Vec::new();
        for (year, month) in periods {
            match compose_one_period(
                year,
                month,
                items_arc.as_ref(),
                stats_arc.as_ref(),
                &grid,
                endpoint.as_ref(),
                http.clone(),
                request_semaphore.clone(),
                band_names_arc.as_ref(),
                sel,
                aod_by_day.as_ref(),
                aod_max,
                reject_unknown,
            )
            .await
            {
                Ok(build) => results.push(PeriodResult { year, month, build }),
                Err(e) => errs.push(format!("period {year:04}-{month:02} err: {e:#}")),
            }
        }
        if results.is_empty() && !errs.is_empty() {
            anyhow::bail!("all periods failed:\n  {}", errs.join("\n  "));
        }
        // Sort by (year, month) for stable output ordering.
        results.sort_by_key(|r| (r.year, r.month));

        // Surface the shared (amortized-once) phase costs on every period,
        // mirroring what build_composite returns per call. These are not
        // per-period — they're the single search + single scout shared
        // across the whole batch.
        for r in results.iter_mut() {
            r.build
                .timings
                .insert("shared_list_scenes".to_string(), dt_list);
            r.build.timings.insert("shared_scout".to_string(), dt_scout);
        }

        tracing::info!(
            list_scenes = dt_list,
            scout = dt_scout,
            periods = results.len(),
            total = t_total.elapsed().as_secs_f64(),
            "build_monthly_composites finished",
        );
        Ok::<Vec<PeriodResult>, anyhow::Error>(results)
    })
}

/// Fetch + compose for a single (year, month) period using
/// pre-scouted shared item / stats data. Returns one BuildResult.
#[allow(clippy::too_many_arguments)]
async fn compose_one_period(
    year: u32,
    month: u32,
    items: &[crate::stac::StacItem],
    stats: &HashMap<String, crate::pipeline::SceneStats>,
    grid: &GridSpec,
    endpoint: &EndpointConfig,
    http: Arc<reqwest::Client>,
    sem: Arc<tokio::sync::Semaphore>,
    band_names_out: &[String],
    sel: SelectParams,
    aod_by_day: Option<&HashMap<String, f64>>,
    aod_max: Option<f64>,
    reject_unknown: bool,
) -> anyhow::Result<BuildResult> {
    use futures::stream::{FuturesUnordered, StreamExt};
    use std::time::Instant;
    let t_total = Instant::now();
    let mut timings: HashMap<String, f64> = HashMap::new();

    // Filter items to this period.
    let prefix = format!("{year:04}-{month:02}");
    let mut period_items: Vec<crate::stac::StacItem> = items
        .iter()
        .filter(|it| it.datetime.starts_with(&prefix))
        .cloned()
        .collect();
    // Optional MAIAC-AOD day gate: drop atmospherically dirty days BEFORE
    // best-pixel selection/compositing, so the composite uses only low-AOD
    // acquisitions. No-op unless both `aod_max` and `aod_by_day` are supplied;
    // days absent from the map are kept (unknown != dirty). See
    // `pipeline::day_aod_passes`.
    if aod_max.is_some() && aod_by_day.is_some() {
        let before = period_items.len();
        period_items
            .retain(|it| {
                crate::pipeline::day_aod_passes(&it.datetime, aod_by_day, aod_max, reject_unknown)
            });
        timings.insert("aod_rejected".to_string(), (before - period_items.len()) as f64);
    }

    // Select scenes. Adaptive path uses per-chunk scout coverage to spend
    // depth only where it's needed; the legacy path takes a fixed top_k per
    // required tile.
    let chunks =
        chunks_from_grid(grid.bounds, grid.resolution, (grid.width, grid.height), SELECT_CHUNK_SIZE);
    let scene_geoms: Vec<(usize, String, serde_json::Value)> = period_items
        .iter()
        .enumerate()
        .filter(|(_, s)| !s.mgrs_tile.is_empty() && !s.geometry.is_null())
        .map(|(idx, s)| (idx, s.mgrs_tile.clone(), s.geometry.clone()))
        .collect();
    let partition = build_partition(
        &chunks,
        &grid.proj_def(),
        &scene_geoms,
        1,
        (grid.resolution * grid.resolution) as f64,
    )?;
    // `windows[i]` is the pixel sub-window to read scene `i` over (Level 2
    // windowed fetch); `None` = read the full grid (Level 1 / legacy).
    let (scenes_for_fetch, windows): (Vec<crate::stac::StacItem>, Vec<Option<FetchWin>>) =
        if sel.adaptive() {
            select_adaptive(
                &period_items,
                stats,
                partition.as_ref(),
                chunks.len(),
                sel.coverage_target,
                sel.min_k,
                sel.max_k,
            )
            .into_iter()
            .map(|p| {
                let win = if sel.windowed { chunks_to_window(&p.chunk_ids, grid) } else { None };
                (p.scene.clone(), win)
            })
            .unzip()
        } else {
            let picks = if let Some(p) = &partition {
                select_chunk_tile_aware(&period_items, stats, p, sel.top_k)
            } else {
                select_top_k(&period_items, stats, sel.top_k)
            };
            picks.iter().map(|p| (p.scene.clone(), None)).unzip()
        };
    // Per-scene read grid: the sub-window when windowed, else the full grid.
    let scene_grids: Vec<GridSpec> = windows
        .iter()
        .map(|w| w.map(|win| subgrid(grid, &win)).unwrap_or_else(|| grid.clone()))
        .collect();
    // Deterministic proxy for fetch volume: total source pixels read across
    // scenes (bytes ∝ this × bands). Windowed reads shrink it for the
    // marginal scenes that only a few chunks requested.
    let read_megapixels: f64 = scene_grids
        .iter()
        .map(|g| g.width as f64 * g.height as f64)
        .sum::<f64>()
        / 1e6;

    // Fetch + compose, mirroring run_build's loop.
    let t = Instant::now();
    let quality_kind = endpoint.quality_kind;
    let apply_s2_offset = endpoint.applies_s2_boa_offset();
    let mut band_tasks = FuturesUnordered::new();
    for (scene_idx, scene) in scenes_for_fetch.iter().enumerate() {
        let read_grid = scene_grids[scene_idx].clone();
        let scene_src = scene_source_proj(endpoint, scene);
        for (band_idx, band_name) in band_names_out.iter().enumerate() {
            let Some(asset) = endpoint.asset_for(&scene.collection, band_name) else {
                continue;
            };
            let asset = asset.to_string();
            let scene = scene.clone();
            let http = http.clone();
            let grid = read_grid.clone();
            let sem = sem.clone();
            let scene_src = scene_src.clone();
            band_tasks.push(tokio::spawn(async move {
                let res =
                    fetch_band(http, &scene, &asset, &grid, sem, scene_src.as_deref(), apply_s2_offset)
                        .await?;
                Ok::<(usize, usize, Option<Vec<u16>>), anyhow::Error>((
                    scene_idx, band_idx, res,
                ))
            }));
        }
        let Some(quality_asset) = endpoint.quality_asset_for(&scene.collection) else {
            continue;
        };
        let quality_asset = quality_asset.to_string();
        let scene = scene.clone();
        let http = http.clone();
        let grid = read_grid.clone();
        let sem = sem.clone();
        let scene_src = scene_src.clone();
        band_tasks.push(tokio::spawn(async move {
            let res = fetch_quality(
                http,
                &scene,
                &grid,
                sem,
                &quality_asset,
                quality_kind,
                scene_src.as_deref(),
            )
            .await?;
            Ok::<(usize, usize, Option<Vec<u16>>), anyhow::Error>((
                scene_idx,
                usize::MAX,
                res.map(|q| q.iter().map(|&b| b as u16).collect()),
            ))
        }));
    }
    let mut bands_by_scene: Vec<Vec<Option<Vec<u16>>>> =
        vec![vec![None; band_names_out.len()]; scenes_for_fetch.len()];
    let mut quality_by_scene: Vec<Option<Vec<u8>>> = vec![None; scenes_for_fetch.len()];
    while let Some(res) = band_tasks.next().await {
        match res {
            Ok(Ok((scene_idx, band_idx, data))) => {
                if band_idx == usize::MAX {
                    quality_by_scene[scene_idx] =
                        data.map(|v| v.iter().map(|&x| x as u8).collect());
                } else {
                    bands_by_scene[scene_idx][band_idx] = data;
                }
            }
            Ok(Err(e)) => eprintln!("fetch error: {e}"),
            Err(e) => eprintln!("fetch task panic: {e}"),
        }
    }
    let (grid_w, grid_h) = (grid.width as usize, grid.height as usize);
    let nodata_fill = quality_kind.nodata_fill();
    let mut observations: Vec<(String, Vec<Vec<u16>>, Vec<u8>)> = Vec::new();
    for (idx, scene) in scenes_for_fetch.iter().enumerate() {
        let quality = match quality_by_scene[idx].take() {
            Some(v) => v,
            None => continue,
        };
        let mut bands_data = Vec::with_capacity(band_names_out.len());
        let mut ok = true;
        for b in 0..band_names_out.len() {
            match bands_by_scene[idx][b].take() {
                Some(d) => bands_data.push(d),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        // Windowed reads come back sized to the sub-window — scatter them
        // into full-grid buffers (bands padded with 0, quality with a
        // nodata byte so compose ignores the padding) so compose_best_pixel
        // sees uniform full-grid observations.
        if let Some(win) = windows[idx] {
            bands_data = bands_data
                .into_iter()
                .map(|b| scatter(&b, &win, grid_w, grid_h, 0u16))
                .collect();
            let quality = scatter(&quality, &win, grid_w, grid_h, nodata_fill);
            observations.push((scene.id.clone(), bands_data, quality));
        } else {
            observations.push((scene.id.clone(), bands_data, quality));
        }
    }
    timings.insert("fetch".to_string(), t.elapsed().as_secs_f64());
    timings.insert("read_megapixels".to_string(), read_megapixels);

    let t = Instant::now();
    let composite =
        compose_best_pixel(grid, band_names_out.len(), observations, quality_kind);
    timings.insert("compose".to_string(), t.elapsed().as_secs_f64());
    timings.insert("total".to_string(), t_total.elapsed().as_secs_f64());

    let partition_tiles = partition
        .as_ref()
        .map(|p| p.tiles.clone())
        .unwrap_or_default();
    let multi_tile_chunks = partition
        .as_ref()
        .map(|p| {
            p.requirements
                .values()
                .filter(|r| r.required_tiles.len() > 1)
                .count()
        })
        .unwrap_or(0);

    Ok(BuildResult {
        grid: grid.clone(),
        bands: composite.bands,
        band_names: band_names_out.to_vec(),
        quality: composite.quality,
        observation_count: composite.observation_count,
        selected_observation: composite.selected_observation,
        source_ids: composite.source_ids,
        timings,
        endpoint_url: endpoint.stac_url.clone(),
        collection: endpoint.collections_key(),
        partition_tiles,
        multi_tile_chunks,
    })
}

fn encode_result(py: Python<'_>, r: BuildResult) -> PyResult<Bound<'_, PyDict>> {
    let out = PyDict::new_bound(py);

    // Per-band numpy arrays, keyed by the (possibly subset) band names.
    // Go via ndarray::Array2 → numpy zero-copy because numpy 0.22's
    // Bound<PyArray1> doesn't expose reshape directly.
    let bands = PyDict::new_bound(py);
    let h = r.grid.height as usize;
    let w = r.grid.width as usize;
    let mut bands_owned = r.bands;
    let band_names_out = r.band_names.clone();
    for (i, name) in band_names_out.iter().enumerate() {
        let band = std::mem::take(&mut bands_owned[i]);
        let arr2 = Array2::from_shape_vec((h, w), band)
            .map_err(|e| PyRuntimeError::new_err(format!("reshape band {name}: {e}")))?;
        bands.set_item(name.as_str(), arr2.into_pyarray_bound(py))?;
    }
    out.set_item("bands", bands)?;

    // Auxiliary arrays.
    let quality_arr = Array2::from_shape_vec((h, w), r.quality)
        .map_err(|e| PyRuntimeError::new_err(format!("reshape quality: {e}")))?;
    out.set_item("quality", quality_arr.into_pyarray_bound(py))?;
    let obs_arr = Array2::from_shape_vec((h, w), r.observation_count)
        .map_err(|e| PyRuntimeError::new_err(format!("reshape obs: {e}")))?;
    out.set_item("observation_count", obs_arr.into_pyarray_bound(py))?;
    let sel_arr = Array2::from_shape_vec((h, w), r.selected_observation)
        .map_err(|e| PyRuntimeError::new_err(format!("reshape sel: {e}")))?;
    out.set_item("selected_observation", sel_arr.into_pyarray_bound(py))?;

    // Grid metadata.
    let grid = PyDict::new_bound(py);
    grid.set_item("bounds", r.grid.bounds.to_vec())?;
    grid.set_item("epsg", r.grid.epsg)?;
    grid.set_item("proj4", r.grid.proj4.clone())?;
    grid.set_item("crs", r.grid.proj_def())?;
    grid.set_item("resolution", r.grid.resolution)?;
    grid.set_item("width", r.grid.width)?;
    grid.set_item("height", r.grid.height)?;
    grid.set_item("transform", r.grid.affine_transform().to_vec())?;
    out.set_item("grid", grid)?;

    // Endpoint + source metadata.
    out.set_item("endpoint", r.endpoint_url)?;
    out.set_item("collection", r.collection)?;
    let sources = PyList::new_bound(py, &r.source_ids);
    out.set_item("source_ids", sources)?;
    let partition_tiles = PyList::new_bound(py, &r.partition_tiles);
    out.set_item("partition_tiles", partition_tiles)?;
    out.set_item("multi_tile_chunks", r.multi_tile_chunks)?;

    // Timings.
    let timings = PyDict::new_bound(py);
    for (k, v) in &r.timings {
        timings.set_item(k, v)?;
    }
    out.set_item("timings", timings)?;

    // Stable band order for downstream code that wants iteration order.
    let band_names = PyList::new_bound(py, &r.band_names);
    out.set_item("band_names", band_names)?;

    Ok(out)
}

fn encode_period_results(
    py: Python<'_>,
    results: Vec<PeriodResult>,
) -> PyResult<Bound<'_, PyList>> {
    let out = PyList::empty_bound(py);
    for r in results {
        let dict = encode_result(py, r.build)?;
        dict.set_item("year", r.year)?;
        dict.set_item("month", r.month)?;
        out.append(dict)?;
    }
    Ok(out)
}

/// Apply the 6S surface-reflectance correction to a TOA reflectance stack,
/// using the crate's `atcorr` core. `toa` is `[band, H, W]` reflectance (0..1);
/// `xap`/`xbp`/`xcp` are one coefficient per band (already interpolated to the
/// scene's water vapour by the caller). Returns the corrected surface
/// reflectance stack: negatives clamp to 0, non-finite (nodata) passes through.
///
/// This is the L1C custom-AC composite's per-pixel correction — exposed so the
/// Python pipeline in `bestpixel/l1c.py` runs it in Rust rather than numpy.
#[pyfunction]
fn correct_toa<'py>(
    py: Python<'py>,
    toa: PyReadonlyArray3<'py, f32>,
    xap: Vec<f32>,
    xbp: Vec<f32>,
    xcp: Vec<f32>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let arr = toa.as_array();
    let shape = arr.shape();
    let (nb, h, w) = (shape[0], shape[1], shape[2]);
    if xap.len() != nb || xbp.len() != nb || xcp.len() != nb {
        return Err(PyRuntimeError::new_err(format!(
            "coeff lengths xap={} xbp={} xcp={} must equal band count {nb}",
            xap.len(),
            xbp.len(),
            xcp.len()
        )));
    }
    let mut out = Array3::<f32>::zeros((nb, h, w));
    for b in 0..nb {
        let (a, bb, c) = (xap[b], xbp[b], xcp[b]);
        for i in 0..h {
            for j in 0..w {
                let t = arr[[b, i, j]];
                out[[b, i, j]] = if t.is_finite() {
                    crate::atcorr::correct_refl(t, a, bb, c).max(0.0)
                } else {
                    f32::NAN
                };
            }
        }
    }
    Ok(out.into_pyarray_bound(py))
}

#[pymodule]
fn bestpixel(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Opt-in tracing to stderr when RUST_LOG is set (e.g.
    // RUST_LOG=bestpixel=debug surfaces fetch_band's per-fetch
    // open/read/stitch/resample timings and tile counts). No-op otherwise.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .try_init();
    m.add_function(wrap_pyfunction!(build_composite, m)?)?;
    m.add_function(wrap_pyfunction!(build_monthly_composites, m)?)?;
    m.add_function(wrap_pyfunction!(correct_toa, m)?)?;
    Ok(())
}
