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
use std::sync::Arc;

use ndarray::Array2;
use numpy::IntoPyArray;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::disk_cache::{grid_signature, DiskCache};
use crate::endpoint::{auto_pick, EndpointConfig, EndpointKind, BAND_NAMES};
use crate::grid::GridSpec;
use crate::pipeline::{
    compose_best_pixel, fetch_band, fetch_quality, scout_scene, select_chunk_tile_aware,
    select_top_k,
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
    concurrency = 600,
    endpoint = "auto".to_string(),
    disk_cache = None,
    scout_factor = 8,
    bands = None,
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
) -> PyResult<Bound<'_, PyDict>> {
    // Release the GIL while the heavy async work runs — lets concurrent
    // Python threads do other things even though we block on tokio.
    let result = py
        .allow_threads(|| {
            run_build(
                bbox, datetime, resolution, top_k, max_cloud_cover, concurrency, endpoint,
                disk_cache, scout_factor, bands,
            )
        })
        .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
    encode_result(py, result)
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

fn run_build(
    bbox: [f64; 4],
    datetime: String,
    resolution: f64,
    top_k: usize,
    max_cloud_cover: f64,
    concurrency: usize,
    endpoint: String,
    disk_cache: Option<String>,
    scout_factor: u32,
    bands_subset: Option<Vec<String>>,
) -> anyhow::Result<BuildResult> {
    use anyhow::Context;
    use futures::stream::{FuturesUnordered, StreamExt};
    use std::time::Instant;

    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);
    let safe_concurrency = (cpus.saturating_mul(50)).max(50);
    let effective_concurrency = concurrency.min(safe_concurrency);

    let rt = if cpus <= 1 {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?
    } else {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(cpus.max(2))
            .enable_all()
            .build()?
    };
    rt.block_on(async move {
        let grid = GridSpec::from_wgs84_bounds(bbox, resolution);
        let endpoint_kind = if endpoint == "auto" {
            auto_pick()
        } else {
            EndpointKind::parse(&endpoint)?
        };
        let endpoint = Arc::new(EndpointConfig::build(endpoint_kind));

        let http = Arc::new(
            reqwest::Client::builder()
                .gzip(true)
                .http2_adaptive_window(true)
                .pool_max_idle_per_host(256)
                .tcp_keepalive(std::time::Duration::from_secs(60))
                .tcp_nodelay(true)
                .build()?,
        );
        let request_semaphore =
            Arc::new(tokio::sync::Semaphore::new(effective_concurrency.max(1)));
        let cache = disk_cache.as_ref().map(|p| DiskCache::new(PathBuf::from(p)));

        let mut timings: HashMap<String, f64> = HashMap::new();
        let t_total = Instant::now();

        // 1. list_scenes
        let t = Instant::now();
        let stac = StacClient::new(
            &endpoint.stac_url,
            &endpoint.collection,
            bbox,
            datetime.clone(),
            Some(max_cloud_cover),
        )?;
        let raw_items = if let Some(c) = &cache {
            let key = c.search_key(
                &endpoint.stac_url,
                &endpoint.collection,
                bbox,
                &datetime,
                Some(max_cloud_cover),
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
        // Sign for PC if needed.
        let signed_items: Vec<serde_json::Value> = if matches!(endpoint.kind, EndpointKind::PlanetaryComputer) {
            let mut signed = Vec::with_capacity(raw_items.len());
            for raw in raw_items {
                signed.push(endpoint.sign_item(&http, raw).await?);
            }
            signed
        } else {
            raw_items
        };
        let items = StacClient::items_from_raw(signed_items);
        timings.insert("list_scenes".to_string(), t.elapsed().as_secs_f64());

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
                    &endpoint.collection,
                    &item.id,
                    &gsig,
                    512,
                    scout_factor,
                );
                if let Some(cached) = c.load_scout(&key)? {
                    let usable = cached
                        .iter()
                        .map(|s| s.usable_fraction)
                        .fold(0.0_f32, |a, b| a.max(b));
                    let mean_clear = cached
                        .iter()
                        .map(|s| s.mean_clear)
                        .filter(|v| v.is_finite())
                        .fold(f32::NAN, |a, b| if a.is_finite() { a.max(b) } else { b });
                    stats_map.insert(
                        item.id.clone(),
                        crate::pipeline::SceneStats {
                            item_id: item.id.clone(),
                            usable_fraction: usable,
                            mean_clear,
                        },
                    );
                    continue;
                }
                to_scout.push(item.clone());
            }
        } else {
            to_scout.extend(items.iter().cloned());
        }
        let mut tasks = FuturesUnordered::new();
        for item in to_scout {
            let http = http.clone();
            let grid = grid;
            let sem = request_semaphore.clone();
            let scl_asset = endpoint.scl_asset.clone();
            tasks.push(tokio::spawn(async move {
                scout_scene(http, &item, &grid, coarse_resolution, sem, &scl_asset).await
            }));
        }
        let scout_cache = cache.clone();
        while let Some(res) = tasks.next().await {
            match res {
                Ok(Ok(s)) => {
                    if let Some(c) = &scout_cache {
                        let key = c.scout_key(
                            &endpoint.stac_url,
                            &endpoint.collection,
                            &s.item_id,
                            &gsig,
                            512,
                            scout_factor,
                        );
                        let stat = crate::pipeline::SceneChunkStat {
                            chunk_id: 0,
                            usable_fraction: s.usable_fraction,
                            mean_clear: s.mean_clear,
                        };
                        let _ = c.store_scout(&key, &[stat]);
                    }
                    stats_map.insert(s.item_id.clone(), s);
                }
                Ok(Err(e)) => eprintln!("scout error: {e}"),
                Err(e) => eprintln!("scout task panic: {e}"),
            }
        }
        timings.insert("scout".to_string(), t.elapsed().as_secs_f64());

        // 3. tile partition + select
        let chunks =
            chunks_from_grid(grid.bounds, grid.resolution, (grid.width, grid.height), 512);
        let scene_geoms: Vec<(usize, String, serde_json::Value)> = items
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.mgrs_tile.is_empty() && !s.geometry.is_null())
            .map(|(idx, s)| (idx, s.mgrs_tile.clone(), s.geometry.clone()))
            .collect();
        let partition = build_partition(
            &chunks,
            grid.epsg,
            &scene_geoms,
            1,
            (grid.resolution * grid.resolution) as f64,
        )?;
        let picks = if let Some(p) = &partition {
            select_chunk_tile_aware(&items, &stats_map, p, top_k)
        } else {
            select_top_k(&items, &stats_map, top_k)
        };

        // 4. fetch
        let t = Instant::now();
        // Resolve the band subset (if any) into (stable_name, asset_key) pairs,
        // preserving the caller-requested order. If no subset is given we
        // fetch all 12 bands in the canonical BAND_NAMES order.
        let (band_names_out, band_assets): (Vec<String>, Vec<String>) =
            if let Some(subset) = bands_subset.as_ref() {
                let asset_lookup: HashMap<&str, &str> = BAND_NAMES
                    .iter()
                    .copied()
                    .zip(endpoint.band_assets.iter().map(|s| s.as_str()))
                    .collect();
                let mut names = Vec::with_capacity(subset.len());
                let mut assets = Vec::with_capacity(subset.len());
                for n in subset {
                    let a = asset_lookup.get(n.as_str()).copied().ok_or_else(|| {
                        anyhow::anyhow!(
                            "unknown band {n:?}; valid: {:?}",
                            BAND_NAMES
                        )
                    })?;
                    names.push(n.clone());
                    assets.push(a.to_string());
                }
                (names, assets)
            } else {
                (
                    BAND_NAMES.iter().map(|s| (*s).to_string()).collect(),
                    endpoint.band_assets.clone(),
                )
            };
        let scenes_for_fetch: Vec<crate::stac::StacItem> =
            picks.iter().map(|p| p.scene.clone()).collect();
        let mut band_tasks = FuturesUnordered::new();
        for (scene_idx, scene) in scenes_for_fetch.iter().enumerate() {
            for (band_idx, band) in band_assets.iter().enumerate() {
                let scene = scene.clone();
                let band = band.clone();
                let http = http.clone();
                let grid = grid;
                let sem = request_semaphore.clone();
                band_tasks.push(tokio::spawn(async move {
                    let res = fetch_band(http, &scene, &band, &grid, sem).await?;
                    Ok::<(usize, usize, Option<Vec<u16>>), anyhow::Error>((
                        scene_idx, band_idx, res,
                    ))
                }));
            }
            let scene = scene.clone();
            let http = http.clone();
            let grid = grid;
            let sem = request_semaphore.clone();
            let scl_asset = endpoint.scl_asset.clone();
            band_tasks.push(tokio::spawn(async move {
                let res = fetch_quality(http, &scene, &grid, sem, &scl_asset).await?;
                Ok::<(usize, usize, Option<Vec<u16>>), anyhow::Error>((
                    scene_idx,
                    usize::MAX,
                    res.map(|scl| scl.iter().map(|&b| b as u16).collect()),
                ))
            }));
        }
        let mut bands_by_scene: Vec<Vec<Option<Vec<u16>>>> =
            vec![vec![None; band_assets.len()]; scenes_for_fetch.len()];
        let mut scl_by_scene: Vec<Option<Vec<u8>>> = vec![None; scenes_for_fetch.len()];
        while let Some(res) = band_tasks.next().await {
            match res {
                Ok(Ok((scene_idx, band_idx, data))) => {
                    if band_idx == usize::MAX {
                        scl_by_scene[scene_idx] =
                            data.map(|v| v.iter().map(|&x| x as u8).collect());
                    } else {
                        bands_by_scene[scene_idx][band_idx] = data;
                    }
                }
                Ok(Err(e)) => eprintln!("fetch error: {e}"),
                Err(e) => eprintln!("fetch task panic: {e}"),
            }
        }
        let mut observations: Vec<(String, Vec<Vec<u16>>, Vec<u8>)> = Vec::new();
        for (idx, scene) in scenes_for_fetch.iter().enumerate() {
            let scl = match scl_by_scene[idx].take() {
                Some(v) => v,
                None => continue,
            };
            let mut bands_data = Vec::with_capacity(band_assets.len());
            let mut ok = true;
            for b in 0..band_assets.len() {
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
            observations.push((scene.id.clone(), bands_data, scl));
        }
        timings.insert("fetch".to_string(), t.elapsed().as_secs_f64());

        // 5. compose
        let t = Instant::now();
        let composite = compose_best_pixel(grid, band_assets.len(), observations);
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
            grid,
            bands: composite.bands,
            band_names: band_names_out,
            quality: composite.quality,
            observation_count: composite.observation_count,
            selected_observation: composite.selected_observation,
            source_ids: composite.source_ids,
            timings,
            endpoint_url: endpoint.stac_url.clone(),
            collection: endpoint.collection.clone(),
            partition_tiles,
            multi_tile_chunks,
        })
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

#[pymodule]
fn surface_priors_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_composite, m)?)?;
    Ok(())
}
