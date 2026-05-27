//! End-to-end Rust monthly composite: STAC → scout → select → fetch →
//! best-pixel compose → GeoTIFF.

use anyhow::{Context, Result};
use clap::Parser;
use futures::stream::{FuturesUnordered, StreamExt};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use bestpixel::disk_cache::{grid_signature, DiskCache};
use bestpixel::endpoint::{auto_pick, EndpointConfig, EndpointKind};
use bestpixel::grid::GridSpec;
use bestpixel::pipeline::{
    compose_best_pixel, fetch_band, fetch_quality, scout_scene, select_chunk_tile_aware,
    select_top_k, Timing,
};
use bestpixel::stac::StacClient;

#[derive(Parser, Debug)]
#[command(about = "End-to-end S2 L2A monthly composite — Rust port.")]
struct Cli {
    /// AOI in WGS84: west south east north.
    #[arg(long, value_delimiter = ',', default_values_t = vec![30.5_f64, 30.5, 31.6, 31.5])]
    bbox: Vec<f64>,

    /// Datetime range "YYYY-MM-DD/YYYY-MM-DD" or RFC3339.
    #[arg(long, default_value = "2024-07-01/2024-07-31")]
    datetime: String,

    #[arg(long, default_value_t = 60.0)]
    resolution: f64,

    /// SCL downsample factor for scout (matches Python's scout_factor=8).
    #[arg(long, default_value_t = 8)]
    scout_factor: u32,

    #[arg(long, default_value_t = 3)]
    top_k: usize,

    #[arg(long, default_value_t = 90.0)]
    max_cloud_cover: f64,

    /// HTTP concurrency cap for COG tile fetches.
    #[arg(long, default_value_t = 200)]
    concurrency: usize,

    #[arg(long, default_value = "/tmp/spx-rust-output")]
    out_dir: PathBuf,

    /// Only emit timing JSON; skip GeoTIFF writes.
    #[arg(long, default_value_t = false)]
    no_write: bool,

    /// Persist STAC search results to this directory across processes.
    #[arg(long)]
    disk_cache: Option<PathBuf>,

    /// STAC endpoint: `auto` (default — picks PC unless running in
    /// AWS us-* region), `pc`, or `earth-search`.
    #[arg(long, default_value = "auto")]
    endpoint: String,
}

fn main() -> Result<()> {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);
    // On a single core, the multi-threaded runtime + spawn_blocking
    // pool + rayon all compete for one CPU and the coordination tax
    // eats the wins. Switch to the current_thread flavour: a single
    // OS thread driving the reactor, no worker pool, no inter-thread
    // signalling. Async tasks still multiplex IO-wait perfectly on
    // the one core, but CPU-bound work runs straight-line.
    if cpus <= 1 {
        return tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio current_thread runtime")
            .block_on(async_main());
    }
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(cpus.max(2))
        .enable_all()
        .build()
        .expect("tokio runtime")
        .block_on(async_main())
}

async fn async_main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();
    let cli = Cli::parse();
    if cli.bbox.len() != 4 {
        anyhow::bail!("--bbox needs exactly 4 numbers");
    }
    let bbox: [f64; 4] = [cli.bbox[0], cli.bbox[1], cli.bbox[2], cli.bbox[3]];
    // Resolve endpoint up front so we can build the right grid (UTM
    // for S2/HLS, MODIS Sinusoidal for MCD43A4).
    let endpoint_kind = if cli.endpoint == "auto" {
        auto_pick()
    } else {
        EndpointKind::parse(&cli.endpoint)?
    };
    let endpoint_for_grid = EndpointConfig::build(endpoint_kind);
    let grid = if endpoint_for_grid.uses_modis_sinusoidal() {
        GridSpec::from_wgs84_bounds_modis_sinu(bbox, cli.resolution)
    } else {
        GridSpec::from_wgs84_bounds(bbox, cli.resolution)
    };
    eprintln!(
        "grid: crs={} bounds={:?} size={}x{}",
        grid.proj_def(), grid.bounds, grid.width, grid.height
    );

    // Cap the HTTP semaphore to what the available CPUs can actually
    // service. Above ~50 in-flight requests per core the async reactor
    // can't keep up with TLS + socket polling, server connections time
    // out, and ~80% of fetches silently drop. Empirically ~50/CPU is
    // the safe upper bound; we allow up to the user-requested
    // `--concurrency` but never exceed cpus * 50.
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);
    let safe_concurrency = (cpus.saturating_mul(50)).max(50);
    let effective_concurrency = cli.concurrency.min(safe_concurrency);
    if effective_concurrency < cli.concurrency {
        eprintln!(
            "concurrency: requested {} clamped to {} for {} CPU(s) (avoid reactor starvation)",
            cli.concurrency, effective_concurrency, cpus
        );
    } else {
        eprintln!("concurrency: {} ({} CPU(s))", effective_concurrency, cpus);
    }

    let http = Arc::new(
        reqwest::Client::builder()
            .gzip(true)
            .http2_adaptive_window(true)
            .pool_max_idle_per_host(256)
            .tcp_keepalive(std::time::Duration::from_secs(60))
            .tcp_nodelay(true)
            .build()
            .context("build reqwest client")?,
    );

    // Endpoint already resolved when picking grid CRS above; rebuild
    // an Arc-wrapped copy for the rest of the pipeline.
    drop(endpoint_for_grid);
    let endpoint = Arc::new(EndpointConfig::build(endpoint_kind));
    let collections_key = endpoint.collections_key();
    eprintln!(
        "endpoint: {:?} ({})  collections: {}",
        endpoint.kind, endpoint.stac_url, collections_key
    );

    let mut timing = Timing::default();
    let t_total = Instant::now();
    // One global semaphore bounds total in-flight HTTP requests across
    // every COG read (scout + fetch_band + fetch_quality). Element84's
    // documented ceiling is ~100 concurrent per IP; 200 actually works
    // with HTTP/2 multiplexing because reqwest reuses a single TCP
    // connection per host and the limit is per-stream, not per-conn.
    let request_semaphore = Arc::new(tokio::sync::Semaphore::new(effective_concurrency.max(1)));

    let cache = cli.disk_cache.as_ref().map(|p| DiskCache::new(p));
    bestpixel::cog::set_cog_disk_cache(cache.clone());
    let _grid_sig = grid_signature(grid.bounds, grid.epsg, grid.resolution, grid.width, grid.height);

    // --- List scenes -------------------------------------------------------
    // Disk cache stores the raw STAC item dicts (unsigned) so that
    // SAS-token signers re-sign on each load.
    let t = Instant::now();
    let cloud_cover_filter = if endpoint.supports_cloud_cover_filter() {
        Some(cli.max_cloud_cover)
    } else {
        None
    };
    let stac = StacClient::new(
        &endpoint.stac_url,
        endpoint.collections.clone(),
        bbox,
        cli.datetime.clone(),
        cloud_cover_filter,
    )?;
    let raw_items = if let Some(c) = &cache {
        let key = c.search_key(
            &endpoint.stac_url,
            &collections_key,
            bbox,
            &cli.datetime,
            cloud_cover_filter,
        );
        match c.load_search(&key)? {
            Some(items) => {
                tracing::debug!(items = items.len(), "list_scenes disk-cache hit");
                items
            }
            None => {
                let fresh = stac.search_raw().await.context("STAC search")?;
                c.store_search(&key, &fresh)?;
                fresh
            }
        }
    } else {
        stac.search_raw().await.context("STAC search")?
    };
    // If we're talking to PC, append the SAS token to each asset href
    // *before* converting to StacItem (so all downstream HTTP reads
    // include the SAS query string). Cache stores the raw unsigned
    // items so subsequent runs re-sign with a fresh token.
    // sign_item is a no-op for anonymous endpoints.
    let signed_items: Vec<serde_json::Value> = {
        let mut signed = Vec::with_capacity(raw_items.len());
        for raw in raw_items {
            signed.push(endpoint.sign_item(&http, raw).await?);
        }
        signed
    };
    let items = StacClient::items_from_raw(signed_items);
    timing.list_scenes = t.elapsed().as_secs_f64();
    eprintln!("list_scenes:    {:6.2}s  ({} items)", timing.list_scenes, items.len());

    // --- Scout (per-scene cloud stats in parallel) -------------------------
    // Disk cache key includes scene id + grid signature; a rerun of
    // the same year hits the cache for every scouted scene.
    let t = Instant::now();
    let coarse_resolution = cli.resolution * cli.scout_factor as f64;
    let mut stats_map: HashMap<String, bestpixel::pipeline::SceneStats> = HashMap::new();
    let mut to_scout: Vec<bestpixel::stac::StacItem> = Vec::new();
    if let Some(c) = &cache {
        let gsig = grid_signature(grid.bounds, grid.epsg, grid.resolution, grid.width, grid.height);
        for item in &items {
            let key = c.scout_key(
                &endpoint.stac_url,
                &collections_key,
                &item.id,
                &gsig,
                512, // chunk_size for stable key alignment with partition
                cli.scout_factor,
            );
            if let Some(cached) = c.load_scout(&key)? {
                stats_map.insert(
                    item.id.clone(),
                    bestpixel::pipeline::scenestats_from_cache(&item.id, &cached),
                );
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
        tasks.push(tokio::spawn(async move {
            scout_scene(
                http,
                &item,
                &grid,
                coarse_resolution,
                sem,
                &quality_asset,
                quality_kind,
                None,
            )
            .await
        }));
    }
    let scout_cache = cache.clone();
    let gsig_for_store = grid_signature(grid.bounds, grid.epsg, grid.resolution, grid.width, grid.height);
    while let Some(res) = tasks.next().await {
        match res {
            Ok(Ok(s)) => {
                if let Some(c) = &scout_cache {
                    let key = c.scout_key(
                        &endpoint.stac_url,
                        &collections_key,
                        &s.item_id,
                        &gsig_for_store,
                        512,
                        cli.scout_factor,
                    );
                    let _ = c.store_scout(&key, &bestpixel::pipeline::scenestats_to_cache(&s));
                }
                stats_map.insert(s.item_id.clone(), s);
            }
            Ok(Err(e)) => eprintln!("scout error: {e}"),
            Err(e) => eprintln!("scout task panic: {e}"),
        }
    }
    timing.scout = t.elapsed().as_secs_f64();
    eprintln!("scout:          {:6.2}s  ({} scored)", timing.scout, stats_map.len());

    // --- Build chunk-level tile partition + select -------------------------
    let chunks = bestpixel::tile_classification::chunks_from_grid(
        grid.bounds,
        grid.resolution,
        (grid.width, grid.height),
        512,
    );
    let scene_geoms: Vec<(usize, String, serde_json::Value)> = items
        .iter()
        .enumerate()
        .filter(|(_, s)| !s.mgrs_tile.is_empty() && !s.geometry.is_null())
        .map(|(idx, s)| (idx, s.mgrs_tile.clone(), s.geometry.clone()))
        .collect();
    // Try disk cache for tile partition.
    let partition_disk_hit = if let Some(c) = &cache {
        let scenes_sig = bestpixel::tile_classification::scenes_signature(
            &items
                .iter()
                .map(|s| (s.id.clone(), s.mgrs_tile.clone(), s.geometry.clone()))
                .collect::<Vec<_>>(),
        );
        let gsig = grid_signature(grid.bounds, grid.epsg, grid.resolution, grid.width, grid.height);
        let key = c.partition_key(&scenes_sig, &gsig, 512);
        c.load_partition(&key).ok().flatten().map(|p| (p, key))
    } else {
        None
    };
    let partition = if let Some((p, _)) = partition_disk_hit.as_ref() {
        Some(p.clone())
    } else {
        let built = bestpixel::tile_classification::build_partition(
            &chunks,
            &grid.proj_def(),
            &scene_geoms,
            1,
            (grid.resolution * grid.resolution) as f64,
        )?;
        if let (Some(p), Some(c)) = (built.as_ref(), cache.as_ref()) {
            let scenes_sig = bestpixel::tile_classification::scenes_signature(
                &items
                    .iter()
                    .map(|s| (s.id.clone(), s.mgrs_tile.clone(), s.geometry.clone()))
                    .collect::<Vec<_>>(),
            );
            let gsig =
                grid_signature(grid.bounds, grid.epsg, grid.resolution, grid.width, grid.height);
            let key = c.partition_key(&scenes_sig, &gsig, 512);
            let _ = c.store_partition(&key, p);
        }
        built
    };
    let picks = if let Some(p) = &partition {
        let multi_tile = p
            .requirements
            .values()
            .filter(|r| r.required_tiles.len() > 1)
            .count();
        eprintln!(
            "partition:      {} tiles, {} chunks ({} multi-tile)",
            p.tiles.len(),
            p.requirements.len(),
            multi_tile
        );
        select_chunk_tile_aware(&items, &stats_map, p, cli.top_k)
    } else {
        eprintln!("partition:      none — falling back to per-tile top-k");
        select_top_k(&items, &stats_map, cli.top_k)
    };
    eprintln!("selected:       {} scenes (top_k={})", picks.len(), cli.top_k);
    if picks.is_empty() {
        anyhow::bail!("no usable scenes after scout — aborting");
    }

    // --- Fetch picked scenes (bands + quality), one task per (scene, band)
    let t = Instant::now();
    // Iterate stable band names; resolve asset key per-scene because
    // HLS L30 and S30 use different asset keys for the same band.
    let band_names_out: Vec<&'static str> =
        endpoint.band_names_supported.iter().copied().collect();
    let scenes_for_fetch: Vec<bestpixel::stac::StacItem> = picks.iter().map(|p| p.scene.clone()).collect();
    let mut band_tasks = FuturesUnordered::new();
    for (scene_idx, scene) in scenes_for_fetch.iter().enumerate() {
        for (band_idx, band_name) in band_names_out.iter().enumerate() {
            let Some(asset) = endpoint.asset_for(&scene.collection, band_name) else {
                continue;
            };
            let asset = asset.to_string();
            let scene = scene.clone();
            let http = http.clone();
            let grid = grid.clone();
            let sem = request_semaphore.clone();
            band_tasks.push(tokio::spawn(async move {
                let res = fetch_band(http, &scene, &asset, &grid, sem, None).await?;
                Ok::<(usize, usize, Option<Vec<u16>>), anyhow::Error>((scene_idx, band_idx, res))
            }));
        }
        // Quality (SCL or Fmask) too.
        let Some(quality_asset) = endpoint.quality_asset_for(&scene.collection) else {
            continue;
        };
        let quality_asset = quality_asset.to_string();
        let scene = scene.clone();
        let http = http.clone();
        let grid = grid.clone();
        let sem = request_semaphore.clone();
        band_tasks.push(tokio::spawn(async move {
            let res = fetch_quality(http, &scene, &grid, sem, &quality_asset, None).await?;
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
                None => { ok = false; break; }
            }
        }
        if !ok {
            continue;
        }
        observations.push((scene.id.clone(), bands_data, quality));
    }
    timing.fetch = t.elapsed().as_secs_f64();
    eprintln!(
        "fetch:          {:6.2}s  ({} observations)",
        timing.fetch,
        observations.len()
    );

    // --- Compose best-pixel ------------------------------------------------
    let t = Instant::now();
    let composite =
        compose_best_pixel(&grid, band_names_out.len(), observations, quality_kind);
    timing.compose = t.elapsed().as_secs_f64();
    eprintln!("compose:        {:6.2}s", timing.compose);

    // --- Write GeoTIFFs + STAC item ----------------------------------------
    if !cli.no_write {
        let t = Instant::now();
        std::fs::create_dir_all(&cli.out_dir)
            .with_context(|| format!("mkdir {}", cli.out_dir.display()))?;
        let geo_params = |nodata: u16| bestpixel::writer::GeoTiffParams {
            width: grid.width,
            height: grid.height,
            tile_size: 256,
            epsg: grid.epsg,
            origin: [grid.bounds[0], grid.bounds[3]],
            pixel_size: grid.resolution,
            nodata,
        };
        // Parallel writes across all 15 output bands. DEFLATE-tile
        // compression is CPU-heavy (~150 ms per band on this AOI); with
        // 24 cores rayon collapses ~2.3 s sequential wall to ~0.3 s.
        use rayon::prelude::*;
        // Output filenames are stable band names regardless of which
        // STAC endpoint we read from — so a "red.tif" produced via PC
        // is interchangeable with a "red.tif" produced via Element84
        // (and HLS, where the asset names differ but the harmonized
        // common-band names match).
        let band_paths: Vec<(std::path::PathBuf, usize)> = band_names_out
            .iter()
            .enumerate()
            .map(|(i, name)| (cli.out_dir.join(format!("{name}.tif")), i))
            .collect();
        band_paths
            .par_iter()
            .try_for_each(|(path, i)| -> anyhow::Result<()> {
                bestpixel::writer::write_uint16_geotiff(
                    path,
                    &composite.bands[*i],
                    geo_params(65535),
                )
                .with_context(|| format!("write {}", path.display()))
            })?;
        rayon::scope(|s| {
            s.spawn(|_| {
                let _ = bestpixel::writer::write_uint16_geotiff(
                    &cli.out_dir.join("quality.tif"),
                    &composite.quality,
                    geo_params(65535),
                );
            });
            s.spawn(|_| {
                let _ = bestpixel::writer::write_uint16_geotiff(
                    &cli.out_dir.join("observation_count.tif"),
                    &composite.observation_count,
                    geo_params(0),
                );
            });
            s.spawn(|_| {
                let _ = bestpixel::writer::write_int16_geotiff(
                    &cli.out_dir.join("selected_observation.tif"),
                    &composite.selected_observation,
                    geo_params(0xFFFF),
                );
            });
        });
        // STAC item JSON.
        let mut assets = serde_json::Map::new();
        for name in band_names_out.iter() {
            assets.insert(
                name.to_string(),
                serde_json::json!({
                    "href": format!("{name}.tif"),
                    "type": "image/tiff; application=geotiff"
                }),
            );
        }
        for aux in ["quality", "observation_count", "selected_observation"] {
            assets.insert(
                aux.to_string(),
                serde_json::json!({"href": format!("{aux}.tif"), "type": "image/tiff; application=geotiff"}),
            );
        }
        let partition_summary = partition.as_ref().map(|p| {
            serde_json::json!({
                "tiles": p.tiles,
                "multi_tile_chunk_count": p.requirements.values().filter(|r| r.required_tiles.len() > 1).count(),
                "unreachable_chunk_count": p.requirements.values().filter(|r| r.unreachable_pixel_fraction > 0.0).count(),
            })
        });
        let source_items: Vec<_> = composite
            .source_ids
            .iter()
            .map(|id| serde_json::json!({"source_id": id}))
            .collect();
        let stac_item = serde_json::json!({
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": format!("spx-rust-{}", cli.datetime.replace('/', "_")),
            "bbox": bbox,
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [bbox[0], bbox[1]], [bbox[2], bbox[1]],
                    [bbox[2], bbox[3]], [bbox[0], bbox[3]], [bbox[0], bbox[1]],
                ]],
            },
            "properties": {
                "datetime": cli.datetime,
                "proj:epsg": grid.epsg,
                "proj:bbox": grid.bounds,
                "proj:shape": [grid.height, grid.width],
                "proj:transform": grid.affine_transform(),
                "surface:compositor": "rust_best_pixel_v1",
                "surface:band_names": band_names_out.clone(),
                "surface:stac_url": &endpoint.stac_url,
                "surface:collection": &collections_key,
                "surface:source_count": composite.source_ids.len(),
                "surface:source_items": source_items,
                "surface:partition": partition_summary,
            },
            "assets": assets,
        });
        std::fs::write(
            cli.out_dir.join("stac-item.json"),
            serde_json::to_string_pretty(&stac_item)?,
        )
        .with_context(|| format!("write stac-item.json"))?;

        timing.write = t.elapsed().as_secs_f64();
        eprintln!(
            "write:          {:6.2}s  → {}",
            timing.write,
            cli.out_dir.display()
        );
    }

    timing.total = t_total.elapsed().as_secs_f64();
    eprintln!("───────────────────────────");
    eprintln!("TOTAL:          {:6.2}s", timing.total);

    let report = serde_json::json!({
        "grid": {
            "epsg": grid.epsg,
            "bounds": grid.bounds,
            "size": [grid.width, grid.height],
        },
        "scenes": items.len(),
        "scored": stats_map.len(),
        "observations": composite.source_ids.len(),
        "timing": {
            "list_scenes": timing.list_scenes,
            "scout": timing.scout,
            "fetch": timing.fetch,
            "compose": timing.compose,
            "write": timing.write,
            "total": timing.total,
        },
    });
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}
