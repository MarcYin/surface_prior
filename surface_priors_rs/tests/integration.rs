//! Integration tests that don't hit the network.
//!
//! Covers the COG reader's predictor inversion, grid math, and tile
//! classifier — the pieces that have to match Python's behaviour
//! exactly. STAC + HTTP code paths are validated by the live
//! verification script in CI, not here.

use surface_priors_rs::grid::GridSpec;
use surface_priors_rs::projx;
use surface_priors_rs::tile_classification::{build_partition, chunks_from_grid};

#[test]
fn grid_from_wgs84_matches_python_snapped_bounds() {
    // AOI used by every verification run in the Python pipeline.
    let grid = GridSpec::from_wgs84_bounds([30.5, 30.5, 31.6, 31.5], 60.0);
    assert_eq!(grid.epsg, 32636);
    // Python's PROJ output: bounds = (260040, 3375000, 367080, 3487740),
    // size = 1784 × 1879. Match exact pixel counts.
    assert_eq!(grid.bounds, [260040.0, 3375000.0, 367080.0, 3487740.0]);
    assert_eq!(grid.width, 1784);
    assert_eq!(grid.height, 1879);
}

#[test]
fn proj_transform_round_trips_within_pixel() {
    // Cairo-ish point.
    let (x, y) = projx::transform_point("EPSG:4326", "EPSG:32636", 31.0, 30.5).expect("transform");
    let (lon, lat) = projx::transform_point("EPSG:32636", "EPSG:4326", x, y).expect("invert");
    assert!((lon - 31.0).abs() < 1e-6, "lon round-trip {lon}");
    assert!((lat - 30.5).abs() < 1e-6, "lat round-trip {lat}");
}

#[test]
fn chunks_from_grid_partition_into_512_blocks() {
    let chunks = chunks_from_grid([0.0, 0.0, 1024.0, 1024.0], 1.0, (1024, 1024), 512);
    // 2 × 2 layout.
    assert_eq!(chunks.len(), 4);
    assert_eq!(chunks[0].chunk_id, 0);
    assert_eq!(chunks[0].bounds_utm, [0.0, 512.0, 512.0, 1024.0]); // top-left
    assert_eq!(chunks[3].bounds_utm, [512.0, 0.0, 1024.0, 512.0]); // bottom-right
}

#[test]
fn classify_seam_chunks_as_multi_tile() {
    // Use a real UTM-zone grid; tile geometries given in WGS84 will be
    // reprojected via PROJ to that CRS, matching the production path.
    // AOI: 30..31°E, 30..31°N → roughly the same Egyptian region the
    // verification AOI uses.
    let grid = GridSpec::from_wgs84_bounds([30.0, 30.0, 31.0, 31.0], 60.0);
    let chunks = chunks_from_grid(grid.bounds, grid.resolution, (grid.width, grid.height), 512);
    // Two tile polygons: T covers the western 60%, U covers the
    // eastern 60%, in WGS84 lon/lat.
    let tile_t = serde_json::json!({
        "type": "Polygon",
        "coordinates": [[
            [29.0, 29.0], [30.6, 29.0], [30.6, 32.0], [29.0, 32.0], [29.0, 29.0]
        ]],
    });
    let tile_u = serde_json::json!({
        "type": "Polygon",
        "coordinates": [[
            [30.4, 29.0], [32.0, 29.0], [32.0, 32.0], [30.4, 32.0], [30.4, 29.0]
        ]],
    });
    let scenes = vec![
        (0usize, "T".to_string(), tile_t),
        (1usize, "U".to_string(), tile_u),
    ];
    let pixel_area = grid.resolution * grid.resolution;
    let part = build_partition(&chunks, &grid.proj_def(), &scenes, 1, pixel_area)
        .unwrap()
        .unwrap();
    // At least one chunk must be classified as multi-tile.
    let multi_tile = part
        .requirements
        .values()
        .filter(|r| r.required_tiles.len() > 1)
        .count();
    assert!(multi_tile >= 1, "expected at least one seam chunk; got 0");
    // No chunk should be unreachable on this fully-covered AOI.
    let unreachable = part
        .requirements
        .values()
        .filter(|r| r.unreachable_pixel_fraction > 0.001)
        .count();
    assert_eq!(unreachable, 0, "AOI should be fully reachable");
}

#[test]
fn classify_pure_overlap_chunk_picks_one_tile() {
    let grid = GridSpec::from_wgs84_bounds([30.0, 30.0, 30.5, 30.5], 60.0);
    let chunks = chunks_from_grid(grid.bounds, grid.resolution, (grid.width, grid.height), 4096);
    // Both tiles fully cover the AOI.
    let tile_t = serde_json::json!({
        "type": "Polygon",
        "coordinates": [[
            [29.9, 29.9], [30.6, 29.9], [30.6, 30.6], [29.9, 30.6], [29.9, 29.9]
        ]],
    });
    let tile_u = serde_json::json!({
        "type": "Polygon",
        "coordinates": [[
            [29.9, 29.9], [30.6, 29.9], [30.6, 30.6], [29.9, 30.6], [29.9, 29.9]
        ]],
    });
    let scenes = vec![
        (0usize, "T".to_string(), tile_t),
        (1usize, "U".to_string(), tile_u),
    ];
    let pixel_area = grid.resolution * grid.resolution;
    let part = build_partition(&chunks, &grid.proj_def(), &scenes, 1, pixel_area)
        .unwrap()
        .unwrap();
    assert_eq!(part.requirements[&0].required_tiles.len(), 1);
    assert!(part.requirements[&0].unreachable_pixel_fraction < 0.001);
}
