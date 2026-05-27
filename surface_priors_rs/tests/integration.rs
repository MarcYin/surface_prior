//! Integration tests that don't hit the network.
//!
//! Covers the COG reader's predictor inversion, grid math, and tile
//! classifier — the pieces that have to match Python's behaviour
//! exactly. STAC + HTTP code paths are validated by the live
//! verification script in CI, not here.

use std::collections::HashMap;
use bestpixel::endpoint::QualityKind;
use bestpixel::grid::GridSpec;
use bestpixel::pipeline::{
    per_chunk_clear, per_chunk_masks, s2_boa_offset, select_adaptive, ChunkMask, SceneStats,
    CHUNK_MASK_DIM,
};
use bestpixel::projx;
use bestpixel::stac::StacItem;
use bestpixel::tile_classification::{build_partition, chunks_from_grid};

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

// --- Adaptive-depth selection --------------------------------------------

/// Minimal STAC item carrying only the id `select_adaptive` looks up.
fn item(id: &str) -> StacItem {
    item_tile(id, "")
}

/// Like [`item`] but with an MGRS tile, for tile-aware selection tests.
fn item_tile(id: &str, mgrs_tile: &str) -> StacItem {
    StacItem {
        id: id.to_string(),
        datetime: String::new(),
        mgrs_tile: mgrs_tile.to_string(),
        geometry: serde_json::Value::Null,
        assets: HashMap::new(),
        properties: serde_json::Value::Null,
        collection: String::new(),
    }
}

const MASK_CELLS: usize = CHUNK_MASK_DIM * CHUNK_MASK_DIM;
const MASK_BYTES: usize = MASK_CELLS / 8;

fn popcount(bits: &[u8]) -> u32 {
    bits.iter().map(|b| b.count_ones()).sum()
}

/// A chunk mask with `valid` cells observed and `clear` cells clear,
/// each given as a half-open cell range.
fn cmask(valid: std::ops::Range<usize>, clear: std::ops::Range<usize>) -> ChunkMask {
    let set = |range: std::ops::Range<usize>| {
        let mut b = vec![0u8; MASK_BYTES];
        for c in range {
            b[c / 8] |= 1 << (c % 8);
        }
        b
    };
    ChunkMask { clear: set(clear), valid: set(valid) }
}

/// Whole chunk observed and fully clear.
fn full() -> ChunkMask {
    cmask(0..MASK_CELLS, 0..MASK_CELLS)
}

/// Whole chunk observed, nothing clear.
fn cloudy() -> ChunkMask {
    cmask(0..MASK_CELLS, 0..0)
}

/// Scene stats from per-chunk masks; chunk_clear/usable derive from them.
fn stats(id: &str, masks: Vec<ChunkMask>) -> SceneStats {
    let chunk_clear: Vec<f32> = masks
        .iter()
        .map(|m| {
            let v = popcount(&m.valid);
            if v == 0 { 0.0 } else { popcount(&m.clear) as f32 / v as f32 }
        })
        .collect();
    SceneStats {
        item_id: id.to_string(),
        usable_fraction: chunk_clear.iter().copied().fold(0.0, f32::max),
        mean_clear: 0.5,
        chunk_clear,
        chunk_masks: masks,
    }
}

fn stats_map(entries: &[SceneStats]) -> HashMap<String, SceneStats> {
    entries.iter().map(|s| (s.item_id.clone(), s.clone())).collect()
}

#[test]
fn adaptive_takes_min_k_for_redundancy() {
    // Three fully-clear scenes: one already covers the chunk, but min_k=2
    // forces a second for best-pixel redundancy — and no more.
    let scenes: Vec<StacItem> = (0..3).map(|i| item(&format!("s{i}"))).collect();
    let sm = stats_map(
        &(0..3).map(|i| stats(&format!("s{i}"), vec![full()])).collect::<Vec<_>>(),
    );
    let picks = select_adaptive(&scenes, &sm, None, 1, 0.95, 2, 8);
    assert_eq!(picks.len(), 2, "covered chunk stops at min_k");
}

#[test]
fn adaptive_stacks_complementary_scenes_until_covered() {
    // A clears the left half, B the right half, C duplicates A's left.
    // Union of A+B covers the whole chunk, so depth is 2 — and the
    // redundant C is never pulled because it adds no new clear cells.
    let scenes = vec![item("A"), item("B"), item("C")];
    let sm = stats_map(&[
        stats("A", vec![cmask(0..MASK_CELLS, 0..MASK_CELLS / 2)]),
        stats("B", vec![cmask(0..MASK_CELLS, MASK_CELLS / 2..MASK_CELLS)]),
        stats("C", vec![cmask(0..MASK_CELLS, 0..MASK_CELLS / 2)]),
    ]);
    let picks = select_adaptive(&scenes, &sm, None, 1, 0.95, 1, 8);
    let mut ids: Vec<&str> = picks.iter().map(|p| p.scene.id.as_str()).collect();
    ids.sort_unstable();
    assert_eq!(ids, vec!["A", "B"], "complementary pair covers; dup skipped");
}

#[test]
fn adaptive_caps_at_max_k_when_target_unreachable() {
    // Four scenes each clearing a distinct 10% slice — the union never
    // reaches the 0.95 target, so depth is bounded by max_k=3, not k=4.
    let slice = MASK_CELLS / 10;
    let scenes: Vec<StacItem> = (0..4).map(|i| item(&format!("s{i}"))).collect();
    let sm = stats_map(
        &(0..4)
            .map(|i| stats(&format!("s{i}"), vec![cmask(0..MASK_CELLS, i * slice..(i + 1) * slice)]))
            .collect::<Vec<_>>(),
    );
    let picks = select_adaptive(&scenes, &sm, None, 1, 0.95, 1, 3);
    assert_eq!(picks.len(), 3, "thin chunk pulls up to max_k");
}

#[test]
fn adaptive_prefers_full_observer_over_clearer_partial() {
    // F observes the whole chunk but is only 50% clear; P observes just
    // 60% of it but is fully clear there (60% clear cells — a higher
    // marginal gain). With room for one scene, completeness wins: F is
    // picked, not the clearer-but-partial P.
    let p_obs = MASK_CELLS * 6 / 10;
    let scenes = vec![item("F"), item("P")];
    let sm = stats_map(&[
        stats("F", vec![cmask(0..MASK_CELLS, 0..MASK_CELLS / 2)]),
        stats("P", vec![cmask(0..p_obs, 0..p_obs)]),
    ]);
    let picks = select_adaptive(&scenes, &sm, None, 1, 0.4, 1, 1);
    assert_eq!(picks.len(), 1);
    assert_eq!(picks[0].scene.id, "F", "full observer preferred over clearer partial");
}

#[test]
fn adaptive_dedups_scenes_and_carries_requesting_chunks() {
    // A observes both chunks; B observes only chunk 0. With min_k=2 both
    // are taken for chunk 0, and only A for chunk 1 (B doesn't observe
    // it). A appears once carrying both chunk ids; B once carrying [0].
    let scenes = vec![item("A"), item("B")];
    let sm = stats_map(&[
        stats("A", vec![full(), full()]),
        stats("B", vec![full(), ChunkMask::default()]),
    ]);
    let picks = select_adaptive(&scenes, &sm, None, 2, 0.5, 2, 8);
    assert_eq!(picks.len(), 2);
    // BTreeMap over scene index keeps A (idx 0) before B (idx 1).
    assert_eq!(picks[0].scene.id, "A");
    assert_eq!(picks[0].chunk_ids, vec![0, 1]);
    assert_eq!(picks[1].scene.id, "B");
    assert_eq!(picks[1].chunk_ids, vec![0]);
}

#[test]
fn adaptive_covers_each_tile_on_a_seam_chunk() {
    // Chunk 0 straddles tiles T and U: T scenes observe+clear the left
    // half, U scenes the right half. The tile-aware path takes min_k=2
    // per required tile, so all four scenes are selected (both halves get
    // redundancy) — the fix for the Level-2 seam under-coverage.
    use std::collections::BTreeMap;
    use bestpixel::tile_classification::{ChunkTileRequirement, TilePartition};

    let half = MASK_CELLS / 2;
    let scenes = vec![
        item_tile("T1", "T"),
        item_tile("T2", "T"),
        item_tile("U1", "U"),
        item_tile("U2", "U"),
    ];
    let sm = stats_map(&[
        stats("T1", vec![cmask(0..half, 0..half)]),
        stats("T2", vec![cmask(0..half, 0..half)]),
        stats("U1", vec![cmask(half..MASK_CELLS, half..MASK_CELLS)]),
        stats("U2", vec![cmask(half..MASK_CELLS, half..MASK_CELLS)]),
    ]);

    let mut requirements = BTreeMap::new();
    requirements.insert(
        0u32,
        ChunkTileRequirement {
            chunk_id: 0,
            required_tiles: vec!["T".to_string(), "U".to_string()],
            unreachable_pixel_fraction: 0.0,
        },
    );
    let partition = TilePartition {
        requirements,
        scene_to_tile: BTreeMap::new(),
        tiles: vec!["T".to_string(), "U".to_string()],
    };
    let picks = select_adaptive(&scenes, &sm, Some(&partition), 1, 0.95, 2, 8);
    let mut ids: Vec<&str> = picks.iter().map(|p| p.scene.id.as_str()).collect();
    ids.sort_unstable();
    assert_eq!(ids, vec!["T1", "T2", "U1", "U2"], "min_k per tile on the seam");
}

#[test]
fn adaptive_skips_scenes_without_usable_stats() {
    // C has no stats entry, D's mean_clear is NaN, E observes but is fully
    // cloudy — only F contributes clear cells, so only F is selected.
    let scenes = vec![item("C"), item("D"), item("E"), item("F")];
    let mut d = stats("D", vec![full()]);
    d.mean_clear = f32::NAN;
    let sm = stats_map(&[d, stats("E", vec![cloudy()]), stats("F", vec![full()])]);
    let picks = select_adaptive(&scenes, &sm, None, 1, 0.5, 1, 8);
    assert_eq!(picks.len(), 1);
    assert_eq!(picks[0].scene.id, "F");
}

#[test]
fn s2_boa_offset_applies_only_for_n0400_plus() {
    let s2 = |id: &str, baseline: Option<&str>| {
        let mut it = item_tile(id, "T36RTV");
        it.collection = "sentinel-2-l2a".to_string();
        it.properties = match baseline {
            Some(b) => serde_json::json!({ "s2:processing_baseline": b }),
            None => serde_json::Value::Null,
        };
        it
    };
    // Property-driven: ≥ 04.00 gets the +1000 offset, earlier does not.
    assert_eq!(s2_boa_offset(&s2("x", Some("05.00"))), 1000);
    assert_eq!(s2_boa_offset(&s2("x", Some("04.00"))), 1000);
    assert_eq!(s2_boa_offset(&s2("x", Some("02.14"))), 0);
    // Id-token fallback when the property is absent.
    assert_eq!(s2_boa_offset(&s2("S2B_MSIL2A_20220725T083609_N0400_R064_T36RTV_x", None)), 1000);
    assert_eq!(s2_boa_offset(&s2("S2A_MSIL2A_20200701T083601_N0214_R064_T36RTV_x", None)), 0);
    // Non-S2 products carry no offset regardless.
    let mut hls = item_tile("HLS.S30.x", "T36RTV");
    hls.collection = "hls2-s30".to_string();
    assert_eq!(s2_boa_offset(&hls), 0);
}

#[test]
fn per_chunk_masks_marks_clear_and_nodata() {
    // 64×64 single chunk. All-clear SCL → every mask cell valid + clear.
    let clear_buf = vec![4u8; 64 * 64];
    let m = &per_chunk_masks(&clear_buf, 64, 64, 64, QualityKind::Scl)[0];
    assert_eq!(popcount(&m.valid), MASK_CELLS as u32, "all cells observed");
    assert_eq!(popcount(&m.clear), MASK_CELLS as u32, "all cells clear");

    // All-nodata SCL (0) → unobserved chunk, empty mask.
    let nodata_buf = vec![0u8; 64 * 64];
    let m = &per_chunk_masks(&nodata_buf, 64, 64, 64, QualityKind::Scl)[0];
    assert!(m.valid.is_empty(), "nodata chunk yields empty mask");
}

#[test]
fn per_chunk_clear_bins_row_major_blocks() {
    // 4×2 SCL buffer, 2-px chunks → a 2×1 chunk layout (chunks 0 and 1).
    // SCL clear classes are {4,5,6,11}; 0 is nodata; 9 is neither.
    //   row 0: [4, 4, 4, 0]
    //   row 1: [5, 0, 0, 0]
    // chunk 0 (cols 0-1): 4,4,5,0 → 3 clear / 4 = 0.75
    // chunk 1 (cols 2-3): 4,0,0,0 → 1 clear / 4 = 0.25
    let buf: [u8; 8] = [4, 4, 4, 0, 5, 0, 0, 0];
    let out = per_chunk_clear(&buf, 4, 2, 2, QualityKind::Scl);
    assert_eq!(out.len(), 2);
    assert!((out[0] - 0.75).abs() < 1e-6, "chunk 0 = {}", out[0]);
    assert!((out[1] - 0.25).abs() < 1e-6, "chunk 1 = {}", out[1]);
}
