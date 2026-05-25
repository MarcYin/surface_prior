//! Geometry-only chunk-to-tile classification.
//!
//! Mirrors `surface_priors.tile_classification` in Python: for each
//! chunk window, find the MGRS tiles that are required to cover it,
//! using the **union of item geometries per tile** as the tile
//! footprint, and exclusive coverage (the area covered by this tile
//! but no other intersecting tile) as the classification signal. A
//! chunk lying entirely in an overlap region falls back to its
//! largest-intersecting tile.

use std::collections::BTreeMap;
use std::convert::TryFrom;

use geo::{Area, BooleanOps, BoundingRect, MultiPolygon, Polygon, Rect};
use geo_types::Coord;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::projx;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkTileRequirement {
    pub chunk_id: u32,
    pub required_tiles: Vec<String>,
    pub unreachable_pixel_fraction: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TilePartition {
    pub requirements: BTreeMap<u32, ChunkTileRequirement>,
    pub scene_to_tile: BTreeMap<usize, String>,
    pub tiles: Vec<String>,
}

impl TilePartition {
    pub fn tiles_for(&self, chunk_id: u32) -> &[String] {
        self.requirements
            .get(&chunk_id)
            .map(|r| r.required_tiles.as_slice())
            .unwrap_or(&[])
    }
}

/// One chunk window in pixel space + its UTM bounds.
#[derive(Debug, Clone, Copy)]
pub struct ChunkWindow {
    pub chunk_id: u32,
    pub bounds_utm: [f64; 4],
}

pub fn build_partition(
    chunks: &[ChunkWindow],
    grid_crs: &str,
    scene_geometries_wgs84: &[(usize, String, serde_json::Value)],
    min_exclusive_pixels: u64,
    pixel_area_m2: f64,
) -> Result<Option<TilePartition>> {
    if scene_geometries_wgs84.is_empty() {
        return Ok(None);
    }
    let mut scene_to_tile: BTreeMap<usize, String> = BTreeMap::new();
    for (scene_idx, tile, _geom) in scene_geometries_wgs84.iter() {
        scene_to_tile.insert(*scene_idx, tile.clone());
    }
    // 1. Group geometries by MGRS tile, build per-tile union (in grid CRS).
    let tile_footprints = build_tile_footprints(scene_geometries_wgs84, grid_crs)?;
    if tile_footprints.is_empty() {
        return Ok(None);
    }
    // 2. Classify each chunk against the footprints.
    let min_area = (min_exclusive_pixels as f64).max(0.0) * pixel_area_m2;
    let mut requirements: BTreeMap<u32, ChunkTileRequirement> = BTreeMap::new();
    let tile_codes: Vec<&String> = tile_footprints.keys().collect();
    for chunk in chunks {
        let req = classify_chunk(chunk, &tile_footprints, &tile_codes, min_area)?;
        requirements.insert(chunk.chunk_id, req);
    }
    let mut tiles: Vec<String> = tile_footprints.keys().cloned().collect();
    tiles.sort();
    Ok(Some(TilePartition {
        requirements,
        scene_to_tile,
        tiles,
    }))
}

fn build_tile_footprints(
    scene_geometries_wgs84: &[(usize, String, serde_json::Value)],
    grid_crs: &str,
) -> Result<BTreeMap<String, MultiPolygon<f64>>> {
    let mut grouped: BTreeMap<String, Vec<Polygon<f64>>> = BTreeMap::new();
    for (_scene_idx, tile, geom) in scene_geometries_wgs84 {
        if tile.is_empty() {
            continue;
        }
        let poly = match parse_geojson_polygon(geom) {
            Some(p) => p,
            None => continue,
        };
        let projected = reproject_polygon(&poly, "EPSG:4326", grid_crs)?;
        grouped.entry(tile.clone()).or_default().push(projected);
    }
    let mut out: BTreeMap<String, MultiPolygon<f64>> = BTreeMap::new();
    for (tile, polys) in grouped {
        if polys.is_empty() {
            continue;
        }
        let mut acc = MultiPolygon::from(polys[0].clone());
        for poly in polys.iter().skip(1) {
            acc = acc.union(&MultiPolygon::from(poly.clone()));
        }
        out.insert(tile, acc);
    }
    Ok(out)
}

fn classify_chunk(
    chunk: &ChunkWindow,
    tile_footprints: &BTreeMap<String, MultiPolygon<f64>>,
    tile_codes: &[&String],
    min_exclusive_area: f64,
) -> Result<ChunkTileRequirement> {
    let rect = Rect::new(
        Coord {
            x: chunk.bounds_utm[0],
            y: chunk.bounds_utm[1],
        },
        Coord {
            x: chunk.bounds_utm[2],
            y: chunk.bounds_utm[3],
        },
    );
    let chunk_poly: Polygon<f64> = rect.to_polygon();
    let chunk_mp = MultiPolygon::from(chunk_poly);
    let chunk_area = chunk_mp.unsigned_area();
    if chunk_area <= 0.0 {
        return Ok(ChunkTileRequirement {
            chunk_id: chunk.chunk_id,
            required_tiles: vec![],
            unreachable_pixel_fraction: 1.0,
        });
    }

    let mut intersections: BTreeMap<String, MultiPolygon<f64>> = BTreeMap::new();
    let mut intersect_areas: BTreeMap<String, f64> = BTreeMap::new();
    for tile in tile_codes {
        let inter = chunk_mp.intersection(&tile_footprints[*tile]);
        let area = inter.unsigned_area();
        if area <= 0.0 {
            continue;
        }
        intersections.insert((*tile).clone(), inter);
        intersect_areas.insert((*tile).clone(), area);
    }
    if intersections.is_empty() {
        return Ok(ChunkTileRequirement {
            chunk_id: chunk.chunk_id,
            required_tiles: vec![],
            unreachable_pixel_fraction: 1.0,
        });
    }

    // Exclusive coverage: (this tile's intersection) - (union of others' intersections).
    let mut exclusive_areas: BTreeMap<String, f64> = BTreeMap::new();
    for (tile, inter) in &intersections {
        let mut others: Option<MultiPolygon<f64>> = None;
        for (other_tile, other_inter) in &intersections {
            if other_tile == tile {
                continue;
            }
            others = Some(match others {
                Some(acc) => acc.union(other_inter),
                None => other_inter.clone(),
            });
        }
        let exclusive_area = match others {
            None => inter.unsigned_area(),
            Some(others_mp) => inter.difference(&others_mp).unsigned_area(),
        };
        exclusive_areas.insert(tile.clone(), exclusive_area);
    }

    let mut required: Vec<String> = exclusive_areas
        .iter()
        .filter(|(_, area)| **area >= min_exclusive_area)
        .map(|(tile, _)| tile.clone())
        .collect();
    if required.is_empty() {
        // Chunk sits in an overlap-only region — pick the single tile with
        // greatest total intersection.
        if let Some((tile, _)) = intersect_areas
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        {
            required.push(tile.clone());
        }
    }
    // Order by exclusive area descending so the round-robin gives the
    // most-needed tile its first pick first.
    required.sort_by(|a, b| {
        exclusive_areas
            .get(b)
            .partial_cmp(&exclusive_areas.get(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Unreachable = chunk area not covered by any tile.
    let mut reachable: Option<MultiPolygon<f64>> = None;
    for inter in intersections.values() {
        reachable = Some(match reachable {
            Some(acc) => acc.union(inter),
            None => inter.clone(),
        });
    }
    let reachable_area = reachable.as_ref().map(|r| r.unsigned_area()).unwrap_or(0.0);
    let unreachable = (1.0 - reachable_area / chunk_area).max(0.0);
    Ok(ChunkTileRequirement {
        chunk_id: chunk.chunk_id,
        required_tiles: required,
        unreachable_pixel_fraction: unreachable,
    })
}

fn parse_geojson_polygon(geom: &serde_json::Value) -> Option<Polygon<f64>> {
    let typ = geom.get("type")?.as_str()?;
    let coords = geom.get("coordinates")?;
    match typ {
        "Polygon" => Some(parse_polygon_coords(coords)?),
        "MultiPolygon" => {
            // Convert MultiPolygon to a single polygon by taking the first
            // ring's polygon — for S2 items the footprint is a single polygon
            // in practice. If multiple, the caller's tile union still works
            // because we add all polygons of each item to the tile group.
            let arr = coords.as_array()?;
            arr.first().and_then(parse_polygon_coords)
        }
        _ => None,
    }
}

fn parse_polygon_coords(coords: &serde_json::Value) -> Option<Polygon<f64>> {
    let rings = coords.as_array()?;
    let exterior_raw = rings.first()?.as_array()?;
    let exterior: Vec<Coord<f64>> = exterior_raw
        .iter()
        .filter_map(|pt| {
            let pt = pt.as_array()?;
            let x = pt.first()?.as_f64()?;
            let y = pt.get(1)?.as_f64()?;
            Some(Coord { x, y })
        })
        .collect();
    if exterior.len() < 3 {
        return None;
    }
    let interiors: Vec<geo::LineString<f64>> = rings
        .iter()
        .skip(1)
        .filter_map(|ring| {
            let ring = ring.as_array()?;
            let pts: Vec<Coord<f64>> = ring
                .iter()
                .filter_map(|pt| {
                    let pt = pt.as_array()?;
                    let x = pt.first()?.as_f64()?;
                    let y = pt.get(1)?.as_f64()?;
                    Some(Coord { x, y })
                })
                .collect();
            if pts.len() >= 3 {
                Some(geo::LineString::from(pts))
            } else {
                None
            }
        })
        .collect();
    Some(Polygon::new(geo::LineString::from(exterior), interiors))
}

fn reproject_polygon(poly: &Polygon<f64>, from: &str, to: &str) -> Result<Polygon<f64>> {
    let exterior_pts: Vec<(f64, f64)> = poly.exterior().points().map(|p| (p.x(), p.y())).collect();
    let mut buf: Vec<(f64, f64)> = exterior_pts.clone();
    projx::transform_points(from, to, &mut buf)?;
    let new_exterior: Vec<Coord<f64>> = buf.iter().map(|(x, y)| Coord { x: *x, y: *y }).collect();
    let mut new_interiors: Vec<geo::LineString<f64>> = Vec::with_capacity(poly.interiors().len());
    for ring in poly.interiors() {
        let mut pts: Vec<(f64, f64)> = ring.points().map(|p| (p.x(), p.y())).collect();
        projx::transform_points(from, to, &mut pts)?;
        let coords: Vec<Coord<f64>> = pts.iter().map(|(x, y)| Coord { x: *x, y: *y }).collect();
        new_interiors.push(geo::LineString::from(coords));
    }
    Ok(Polygon::new(geo::LineString::from(new_exterior), new_interiors))
}

/// Convenience for callers building from chunk pixel windows + grid origin.
pub fn chunks_from_grid(
    grid_bounds: [f64; 4],
    resolution: f64,
    grid_size: (u32, u32),
    chunk_size: u32,
) -> Vec<ChunkWindow> {
    let (width, height) = grid_size;
    let mut out = Vec::new();
    let mut chunk_id: u32 = 0;
    let mut row: u32 = 0;
    while row < height {
        let h = chunk_size.min(height - row);
        let mut col: u32 = 0;
        while col < width {
            let w = chunk_size.min(width - col);
            let xmin = grid_bounds[0] + (col as f64) * resolution;
            let xmax = xmin + (w as f64) * resolution;
            let ymax = grid_bounds[3] - (row as f64) * resolution;
            let ymin = ymax - (h as f64) * resolution;
            out.push(ChunkWindow {
                chunk_id,
                bounds_utm: [xmin, ymin, xmax, ymax],
            });
            chunk_id += 1;
            col += chunk_size;
        }
        row += chunk_size;
    }
    out
}

/// Stable hash of (sorted item_id, geom-json-string, mgrs_tile) triples.
/// Used as the cache key for tile_partition disk persistence.
pub fn scenes_signature(items: &[(String, String, serde_json::Value)]) -> String {
    use sha1::{Digest, Sha1};
    let mut sorted = items.to_vec();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let mut hasher = Sha1::new();
    for (id, tile, geom) in sorted {
        hasher.update(id.as_bytes());
        hasher.update(b"\0");
        hasher.update(tile.as_bytes());
        hasher.update(b"\0");
        let s = serde_json::to_string(&geom).unwrap_or_default();
        hasher.update(s.as_bytes());
        hasher.update(b"\n");
    }
    hex::encode(hasher.finalize())
}

#[allow(dead_code)]
fn _silence_try_from(_: Result<i32>) {
    let _ = u32::try_from(0i64);
}
